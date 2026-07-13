"""
BigMHC Weight Sweep — pure cache replay, no recompute.

Question: at what weight does BigMHC's immunogenicity score stop hurting
the ensemble and start helping (if it ever does)?

Replays the cached layer scores from the v2 benchmark against:
  (A) A weight sweep: BigMHC's weight from 0.00 to 0.50 in 0.05 steps,
      with the other layers' weights rescaled to fill the gap.
  (B) A conditional gate test: only use BigMHC's score when binding AND
      presentation are both already strong. Matches BigMHC's training
      assumption (it was trained on peptides assumed already presented).

What we DON'T do: tune the winning weight then report metrics on the same
data. That fits to ITSNdb. We report the curve; choosing a final weight is
a separate decision that should be validated via CV (next step).

USAGE
-----
    # default: uses validation/ITSNdb.csv and the benchmark cache
    python -m validation.bigmhc_weight_sweep

    # optional: include structure (currently zero-signal, default off)
    python -m validation.bigmhc_weight_sweep --with-structure

    # optional: write a CSV of the curve
    python -m validation.bigmhc_weight_sweep --csv validation/bigmhc_sweep.csv
"""

import sys
import os
import csv
import json
import hashlib
import argparse
from pathlib import Path
from typing import List, Optional, Tuple, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

from validation.benchmark_itsndb import (
    load_itsndb,
    LayerScores,
    BenchmarkPeptide,
    CACHE_SCHEMA_VERSION,
    _auroc,
    _auprc,
    _precision_recall_at_k,
)


# ---------------------------------------------------------------------------
# Load cached scores (no recompute)
# ---------------------------------------------------------------------------

def load_cached_scores(peptides: List[BenchmarkPeptide],
                       cache_dir: Path) -> List[Tuple[BenchmarkPeptide, LayerScores]]:
    """
    Load LayerScores from the v2 benchmark cache. Skip any peptide whose
    cache entry is missing — we replay only what we have.
    """
    known = {f.name for f in LayerScores.__dataclass_fields__.values()}
    out = []
    missing = 0
    for bp in peptides:
        key = hashlib.md5(
            f"{CACHE_SCHEMA_VERSION}_{bp.peptide}_{bp.hla}".encode()
        ).hexdigest()
        path = cache_dir / f"{key}.json"
        if not path.exists():
            missing += 1
            continue
        try:
            data = json.load(open(path))
        except (json.JSONDecodeError, IOError):
            missing += 1
            continue
        scores = LayerScores(**{k: v for k, v in data.items() if k in known})
        out.append((bp, scores))
    if missing:
        print(f"⚠️  {missing} peptide(s) had no cached scores — skipped.")
    return out


# ---------------------------------------------------------------------------
# Sweep A: vary BigMHC weight, keep relative balance of other layers
# ---------------------------------------------------------------------------

def make_sweep_weights(bigmhc_weight: float,
                        include_structure: bool) -> Dict[str, float]:
    """
    Distribute the remaining (1 - bigmhc_weight) across the other layers.
    Default balance is the v1 ensemble's:
        binding 0.35, presentation 0.35, structure 0.15 (if on), literature 0.15
    rescaled to sum to (1 - bigmhc_weight).

    Literature OFF for honest sweep — that decision happens at the call site.
    """
    base = {"binding": 0.35, "presentation": 0.35}
    if include_structure:
        base["structure"] = 0.15
    # NOTE: literature deliberately not included — honest mode only
    total_base = sum(base.values())
    remaining = 1.0 - bigmhc_weight
    weights = {k: v / total_base * remaining for k, v in base.items()}
    weights["immunogenicity"] = bigmhc_weight
    return weights


def combine(scores: LayerScores, weights: Dict[str, float]) -> Optional[float]:
    """Same renormalized weighted combine as the benchmark."""
    total_w = 0.0
    acc = 0.0
    for key, w in weights.items():
        val = getattr(scores, key)
        if val is not None:
            acc += val * w
            total_w += w
    return acc / total_w if total_w > 0 else None


def evaluate(results, weights) -> Optional[Dict]:
    labels, scores = [], []
    for bp, s in results:
        v = combine(s, weights)
        if v is None:
            continue
        labels.append(bp.label)
        scores.append(v)
    if len(scores) < 5 or sum(labels) in (0, len(labels)):
        return None
    p10, r10 = _precision_recall_at_k(labels, scores, 10)
    p20, r20 = _precision_recall_at_k(labels, scores, 20)
    return {
        "n": len(scores),
        "auroc": _auroc(labels, scores),
        "auprc": _auprc(labels, scores),
        "p_at_10": p10,
        "p_at_20": p20,
    }


# ---------------------------------------------------------------------------
# Sweep B: conditional gating — only use BigMHC when binding+presentation strong
# ---------------------------------------------------------------------------

def evaluate_conditional(results,
                          binding_threshold: float,
                          presentation_threshold: float,
                          bigmhc_weight: float = 0.20) -> Optional[Dict]:
    """
    Only let BigMHC vote when the peptide already passes binding+presentation
    filters. Otherwise BigMHC is ignored and the ensemble uses binding+presentation
    at equal weights.

    This tests the hypothesis: BigMHC has signal but only in its training
    regime (already-presented peptides). Outside that regime it's noise.
    """
    labels, scores = [], []
    n_used_immuno = 0
    n_skipped_immuno = 0
    for bp, s in results:
        if s.binding is None or s.presentation is None:
            continue
        use_immuno = (
            s.binding >= binding_threshold
            and s.presentation >= presentation_threshold
            and s.immunogenicity is not None
        )
        if use_immuno:
            # binding + presentation + immuno (rescaled)
            other = 1.0 - bigmhc_weight
            w = {"binding": other / 2, "presentation": other / 2,
                 "immunogenicity": bigmhc_weight}
            n_used_immuno += 1
        else:
            w = {"binding": 0.5, "presentation": 0.5}
            n_skipped_immuno += 1
        v = combine(s, w)
        if v is None:
            continue
        labels.append(bp.label)
        scores.append(v)
    if len(scores) < 5 or sum(labels) in (0, len(labels)):
        return None
    p10, r10 = _precision_recall_at_k(labels, scores, 10)
    p20, r20 = _precision_recall_at_k(labels, scores, 20)
    return {
        "n": len(scores),
        "auroc": _auroc(labels, scores),
        "auprc": _auprc(labels, scores),
        "p_at_10": p10,
        "p_at_20": p20,
        "n_used_immuno": n_used_immuno,
        "n_skipped_immuno": n_skipped_immuno,
    }


# ---------------------------------------------------------------------------
# Reference baselines (for comparison rows)
# ---------------------------------------------------------------------------

def baseline_l1_binding_only(results):
    return evaluate(results, {"binding": 1.0})

def baseline_l2_bp(results):
    return evaluate(results, {"binding": 0.5, "presentation": 0.5})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="BigMHC weight sweep (cache replay)")
    ap.add_argument("--data", default="validation/ITSNdb.csv")
    ap.add_argument("--cache-dir", default=None,
                    help="Defaults to ~/.cache/neoantigens/benchmark")
    ap.add_argument("--with-structure", action="store_true",
                    help="Include structure layer (currently zero-signal; default off)")
    ap.add_argument("--csv", help="Optional: write the sweep curve to CSV")
    args = ap.parse_args()

    if not os.path.exists(args.data):
        print(f"❌ Data not found: {args.data}")
        sys.exit(1)

    cache_dir = Path(args.cache_dir) if args.cache_dir else (
        Path.home() / ".cache" / "neoantigens" / "benchmark"
    )
    if not cache_dir.exists():
        print(f"❌ Cache dir not found: {cache_dir}")
        print(f"   Run the v2 benchmark first to populate cache.")
        sys.exit(1)

    peptides = load_itsndb(args.data)
    results = load_cached_scores(peptides, cache_dir)

    if not results:
        print(f"❌ No cached scores found in {cache_dir}")
        sys.exit(1)

    n_pos = sum(bp.label for bp, _ in results)
    print("=" * 92)
    print("  BIGMHC WEIGHT SWEEP  (cache replay, no recompute)")
    print("=" * 92)
    print(f"  Peptides scored:  {len(results)}  ({n_pos} positive / {len(results)-n_pos} negative)")
    print(f"  Structure layer:  {'INCLUDED' if args.with_structure else 'EXCLUDED (zero-signal)'}")
    print(f"  Literature layer: EXCLUDED (honest mode)")
    print("=" * 92)

    # --- Baselines for comparison ---
    bl_l1 = baseline_l1_binding_only(results)
    bl_l2 = baseline_l2_bp(results)
    print(f"\n  Baselines:")
    print(f"    L1  binding only                  AUROC {bl_l1['auroc']:.3f}   "
          f"AUPRC {bl_l1['auprc']:.3f}   P@10 {bl_l1['p_at_10']:.2f}")
    print(f"    L2  binding + presentation        AUROC {bl_l2['auroc']:.3f}   "
          f"AUPRC {bl_l2['auprc']:.3f}   P@10 {bl_l2['p_at_10']:.2f}")

    # --- Sweep A: vary BigMHC weight 0.00 -> 0.50 ---
    print(f"\n  Sweep A — vary BigMHC weight (other layers rescaled to fill)")
    print(f"  ───────────────────────────────────────────────────────────────────")
    print(f"  {'bigmhc_w':>10}  {'AUROC':>7}  {'AUPRC':>7}  {'P@10':>5}  {'P@20':>5}  {'Δ vs L2':>8}")
    print(f"  ───────────────────────────────────────────────────────────────────")

    sweep_rows = []
    best = None
    for w_int in range(0, 51, 5):
        w = w_int / 100.0
        weights = make_sweep_weights(w, include_structure=args.with_structure)
        m = evaluate(results, weights)
        if m is None:
            continue
        delta = m["auroc"] - bl_l2["auroc"]
        marker = "  ←" if best is None or m["auroc"] > best["auroc"] else ""
        if best is None or m["auroc"] > best["auroc"]:
            best = {"weight": w, **m}
        print(f"  {w:>10.2f}  {m['auroc']:>7.3f}  {m['auprc']:>7.3f}  "
              f"{m['p_at_10']:>5.2f}  {m['p_at_20']:>5.2f}  {delta:+.3f}{marker}")
        sweep_rows.append({"bigmhc_weight": w, **m})

    print(f"  ───────────────────────────────────────────────────────────────────")
    if best:
        print(f"\n  📈 Best AUROC at BigMHC weight = {best['weight']:.2f}: "
              f"{best['auroc']:.3f}  (vs L2 baseline {bl_l2['auroc']:.3f})")

    # --- Sweep B: conditional gating ---
    print(f"\n  Sweep B — conditional gating (BigMHC only when bind+pres pass thresholds)")
    print(f"  ─────────────────────────────────────────────────────────────────────────")
    print(f"  {'bind_thr':>10}  {'pres_thr':>10}  {'AUROC':>7}  {'AUPRC':>7}  "
          f"{'P@10':>5}  {'used/skip':>11}")
    print(f"  ─────────────────────────────────────────────────────────────────────────")

    cond_rows = []
    best_cond = None
    # Pick thresholds spanning the realistic range we see in cached data
    for bt in (0.40, 0.50, 0.60, 0.70):
        for pt in (0.30, 0.50, 0.70):
            m = evaluate_conditional(results, bt, pt, bigmhc_weight=0.20)
            if m is None:
                continue
            usage = f"{m['n_used_immuno']}/{m['n_skipped_immuno']}"
            marker = "  ←" if best_cond is None or m["auroc"] > best_cond["auroc"] else ""
            if best_cond is None or m["auroc"] > best_cond["auroc"]:
                best_cond = {"bt": bt, "pt": pt, **m}
            print(f"  {bt:>10.2f}  {pt:>10.2f}  {m['auroc']:>7.3f}  {m['auprc']:>7.3f}  "
                  f"{m['p_at_10']:>5.2f}  {usage:>11}{marker}")
            cond_rows.append({"binding_thr": bt, "presentation_thr": pt, **m})

    print(f"  ─────────────────────────────────────────────────────────────────────────")
    if best_cond:
        print(f"\n  📈 Best conditional AUROC: "
              f"binding_thr={best_cond['bt']:.2f}, presentation_thr={best_cond['pt']:.2f}, "
              f"AUROC {best_cond['auroc']:.3f}")
        print(f"     BigMHC voted on {best_cond['n_used_immuno']} peptides, "
              f"ignored on {best_cond['n_skipped_immuno']}.")

    # --- Verdict ---
    print(f"\n{'='*92}")
    print(f"  READING THIS RESULT")
    print(f"{'='*92}")

    if best is None:
        print("  No sweep results — cache may be empty.")
        return

    l2_baseline = bl_l2["auroc"]
    best_w_auroc = best["auroc"]
    bigmhc_at_zero = sweep_rows[0]["auroc"] if sweep_rows else None
    bigmhc_at_30 = next((r["auroc"] for r in sweep_rows if abs(r["bigmhc_weight"] - 0.30) < 0.01), None)

    print(f"  • L2 baseline (no BigMHC, no structure):       AUROC {l2_baseline:.3f}")
    if bigmhc_at_zero is not None:
        print(f"  • Sweep at BigMHC weight = 0  (no immuno):     AUROC {bigmhc_at_zero:.3f}")
    if bigmhc_at_30 is not None:
        print(f"  • Sweep at BigMHC weight = 0.30 (original):    AUROC {bigmhc_at_30:.3f}")
    print(f"  • Best sweep weight  ({best['weight']:.2f}):                  AUROC {best_w_auroc:.3f}")
    if best_cond:
        print(f"  • Best conditional:                            AUROC {best_cond['auroc']:.3f}  "
              f"(BigMHC used on {best_cond['n_used_immuno']}/{best_cond['n_used_immuno']+best_cond['n_skipped_immuno']})")

    print()
    if best_w_auroc <= l2_baseline + 0.005:
        print("  → BigMHC adds no measurable lift even at its best weight on ITSNdb.")
        print("    Honest move: drop it from the ensemble for this dataset class.")
        print("    Keep it in reports (Claude can reason about it qualitatively).")
    elif best["weight"] < 0.15:
        print(f"  → BigMHC has signal but is overweighted at 0.30. Optimal ~{best['weight']:.2f}.")
        print("    Honest move: lower its weight, but validate the chosen number with CV next.")
    else:
        print("  → BigMHC at higher weights actually helps. Keep weight around best.")
    print(f"\n  ⚠️  This sweep was on ITSNdb. Choosing a final weight from this curve")
    print(f"     would fit to ITSNdb. The right next step is 5-fold CV on this same")
    print(f"     cache to estimate weight + uncertainty without test-set contamination.")

    # --- Optional CSV ---
    if args.csv:
        out_path = Path(args.csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["sweep", "bigmhc_weight", "binding_thr",
                             "presentation_thr", "n", "auroc", "auprc",
                             "p_at_10", "p_at_20", "n_used_immuno", "n_skipped_immuno"])
            for r in sweep_rows:
                writer.writerow(["A", r["bigmhc_weight"], "", "",
                                 r["n"], f"{r['auroc']:.4f}", f"{r['auprc']:.4f}",
                                 f"{r['p_at_10']:.3f}", f"{r['p_at_20']:.3f}", "", ""])
            for r in cond_rows:
                writer.writerow(["B", "", r["binding_thr"], r["presentation_thr"],
                                 r["n"], f"{r['auroc']:.4f}", f"{r['auprc']:.4f}",
                                 f"{r['p_at_10']:.3f}", f"{r['p_at_20']:.3f}",
                                 r["n_used_immuno"], r["n_skipped_immuno"]])
        print(f"\n  💾 Sweep curve saved: {out_path}")


if __name__ == "__main__":
    main()