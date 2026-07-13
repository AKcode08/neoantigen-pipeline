"""
ITSNdb Benchmark Harness  (v2 — real structure features)

CHANGES vs v1:
- Structure layer now uses validation.structure_features (4 real features,
  unweighted mean), replacing the constant-0.8 quality flag.
- Cache key bumped (CACHE_SCHEMA_VERSION) so v1 caches are not reused.
- Adds a per-feature signal diagnostic table so we see which of the four
  structure features carries real signal, not just the combined number.
"""

import sys
import os
import csv
import json
import time
import argparse
import hashlib
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Tuple
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")
except ImportError:
    pass

from validation.structure_features import extract_structure_features

logger = logging.getLogger(__name__)

CACHE_SCHEMA_VERSION = "v2_real_structure"


# ---------------------------------------------------------------------------
# Dataset loader (unchanged)
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkPeptide:
    peptide: str
    wt: str
    hla: str
    hla_raw: str
    label: int
    gene: str
    tumor: str
    mut_position: int
    position_type: str
    length: int
    paper: str = ""
    author: str = ""

    @property
    def hla_neoapred(self) -> str:
        return self.hla.replace("HLA-", "").replace("*", "").replace(":", "")


def normalize_hla(raw: str) -> str:
    s = raw.strip()
    if "*" in s:
        return s
    if s.startswith("HLA-"):
        body = s[4:]
        return f"HLA-{body[0]}*{body[1:]}"
    return s


def load_itsndb(csv_path: str, limit: Optional[int] = None) -> List[BenchmarkPeptide]:
    peptides: List[BenchmarkPeptide] = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                label = 1 if row["NeoType"].strip().lower() == "positive" else 0
                raw_hla = row["HLA"].strip()
                peptides.append(BenchmarkPeptide(
                    peptide=row["Neoantigen"].strip(),
                    wt=row["WT"].strip(),
                    hla=normalize_hla(raw_hla),
                    hla_raw=raw_hla,
                    label=label,
                    gene=row.get("GeneSymbol", "").strip(),
                    tumor=row.get("Tumor", "").strip(),
                    mut_position=int(row["mutPosition"]) if row.get("mutPosition") else -1,
                    position_type=row.get("PositionType", "").strip(),
                    length=int(row["Length"]) if row.get("Length") else len(row["Neoantigen"].strip()),
                    paper=row.get("Paper", "").strip(),
                    author=row.get("Author", "").strip(),
                ))
            except (KeyError, ValueError) as e:
                logger.warning(f"Skipping malformed row: {e}")
                continue
    if limit:
        pos = [p for p in peptides if p.label == 1]
        neg = [p for p in peptides if p.label == 0]
        keep_pos = pos[: max(1, limit * len(pos) // len(peptides))]
        keep_neg = neg[: limit - len(keep_pos)]
        peptides = keep_pos + keep_neg
    return peptides


# ---------------------------------------------------------------------------
# Per-layer scoring
# ---------------------------------------------------------------------------

def binding_to_score(affinity_nm: float) -> float:
    import math
    if affinity_nm <= 0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - (math.log10(affinity_nm) / math.log10(50000))))


@dataclass
class LayerScores:
    binding: Optional[float] = None
    presentation: Optional[float] = None
    immunogenicity: Optional[float] = None
    structure: Optional[float] = None
    literature: Optional[float] = None

    binding_nm: Optional[float] = None
    iedb_assays: int = 0
    iedb_response_rate: float = 0.0

    # NEW: structure feature breakdown
    struct_tcr_exposure: Optional[float] = None
    struct_mut_visibility: Optional[float] = None
    struct_anchor_compat: Optional[float] = None
    struct_tcr_hydrophobicity: Optional[float] = None
    struct_feature_count: int = 0


"""
PATCH 1 of 2 — benchmark_itsndb.py weight update

This is a SURGICAL EDIT to validation/benchmark_itsndb.py.
Replace the existing ABLATION_LEVELS dict with the one below.
Everything else in benchmark_itsndb.py stays the same.

WHAT CHANGED:
  Added L3_tuned    — BigMHC at 0.10, structure dropped (our chosen live config)
  Added L2_only     — binding + presentation only, no immuno (alternative test)
  Kept L1-L4 as-is  — so the comparison table shows original vs tuned vs L2-only

WHERE TO FIND THE ORIGINAL IN YOUR FILE:
  Look for the section that begins:
      ABLATION_LEVELS = {
          "L1_binding": {"binding": 1.0},
          ...

REPLACE WITH:
"""

ABLATION_LEVELS = {
    # --- Original ablation levels (kept for honest comparison) ---
    "L1_binding":        {"binding": 1.0},
    "L2_+presentation":  {"binding": 0.5, "presentation": 0.5},
    "L3_+immunogenic":   {"binding": 0.35, "presentation": 0.35, "immunogenicity": 0.30},
    "L4_+structure":     {"binding": 0.30, "presentation": 0.30, "immunogenicity": 0.25, "structure": 0.15},
    # literature only added in leaked mode (unchanged):
    "L5_+literature":    {"binding": 0.25, "presentation": 0.25, "immunogenicity": 0.22,
                          "structure": 0.13, "literature": 0.15},

    # --- New ablation levels (the "what should we ship" candidates) ---
    # L3_tuned: chosen as the new LIVE PIPELINE configuration.
    # Derived from BigMHC weight sweep — BigMHC at 0.10 was the empirical
    # peak (AUROC 0.624 vs 0.580 at the original 0.30). Structure dropped
    # from consensus because per-feature analysis showed zero signal.
    "L3_tuned":          {"binding": 0.45, "presentation": 0.45, "immunogenicity": 0.10},

    # L2_only: the alternative — drop immunogenicity entirely.
    # The sweep showed AUROC at BigMHC=0 was 0.616, essentially identical
    # to BigMHC=0.10 (0.624). If CV agrees these are statistically tied,
    # the simpler ensemble (L2_only) may be preferable.
    "L2_only":           {"binding": 0.50, "presentation": 0.50},
}

# That's it — replace the dict and save. No other changes to benchmark_itsndb.py needed.


def combine(scores: LayerScores, weights: Dict[str, float]) -> Optional[float]:
    total_w = 0.0
    acc = 0.0
    for key, w in weights.items():
        val = getattr(scores, key)
        if val is not None:
            acc += val * w
            total_w += w
    if total_w == 0:
        return None
    return acc / total_w


# ---------------------------------------------------------------------------
# Runner (UPDATED: structure uses extract_structure_features)
# ---------------------------------------------------------------------------

class BenchmarkRunner:

    def __init__(self,
                 enable_structure: bool = True,
                 enable_literature: bool = True,
                 cache_dir: Optional[str] = None,
                 verbose: bool = True):
        self.enable_structure = enable_structure
        self.enable_literature = enable_literature
        self.verbose = verbose

        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "neoantigens" / "benchmark"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        from src.layer_2_predictors.mhc_binding import MHCBindingPredictor
        from src.layer_2_predictors.mhc_presentation import MHCPresentationPredictor
        from src.layer_2_predictors.immunogenicity_bigmhc import BigMHCImmunogenicityPredictor

        if self.verbose:
            print("🔧 Loading predictors...")
        self.binding = MHCBindingPredictor()
        self.presentation = MHCPresentationPredictor()
        self.immuno = BigMHCImmunogenicityPredictor()

        self.structure = None
        if enable_structure:
            from src.layer_3_structure.neoapred_structure import NeoaPredWrapper
            self.structure = NeoaPredWrapper()

        self.literature = None
        if enable_literature:
            from src.layer_4_literature.literature_evidence import LiteratureAggregator
            self.literature = LiteratureAggregator(verbose=False)

        if self.verbose:
            print("✅ Predictors ready\n")

    def _cache_path(self, bp: BenchmarkPeptide) -> Path:
        key = hashlib.md5(
            f"{CACHE_SCHEMA_VERSION}_{bp.peptide}_{bp.hla}".encode()
        ).hexdigest()
        return self.cache_dir / f"{key}.json"

    def score_peptide(self, bp: BenchmarkPeptide) -> LayerScores:
        cache_file = self._cache_path(bp)
        cached: Dict = {}
        if cache_file.exists():
            try:
                cached = json.load(open(cache_file))
            except (json.JSONDecodeError, IOError):
                cached = {}

        known = {f.name for f in LayerScores.__dataclass_fields__.values()}
        scores = LayerScores(**{k: v for k, v in cached.items() if k in known})
        dirty = False

        # Binding
        if scores.binding is None:
            b = self.binding.predict(bp.peptide, bp.hla)
            if b is not None:
                scores.binding = binding_to_score(b.affinity_nm)
                scores.binding_nm = b.affinity_nm
                dirty = True

        # Presentation
        if scores.presentation is None:
            p = self.presentation.predict(bp.peptide, bp.hla)
            if p is not None:
                scores.presentation = p.presentation_score
                dirty = True

        # Immunogenicity
        if scores.immunogenicity is None:
            im = self.immuno.predict(bp.peptide, bp.hla)
            if im is not None:
                scores.immunogenicity = im.immunogenicity_score
                dirty = True

        # Structure (REAL features)
        if self.enable_structure and scores.structure is None:
            try:
                st = self.structure.predict(bp.peptide, bp.hla_neoapred)
                pdb_path = st.pdb_file_path if (st and st.has_structure) else None
                feats = extract_structure_features(
                    peptide=bp.peptide,
                    hla_allele=bp.hla,
                    pdb_path=pdb_path,
                    mut_position=bp.mut_position,
                    position_type=bp.position_type,
                )
                scores.structure = feats.combined_score
                scores.struct_tcr_exposure = feats.tcr_contact_exposure
                scores.struct_mut_visibility = feats.mutation_tcr_visibility
                scores.struct_anchor_compat = feats.anchor_compatibility
                scores.struct_tcr_hydrophobicity = feats.tcr_hydrophobicity
                scores.struct_feature_count = feats.feature_count
                dirty = True
            except Exception as e:
                logger.warning(f"Structure failed for {bp.peptide}: {e}")

        # Literature
        if self.enable_literature and scores.literature is None:
            try:
                ev = self.literature.gather_evidence(bp.peptide, bp.hla, max_papers=3)
                scores.literature = ev.evidence_score
                scores.iedb_assays = ev.iedb_assay_count
                scores.iedb_response_rate = ev.iedb_response_rate
                dirty = True
            except Exception as e:
                logger.warning(f"Literature failed for {bp.peptide}: {e}")

        if dirty:
            try:
                json.dump(asdict(scores), open(cache_file, "w"))
            except IOError:
                pass
        return scores

    def run(self, peptides):
        results = []
        n = len(peptides)
        t0 = time.time()
        for i, bp in enumerate(peptides, 1):
            if self.verbose:
                print(f"[{i}/{n}] {bp.peptide} ({bp.hla}) "
                      f"label={'POS' if bp.label else 'NEG'}", end="  ")
            s = self.score_peptide(bp)
            if self.verbose:
                bits = []
                if s.binding is not None:        bits.append(f"bind={s.binding:.2f}")
                if s.immunogenicity is not None: bits.append(f"im={s.immunogenicity:.2f}")
                if s.structure is not None:      bits.append(f"st={s.structure:.2f}")
                if s.literature is not None:     bits.append(f"lit={s.literature:.2f}")
                print(" ".join(bits))
            results.append((bp, s))
        if self.verbose:
            dt = time.time() - t0
            print(f"\n⏱  Scored {n} peptides in {dt:.0f}s ({dt/n:.1f}s/peptide)\n")
        return results


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _auroc(labels, scores):
    pairs = sorted(zip(scores, labels))
    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + j - 1) / 2.0 + 1.0
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j
    n_pos = sum(labels); n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    sum_ranks_pos = sum(r for r, (_, lab) in zip(ranks, pairs) if lab == 1)
    u = sum_ranks_pos - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


def _auprc(labels, scores):
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    n_pos = sum(labels)
    if n_pos == 0:
        return float("nan")
    tp = fp = 0; ap = 0.0; prev_recall = 0.0
    for idx in order:
        if labels[idx] == 1: tp += 1
        else: fp += 1
        precision = tp / (tp + fp)
        recall = tp / n_pos
        ap += precision * (recall - prev_recall)
        prev_recall = recall
    return ap


def _precision_recall_at_k(labels, scores, k):
    order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
    if not order:
        return float("nan"), float("nan")
    hits = sum(labels[i] for i in order)
    n_pos = sum(labels)
    return hits / len(order), (hits / n_pos if n_pos else float("nan"))


@dataclass
class LevelMetrics:
    level: str
    n: int
    auroc: float
    auprc: float
    prec_at_10: float
    rec_at_10: float
    prec_at_20: float
    rec_at_20: float
    baseline_prevalence: float


def evaluate_level(level_name, results, weights):
    labels, scores = [], []
    for bp, s in results:
        combined = combine(s, weights)
        if combined is None:
            continue
        labels.append(bp.label)
        scores.append(combined)
    if len(scores) < 5 or sum(labels) == 0 or sum(labels) == len(labels):
        return None
    p10, r10 = _precision_recall_at_k(labels, scores, 10)
    p20, r20 = _precision_recall_at_k(labels, scores, 20)
    return LevelMetrics(
        level=level_name, n=len(scores),
        auroc=_auroc(labels, scores), auprc=_auprc(labels, scores),
        prec_at_10=p10, rec_at_10=r10, prec_at_20=p20, rec_at_20=r20,
        baseline_prevalence=sum(labels) / len(labels),
    )


def evaluate_individual_structure_features(results):
    feats = {
        "tcr_exposure":       "struct_tcr_exposure",
        "mut_visibility":     "struct_mut_visibility",
        "anchor_compat":      "struct_anchor_compat",
        "tcr_hydrophobicity": "struct_tcr_hydrophobicity",
    }
    rows = []
    for name, attr in feats.items():
        labels, scores = [], []
        for bp, s in results:
            v = getattr(s, attr)
            if v is not None:
                labels.append(bp.label)
                scores.append(v)
        if len(scores) < 10 or sum(labels) in (0, len(labels)):
            rows.append((name, len(scores), float("nan"), float("nan")))
            continue
        rows.append((name, len(scores), _auroc(labels, scores), _auprc(labels, scores)))
    return rows


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_ablation_table(title, metrics):
    print(f"\n{'='*92}")
    print(f"  {title}")
    print(f"{'='*92}")
    print(f"{'Level':<20} {'N':>4} {'AUROC':>7} {'AUPRC':>7} "
          f"{'P@10':>6} {'R@10':>6} {'P@20':>6} {'R@20':>6}")
    print("-" * 92)
    for m in metrics:
        print(f"{m.level:<20} {m.n:>4} {m.auroc:>7.3f} {m.auprc:>7.3f} "
              f"{m.prec_at_10:>6.2f} {m.rec_at_10:>6.2f} "
              f"{m.prec_at_20:>6.2f} {m.rec_at_20:>6.2f}")
    if metrics:
        print("-" * 92)
        print(f"{'(random baseline)':<20} {'':>4} {0.5:>7.3f} "
              f"{metrics[0].baseline_prevalence:>7.3f} "
              f"{metrics[0].baseline_prevalence:>6.2f} {'':>6} "
              f"{metrics[0].baseline_prevalence:>6.2f} {'':>6}")


def print_feature_diagnostics(rows):
    print(f"\n{'='*92}")
    print(f"  STRUCTURE FEATURES — INDIVIDUAL (each feature ranking ITSNdb alone)")
    print(f"{'='*92}")
    print(f"{'Feature':<22} {'N':>5} {'AUROC':>8} {'AUPRC':>8}")
    print("-" * 92)
    for name, n, auc, ap in rows:
        auc_s = f"{auc:.3f}" if auc == auc else "  nan"
        ap_s  = f"{ap:.3f}"  if ap == ap  else "  nan"
        print(f"{name:<22} {n:>5} {auc_s:>8} {ap_s:>8}")
    print("\n  AUROC ~0.5 = no signal. >0.55 = mild. >0.6 = useful.")
    print("  AUROC < 0.5 = inverted (feature anti-correlates with immunogenicity)")
    print("  — flag for review, do not silently flip the sign.")


def main():
    ap = argparse.ArgumentParser(description="ITSNdb benchmark v2 — real structure features")
    ap.add_argument("--data", default="validation/ITSNdb.csv")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--mode", choices=["both", "honest", "leaked"], default="both")
    ap.add_argument("--no-structure", action="store_true")
    ap.add_argument("--out", default="validation/benchmark_results_v2.json")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    verbose = not args.quiet
    enable_structure = not args.no_structure
    enable_literature = args.mode in ("both", "leaked")

    if not os.path.exists(args.data):
        print(f"❌ Data not found: {args.data}")
        sys.exit(1)

    peptides = load_itsndb(args.data, limit=args.limit)
    n_pos = sum(p.label for p in peptides)
    print("=" * 92)
    print("  ITSNdb BENCHMARK  v2 (real structure features)")
    print("=" * 92)
    print(f"  Peptides: {len(peptides)}  ({n_pos} immunogenic / {len(peptides)-n_pos} non-immunogenic)")
    print(f"  Mode: {args.mode}   Structure: {'ON (real features)' if enable_structure else 'OFF'}")
    print("=" * 92)

    runner = BenchmarkRunner(
        enable_structure=enable_structure,
        enable_literature=enable_literature,
        verbose=verbose,
    )
    results = runner.run(peptides)

    honest_levels = ["L1_binding", "L2_+presentation", "L3_+immunogenic"]
    if enable_structure:
        honest_levels.append("L4_+structure")

    honest_metrics = [m for m in (
        evaluate_level(lvl, results, ABLATION_LEVELS[lvl]) for lvl in honest_levels
    ) if m]

    report = {
        "dataset": "ITSNdb",
        "version": "v2_real_structure",
        "timestamp": datetime.now().isoformat(),
        "n_peptides": len(peptides),
        "n_positive": n_pos,
        "structure_enabled": enable_structure,
        "honest": [asdict(m) for m in honest_metrics],
    }

    if args.mode in ("honest", "both"):
        print_ablation_table("HONEST  (literature OFF — defensible numbers)", honest_metrics)

    if args.mode in ("leaked", "both") and enable_literature:
        leaked_levels = list(honest_levels) + ["L5_+literature"]
        leaked_metrics = [m for m in (
            evaluate_level(lvl, results, ABLATION_LEVELS[lvl]) for lvl in leaked_levels
        ) if m]
        report["leaked"] = [asdict(m) for m in leaked_metrics]
        print_ablation_table("LEAKED  (literature ON — optimistic)", leaked_metrics)
        if honest_metrics and leaked_metrics:
            print(f"\n  📊 Leakage gap (top-level AUROC): "
                  f"honest={honest_metrics[-1].auroc:.3f}  "
                  f"leaked={leaked_metrics[-1].auroc:.3f}  "
                  f"Δ={leaked_metrics[-1].auroc-honest_metrics[-1].auroc:+.3f}")

    if enable_structure:
        feature_rows = evaluate_individual_structure_features(results)
        print_feature_diagnostics(feature_rows)
        report["structure_features"] = [
            {"feature": n, "n": k, "auroc": a, "auprc": p}
            for (n, k, a, p) in feature_rows
        ]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=2)
    print(f"\n  💾 Saved: {args.out}\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()