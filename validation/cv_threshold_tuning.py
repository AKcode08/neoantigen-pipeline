"""
Cross-Validated Threshold Tuning — ITSNdb Benchmark

Replays the cached layer scores under 5-fold stratified CV to honestly estimate
threshold-dependent metrics (MCC, F1, precision/recall at clinical operating
points) without fitting to the test set.

WHAT THIS SOLVES
----------------
The earlier benchmark reported only ranking metrics (AUROC, AUPRC, P@K) which
don't depend on a decision threshold. The ensemble's actual recommendations
(INCLUDE/BORDERLINE/EXCLUDE) DO depend on thresholds (currently 0.7 / 0.3).
Those thresholds were set by intuition, never validated.

Tuning thresholds on ITSNdb and then reporting metrics on the same ITSNdb
peptides = fitting to the test set. The honest move is k-fold CV:
  • For each fold k:
      train threshold(s) τ on the other 4 folds
      apply τ to fold k
      compute MCC, F1, etc. on fold k (which τ never saw)
  • Average across folds. Report std too — the std is what tells us whether
    a difference between layers is real or fold-noise.

TWO MODES
---------
Mode A — Binary (single threshold τ):
  Predict positive if ensemble_score ≥ τ, else negative.
  τ tuned to maximize MCC on training folds.
  Reports: MCC, F1, accuracy, precision, recall, AUROC, AUPRC, P@10, P@20

Mode B — Ternary (clinical operating points):
  INCLUDE threshold τ_inc: optimized for INCLUDE-precision subject to a
    minimum recall floor (default 0.30) — so we don't degenerate to
    "INCLUDE the single highest-scored peptide" gaming the metric.
  EXCLUDE threshold τ_exc: optimized for negative-predictive-value (NPV)
    subject to a minimum negative-recall floor (default 0.30).
  Reports: INCLUDE-precision, INCLUDE-recall, EXCLUDE-NPV, EXCLUDE-recall,
           BORDERLINE-rate (fraction of peptides in the uncertain middle).

ABLATION LEVELS
---------------
L1 binding only / L2 +presentation / L3 +immunogenic / L4 +structure
(literature deliberately excluded — see HONEST mode in main benchmark)

USAGE
-----
    python -m validation.cv_threshold_tuning
    python -m validation.cv_threshold_tuning --folds 5 --min-recall 0.30
"""

import sys
import os
import csv
import json
import hashlib
import random
import argparse
import statistics
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Tuple, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

from validation.benchmark_itsndb import (
    load_itsndb,
    LayerScores,
    BenchmarkPeptide,
    CACHE_SCHEMA_VERSION,
    ABLATION_LEVELS,
    _auroc,
    _auprc,
    _precision_recall_at_k,
)


# ---------------------------------------------------------------------------
# 1. Cache loader (same as bigmhc_weight_sweep)
# ---------------------------------------------------------------------------

def load_cached_scores(peptides, cache_dir: Path):
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
        out.append((bp, LayerScores(**{k: v for k, v in data.items() if k in known})))
    if missing:
        print(f"⚠️  {missing} peptide(s) had no cached scores — skipped.")
    return out


def combine(scores: LayerScores, weights: Dict[str, float]) -> Optional[float]:
    total_w = acc = 0.0
    for key, w in weights.items():
        val = getattr(scores, key)
        if val is not None:
            acc += val * w
            total_w += w
    return acc / total_w if total_w > 0 else None


# ---------------------------------------------------------------------------
# 2. Threshold-dependent metrics
# ---------------------------------------------------------------------------

def confusion(labels: List[int], preds: List[int]) -> Tuple[int, int, int, int]:
    """Returns (tp, fp, tn, fn)."""
    tp = sum(1 for l, p in zip(labels, preds) if l == 1 and p == 1)
    fp = sum(1 for l, p in zip(labels, preds) if l == 0 and p == 1)
    tn = sum(1 for l, p in zip(labels, preds) if l == 0 and p == 0)
    fn = sum(1 for l, p in zip(labels, preds) if l == 1 and p == 0)
    return tp, fp, tn, fn


def mcc(labels: List[int], preds: List[int]) -> float:
    tp, fp, tn, fn = confusion(labels, preds)
    denom_sq = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if denom_sq == 0:
        return 0.0
    return (tp * tn - fp * fn) / (denom_sq ** 0.5)


def f1(labels: List[int], preds: List[int]) -> float:
    tp, fp, tn, fn = confusion(labels, preds)
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def precision_recall(labels: List[int], preds: List[int]) -> Tuple[float, float]:
    tp, fp, tn, fn = confusion(labels, preds)
    prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    rec = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    return prec, rec


def accuracy(labels: List[int], preds: List[int]) -> float:
    if not labels:
        return float("nan")
    return sum(1 for l, p in zip(labels, preds) if l == p) / len(labels)


def npv_and_neg_recall(labels: List[int], preds: List[int]) -> Tuple[float, float]:
    """For EXCLUDE evaluation: NPV = TN/(TN+FN), neg_recall = TN/(TN+FP)."""
    tp, fp, tn, fn = confusion(labels, preds)
    npv = tn / (tn + fn) if (tn + fn) > 0 else float("nan")
    neg_rec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    return npv, neg_rec


# ---------------------------------------------------------------------------
# 3. Threshold optimizers (run on TRAINING folds only)
# ---------------------------------------------------------------------------

def best_single_threshold(scores: List[float], labels: List[int],
                           grid_step: float = 0.01) -> float:
    """Find τ on grid that maximizes MCC."""
    best_tau = 0.5
    best_mcc = -2.0
    tau = 0.0
    while tau <= 1.0001:
        preds = [1 if s >= tau else 0 for s in scores]
        m = mcc(labels, preds)
        if m > best_mcc:
            best_mcc = m
            best_tau = tau
        tau += grid_step
    return best_tau


def best_include_threshold(scores: List[float], labels: List[int],
                            min_recall: float, grid_step: float = 0.01) -> float:
    """
    Find τ_inc that maximizes INCLUDE-precision subject to recall ≥ min_recall.
    Guards against the degenerate "set τ=0.99, INCLUDE one peptide, 100% precision" case.
    """
    best_tau = 0.5
    best_prec = -1.0
    tau = 0.0
    while tau <= 1.0001:
        preds = [1 if s >= tau else 0 for s in scores]
        prec, rec = precision_recall(labels, preds)
        if (not (rec != rec)  # not NaN
                and rec >= min_recall
                and not (prec != prec)
                and prec > best_prec):
            best_prec = prec
            best_tau = tau
        tau += grid_step
    return best_tau


def best_exclude_threshold(scores: List[float], labels: List[int],
                            min_neg_recall: float, grid_step: float = 0.01) -> float:
    """
    Find τ_exc that maximizes EXCLUDE-NPV (i.e., score < τ_exc → predict negative)
    subject to negative-recall ≥ min_neg_recall.
    """
    best_tau = 0.5
    best_npv = -1.0
    tau = 0.0
    while tau <= 1.0001:
        preds = [0 if s < tau else 1 for s in scores]
        npv, neg_rec = npv_and_neg_recall(labels, preds)
        if (not (neg_rec != neg_rec)
                and neg_rec >= min_neg_recall
                and not (npv != npv)
                and npv > best_npv):
            best_npv = npv
            best_tau = tau
        tau += grid_step
    return best_tau


# ---------------------------------------------------------------------------
# 4. Stratified k-fold split
# ---------------------------------------------------------------------------

def stratified_kfold(results, k: int, seed: int) -> List[List[int]]:
    """
    Return a list of k folds, each fold is a list of indices into `results`.
    Stratified so each fold has roughly the same pos/neg ratio.
    """
    rng = random.Random(seed)
    pos_idx = [i for i, (bp, _) in enumerate(results) if bp.label == 1]
    neg_idx = [i for i, (bp, _) in enumerate(results) if bp.label == 0]
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    folds = [[] for _ in range(k)]
    for i, idx in enumerate(pos_idx):
        folds[i % k].append(idx)
    for i, idx in enumerate(neg_idx):
        folds[i % k].append(idx)
    for f in folds:
        rng.shuffle(f)
    return folds


# ---------------------------------------------------------------------------
# 5. Per-fold evaluation
# ---------------------------------------------------------------------------

@dataclass
class FoldResult:
    fold_idx: int
    n_train: int
    n_test: int

    # Mode A — single threshold
    tau_binary: float
    mcc: float
    f1: float
    accuracy: float
    precision: float
    recall: float

    # Mode B — ternary thresholds
    tau_include: float
    tau_exclude: float
    include_precision: float
    include_recall: float
    exclude_npv: float
    exclude_neg_recall: float
    borderline_rate: float

    # Ranking metrics (threshold-independent)
    auroc: float
    auprc: float
    p_at_10: float
    p_at_20: float


def evaluate_fold(level_name: str,
                   weights: Dict[str, float],
                   results,
                   train_idx: List[int],
                   test_idx: List[int],
                   fold_i: int,
                   min_recall: float,
                   min_neg_recall: float) -> Optional[FoldResult]:
    """One fold: tune on train_idx, evaluate on test_idx."""
    # Score every peptide that has the required signals
    def collect(indices):
        scored = []
        for i in indices:
            bp, s = results[i]
            v = combine(s, weights)
            if v is not None:
                scored.append((v, bp.label))
        return scored

    train = collect(train_idx)
    test = collect(test_idx)
    if len(train) < 5 or len(test) < 5:
        return None
    if sum(l for _, l in train) in (0, len(train)):
        return None
    if sum(l for _, l in test) in (0, len(test)):
        return None

    train_scores = [s for s, _ in train]
    train_labels = [l for _, l in train]
    test_scores = [s for s, _ in test]
    test_labels = [l for _, l in test]

    # --- Tune thresholds on training fold ---
    tau_bin = best_single_threshold(train_scores, train_labels)
    tau_inc = best_include_threshold(train_scores, train_labels, min_recall)
    tau_exc = best_exclude_threshold(train_scores, train_labels, min_neg_recall)

    # --- Apply to test fold ---
    # Mode A: binary
    preds_bin = [1 if s >= tau_bin else 0 for s in test_scores]
    prec_bin, rec_bin = precision_recall(test_labels, preds_bin)

    # Mode B: ternary
    # INCLUDE: score ≥ tau_inc → 1, else considered negative for include-precision
    preds_inc = [1 if s >= tau_inc else 0 for s in test_scores]
    inc_prec, inc_rec = precision_recall(test_labels, preds_inc)
    # EXCLUDE: score < tau_exc → predicted negative
    preds_exc = [0 if s < tau_exc else 1 for s in test_scores]
    exc_npv, exc_neg_rec = npv_and_neg_recall(test_labels, preds_exc)
    # Borderline: tau_exc ≤ score < tau_inc (only meaningful if tau_inc > tau_exc)
    if tau_inc > tau_exc:
        borderline_count = sum(1 for s in test_scores if tau_exc <= s < tau_inc)
    else:
        borderline_count = 0
    borderline_rate = borderline_count / len(test_scores) if test_scores else 0.0

    # --- Ranking metrics on test fold ---
    auroc_v = _auroc(test_labels, test_scores)
    auprc_v = _auprc(test_labels, test_scores)
    p10, _ = _precision_recall_at_k(test_labels, test_scores, 10)
    p20, _ = _precision_recall_at_k(test_labels, test_scores, 20)

    return FoldResult(
        fold_idx=fold_i,
        n_train=len(train), n_test=len(test),
        tau_binary=tau_bin,
        mcc=mcc(test_labels, preds_bin),
        f1=f1(test_labels, preds_bin),
        accuracy=accuracy(test_labels, preds_bin),
        precision=prec_bin, recall=rec_bin,
        tau_include=tau_inc, tau_exclude=tau_exc,
        include_precision=inc_prec, include_recall=inc_rec,
        exclude_npv=exc_npv, exclude_neg_recall=exc_neg_rec,
        borderline_rate=borderline_rate,
        auroc=auroc_v, auprc=auprc_v,
        p_at_10=p10, p_at_20=p20,
    )


# ---------------------------------------------------------------------------
# 6. CV aggregation
# ---------------------------------------------------------------------------

def mean_std(values: List[float]) -> Tuple[float, float]:
    clean = [v for v in values if v == v]  # filter NaN
    if not clean:
        return float("nan"), float("nan")
    if len(clean) == 1:
        return clean[0], 0.0
    return statistics.mean(clean), statistics.stdev(clean)


@dataclass
class CVSummary:
    level: str
    n_folds: int
    # All metrics: (mean, std)
    mcc: Tuple[float, float]
    f1: Tuple[float, float]
    accuracy: Tuple[float, float]
    precision: Tuple[float, float]
    recall: Tuple[float, float]
    tau_binary: Tuple[float, float]

    include_precision: Tuple[float, float]
    include_recall: Tuple[float, float]
    exclude_npv: Tuple[float, float]
    exclude_neg_recall: Tuple[float, float]
    borderline_rate: Tuple[float, float]
    tau_include: Tuple[float, float]
    tau_exclude: Tuple[float, float]

    auroc: Tuple[float, float]
    auprc: Tuple[float, float]
    p_at_10: Tuple[float, float]
    p_at_20: Tuple[float, float]


def summarize(level_name: str, folds: List[FoldResult]) -> CVSummary:
    return CVSummary(
        level=level_name, n_folds=len(folds),
        mcc=mean_std([f.mcc for f in folds]),
        f1=mean_std([f.f1 for f in folds]),
        accuracy=mean_std([f.accuracy for f in folds]),
        precision=mean_std([f.precision for f in folds]),
        recall=mean_std([f.recall for f in folds]),
        tau_binary=mean_std([f.tau_binary for f in folds]),
        include_precision=mean_std([f.include_precision for f in folds]),
        include_recall=mean_std([f.include_recall for f in folds]),
        exclude_npv=mean_std([f.exclude_npv for f in folds]),
        exclude_neg_recall=mean_std([f.exclude_neg_recall for f in folds]),
        borderline_rate=mean_std([f.borderline_rate for f in folds]),
        tau_include=mean_std([f.tau_include for f in folds]),
        tau_exclude=mean_std([f.tau_exclude for f in folds]),
        auroc=mean_std([f.auroc for f in folds]),
        auprc=mean_std([f.auprc for f in folds]),
        p_at_10=mean_std([f.p_at_10 for f in folds]),
        p_at_20=mean_std([f.p_at_20 for f in folds]),
    )


# ---------------------------------------------------------------------------
# 7. Reporting
# ---------------------------------------------------------------------------

def fmt(t: Tuple[float, float], digits: int = 3) -> str:
    m, s = t
    if m != m:
        return "   nan"
    return f"{m:.{digits}f}±{s:.{digits}f}"


def print_mode_a_table(summaries: List[CVSummary]):
    print(f"\n{'='*108}")
    print(f"  MODE A — Single binary threshold (τ tuned per-fold to maximize MCC)")
    print(f"{'='*108}")
    print(f"  {'Level':<18} {'τ_bin':>11} {'MCC':>11} {'F1':>11} "
          f"{'Acc':>11} {'Prec':>11} {'Recall':>11}")
    print("  " + "-" * 106)
    for s in summaries:
        print(f"  {s.level:<18} {fmt(s.tau_binary):>11} {fmt(s.mcc):>11} "
              f"{fmt(s.f1):>11} {fmt(s.accuracy):>11} "
              f"{fmt(s.precision):>11} {fmt(s.recall):>11}")
    print("  " + "-" * 106)
    print(f"\n  MCC interpretation: 0 = no skill, 1 = perfect, -1 = perfectly wrong.")
    print(f"  ±std shows fold-to-fold variation. If std > mean, the metric is unreliable.")


def print_mode_b_table(summaries: List[CVSummary]):
    print(f"\n{'='*108}")
    print(f"  MODE B — Ternary thresholds (clinical operating points)")
    print(f"{'='*108}")
    print(f"  {'Level':<18} {'τ_inc':>11} {'INC-prec':>11} {'INC-rec':>11} "
          f"{'τ_exc':>11} {'EXC-npv':>11} {'BORD-rate':>11}")
    print("  " + "-" * 106)
    for s in summaries:
        print(f"  {s.level:<18} {fmt(s.tau_include):>11} {fmt(s.include_precision):>11} "
              f"{fmt(s.include_recall):>11} {fmt(s.tau_exclude):>11} "
              f"{fmt(s.exclude_npv):>11} {fmt(s.borderline_rate):>11}")
    print("  " + "-" * 106)
    print(f"\n  INC-prec: of peptides we said INCLUDE, fraction truly immunogenic")
    print(f"  EXC-npv:  of peptides we said EXCLUDE, fraction truly non-immunogenic")
    print(f"  BORD-rate: fraction in the uncertain middle (lower is better, but tradeoff)")


def print_ranking_table(summaries: List[CVSummary]):
    print(f"\n{'='*108}")
    print(f"  RANKING METRICS (threshold-independent, CV mean ± std)")
    print(f"{'='*108}")
    print(f"  {'Level':<18} {'AUROC':>13} {'AUPRC':>13} {'P@10':>13} {'P@20':>13}")
    print("  " + "-" * 106)
    for s in summaries:
        print(f"  {s.level:<18} {fmt(s.auroc):>13} {fmt(s.auprc):>13} "
              f"{fmt(s.p_at_10, 2):>13} {fmt(s.p_at_20, 2):>13}")
    print("  " + "-" * 106)


# ---------------------------------------------------------------------------
# 8. Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="5-fold CV threshold tuning for ITSNdb benchmark")
    ap.add_argument("--data", default="validation/ITSNdb.csv")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-recall", type=float, default=0.30,
                    help="Minimum recall for INCLUDE threshold optimization "
                         "(prevents degenerate τ=0.99 → 'INCLUDE 1 peptide' gaming)")
    ap.add_argument("--min-neg-recall", type=float, default=0.30,
                    help="Minimum negative-recall for EXCLUDE threshold optimization")
    ap.add_argument("--levels", nargs="+",
                    default=["L1_binding", "L2_+presentation",
                             "L3_+immunogenic", "L4_+structure"])
    ap.add_argument("--out", default="validation/cv_results.json")
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir) if args.cache_dir else (
        Path.home() / ".cache" / "neoantigens" / "benchmark"
    )
    if not cache_dir.exists():
        print(f"❌ Cache dir not found: {cache_dir}")
        print(f"   Run the v2 benchmark first to populate cache.")
        sys.exit(1)

    if not os.path.exists(args.data):
        print(f"❌ Data not found: {args.data}")
        sys.exit(1)

    peptides = load_itsndb(args.data)
    results = load_cached_scores(peptides, cache_dir)
    if not results:
        print(f"❌ No cached scores found.")
        sys.exit(1)

    n_pos = sum(bp.label for bp, _ in results)
    print("=" * 108)
    print(f"  5-FOLD CV THRESHOLD TUNING  (stratified, seed={args.seed})")
    print("=" * 108)
    print(f"  Peptides:         {len(results)}  ({n_pos} positive / {len(results)-n_pos} negative)")
    print(f"  Folds:            {args.folds}  (stratified by label)")
    print(f"  Levels:           {', '.join(args.levels)}")
    print(f"  Mode A guard:     τ_bin tuned for max MCC on training folds")
    print(f"  Mode B guards:    τ_inc requires recall ≥ {args.min_recall}, "
          f"τ_exc requires neg-recall ≥ {args.min_neg_recall}")
    print(f"  Literature layer: EXCLUDED (honest mode)")
    print("=" * 108)

    folds = stratified_kfold(results, args.folds, args.seed)
    all_summaries: List[CVSummary] = []
    all_fold_results: Dict[str, List[FoldResult]] = {}

    for level_name in args.levels:
        if level_name not in ABLATION_LEVELS:
            print(f"⚠️  Unknown level: {level_name}, skipping")
            continue
        weights = ABLATION_LEVELS[level_name]

        fold_results = []
        for k in range(args.folds):
            test_idx = folds[k]
            train_idx = [i for j, f in enumerate(folds) if j != k for i in f]
            fr = evaluate_fold(
                level_name, weights, results,
                train_idx, test_idx, k,
                args.min_recall, args.min_neg_recall,
            )
            if fr is not None:
                fold_results.append(fr)

        if not fold_results:
            print(f"⚠️  Level {level_name}: no valid folds, skipping")
            continue

        all_fold_results[level_name] = fold_results
        all_summaries.append(summarize(level_name, fold_results))

    # --- Print all three tables ---
    print_ranking_table(all_summaries)
    print_mode_a_table(all_summaries)
    print_mode_b_table(all_summaries)

    # --- Per-fold detail for transparency ---
    print(f"\n{'='*108}")
    print(f"  PER-FOLD DETAIL — best level (highest mean MCC)")
    print(f"{'='*108}")
    if all_summaries:
        best_lvl = max(all_summaries, key=lambda s: s.mcc[0] if s.mcc[0] == s.mcc[0] else -2)
        folds_for_best = all_fold_results[best_lvl.level]
        print(f"  Level: {best_lvl.level}")
        print(f"  {'Fold':>5} {'n_test':>7} {'τ_bin':>8} {'MCC':>8} {'F1':>8} "
              f"{'AUROC':>8} {'τ_inc':>8} {'INC-prec':>10}")
        print("  " + "-" * 76)
        for f in folds_for_best:
            print(f"  {f.fold_idx:>5} {f.n_test:>7} {f.tau_binary:>8.2f} "
                  f"{f.mcc:>8.3f} {f.f1:>8.3f} {f.auroc:>8.3f} "
                  f"{f.tau_include:>8.2f} {f.include_precision:>10.3f}")

    # --- Read-this section ---
    print(f"\n{'='*108}")
    print(f"  READING THIS REPORT")
    print(f"{'='*108}")
    print(f"  • MCC: 0=no skill, 1=perfect. ITSNdb is hard — MCC 0.2-0.3 is competitive.")
    print(f"  • If τ_bin std is large (>0.1), the threshold is unstable across folds —")
    print(f"    means there's no single right cutoff and the operating point is fuzzy.")
    print(f"  • INC-prec is the headline clinical number: 'of peptides we recommend,")
    print(f"    what fraction are real immunogens.' Random baseline = {n_pos/len(results):.2f}.")
    print(f"  • BORD-rate is the cost of safety: higher means more peptides need human")
    print(f"    review. A tool that says BORDERLINE for everything has BORD-rate=1.0.")
    print(f"  • The L1→L4 progression shows which layers earn their place AT THE")
    print(f"    DECISION BOUNDARY (not just in ranking). Layers that improve AUROC")
    print(f"    but not MCC are good rankers but don't help classification.")

    # --- Save ---
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    report = {
        "dataset": "ITSNdb",
        "n_peptides": len(results),
        "n_positive": n_pos,
        "n_folds": args.folds,
        "seed": args.seed,
        "min_recall_include": args.min_recall,
        "min_neg_recall_exclude": args.min_neg_recall,
        "summaries": [
            {
                "level": s.level,
                "n_folds": s.n_folds,
                # Flatten (mean, std) pairs
                **{f"{m}_mean": getattr(s, m)[0] for m in [
                    "mcc", "f1", "accuracy", "precision", "recall",
                    "tau_binary",
                    "include_precision", "include_recall",
                    "exclude_npv", "exclude_neg_recall", "borderline_rate",
                    "tau_include", "tau_exclude",
                    "auroc", "auprc", "p_at_10", "p_at_20",
                ]},
                **{f"{m}_std": getattr(s, m)[1] for m in [
                    "mcc", "f1", "accuracy", "precision", "recall",
                    "tau_binary",
                    "include_precision", "include_recall",
                    "exclude_npv", "exclude_neg_recall", "borderline_rate",
                    "tau_include", "tau_exclude",
                    "auroc", "auprc", "p_at_10", "p_at_20",
                ]},
            }
            for s in all_summaries
        ],
        "per_fold": {
            lvl: [asdict(f) for f in fr]
            for lvl, fr in all_fold_results.items()
        },
    }
    json.dump(report, open(args.out, "w"), indent=2)
    print(f"\n  💾 Full per-fold detail saved: {args.out}\n")


if __name__ == "__main__":
    main()
