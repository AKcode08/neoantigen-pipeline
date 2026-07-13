# Neoantigen Prediction Pipeline

A meta-analysis pipeline for prioritizing neoantigen candidates in personalized cancer vaccine development. Combines five independent prediction layers into a validated weighted ensemble, with an LLM synthesis layer that produces expert-grade interpretation and reporting.

**Status:** Research prototype. Validated on ITSNdb (199 experimentally validated peptides, 5-fold CV). External validation on TESLA in progress.

---

## The problem

Teams selecting neoantigens for personalized vaccines run several prediction tools like NetMHCpan, MHCflurry, BigMHC, PRIME — and routinely get conflicting scores. Reconciling them is manual, inconsistent, and rarely documented. This pipeline synthesizes multiple orthogonal signals into a single ranked, reasoned, auditable output.

---

## Architecture

Input is a `(peptide, HLA allele)` pair. Output is a ranked recommendation with a full HTML report.

| Layer | Question answered | Tool |
|---|---|---|
| 1. Binding | Does the peptide bind MHC-I? | MHCflurry 2.0 `Class1AffinityPredictor` |
| 2. Presentation | Will it be displayed on the cell surface? | MHCflurry 2.0 `Class1PresentationPredictor` |
| 3. Immunogenicity | Will T cells respond to it? | BigMHC IM (Albert et al., *Nat Mach Intell* 2023) |
| 4. Structure | What does the 3D peptide-MHC complex reveal? | NeoaPred PepConf (Docker) |
| 5. Literature | What does published evidence say? | IEDB Query API + NCBI E-utilities (PubMed) |
| **Synthesis** | **What does it all mean?** | **Claude Opus 4** |

The synthesis layer parses the generated PDB, reasons over per-residue anchor and TCR-contact geometry, contextualizes the literature hits, and produces a structured expert analysis, including catching category errors the numeric layers cannot (see *Why the LLM layer earns its place* below).

**[→ View a sample report](NLVPMVATV_CMV_epitope_EXCLUDE.html)**
(download and open in a browser as GitHub won't render the interactive 3D viewer inline)
---

## Validation

Benchmarked against **ITSNdb** ([Pertschy et al., 2024](https://github.com/elmerfer/ITSNdb)): 199 experimentally validated tumor neoantigens (129 immunogenic / 70 non-immunogenic). ITSNdb is deliberately pre-filtered so that every peptide already binds and is presented, isolating the hard question: *given presentation, which peptides actually elicit a T-cell response?*

**Protocol:** 5-fold stratified cross-validation. Thresholds tuned exclusively on training folds; all metrics reported on held-out folds. Mean ± standard deviation across folds.

### Results (honest mode: literature layer disabled)

| Configuration | AUROC | MCC | F1 | INCLUDE-precision | BORDERLINE-rate |
|---|---|---|---|---|---|
| L1: binding only | 0.60 ± 0.14 | 0.26 ± 0.06 | 0.40 ± 0.15 | 0.91 ± 0.09 | 0.42 ± 0.24 |
| L2: + presentation | 0.61 ± 0.11 | 0.17 ± 0.14 | 0.46 ± 0.07 | 0.79 ± 0.10 | 0.12 ± 0.05 |
| L3: + immunogenicity (30% wt) | 0.59 ± 0.07 | 0.10 ± 0.17 | 0.34 ± 0.14 | 0.69 ± 0.10 | 0.11 ± 0.11 |
| L4: + structure | 0.57 ± 0.07 | 0.23 ± 0.13 | 0.30 ± 0.11 | 0.69 ± 0.09 | 0.18 ± 0.22 |
| **L3-tuned (shipped)** | **0.62 ± 0.09** | **0.24 ± 0.14** | **0.55 ± 0.13** | **0.81 ± 0.05** | **0.24 ± 0.07** |
| Random baseline | 0.50 | 0.00 | — | 0.65 | — |

Of every 100 peptides the pipeline marks INCLUDE, ~81 are genuine immunogens, 16 points above the dataset prevalence of 65%, with a tight ±5% across folds.

### Leakage controls

Two layers can leak the test label, and both are controlled:

- **Literature layer** retrieves prior IEDB T-cell assay records for benchmark peptides: that is, information retrieval, not prediction. Every analysis runs in two modes: *honest* (literature OFF, the defensible number) and *leaked* (literature ON, an optimistic ceiling). The gap on ITSNdb is **+0.10 AUROC**: quantified, reported, never hidden.
- **BigMHC** was trained on IEDB-overlapping data and may have seen ITSNdb peptides. Its measured contribution is therefore an upper bound. External validation on TESLA is the fix.

All above numbers use honest mode.

---

## What validation actually changed

The benchmark was not a rubber stamp. It overturned two design decisions:

**The immunogenicity layer was over-weighted 3×.** At its original 0.30 ensemble weight, BigMHC *degraded* both ranking and classification — MCC 0.10 ± 0.17, a standard deviation larger than the mean, i.e. instability rather than signal. A weight sweep located the empirical optimum at **0.10**, where the layer contributes a real and stable lift (MCC 0.24 ± 0.14, F1 0.55 ± 0.13). Weight sweeps are not optional.

**The structural layer carried no discriminative signal.** The original implementation returned a constant for every peptide that folded successfully, zero discriminative power by construction. Rebuilding it with four interpretable per-PDB features (TCR-facing contact exposure, mutation-position TCR visibility, anchor-residue compatibility, Kyte-Doolittle hydrophobicity at TCR-facing positions) produced individual feature AUROCs of **0.49–0.50**, indistinguishable from random. The layer was dropped from the consensus score and retained only for the LLM's qualitative reasoning.

This is a negative result about *single-static-PDB hand-engineered features on a binding-pre-filtered dataset*, not a claim that structure is irrelevant to neoantigen prediction. The most likely path to signal (true mutant-vs-wild-type surface comparison, requiring both folds) was not tested.

### Shipped configuration

```
binding 0.45 | presentation 0.45 | immunogenicity 0.10 | structure 0.00
INCLUDE ≥ 0.81 | EXCLUDE < 0.60 | BORDERLINE otherwise
```

Thresholds are CV-derived, not intuition-set. The originals (0.70 / 0.30) produced a >40% BORDERLINE rate.

---

## Why the LLM layer earns its place

The synthesis layer is deliberately **not** ablated in AUROC, and that is a methodological position rather than an omission. Claude reads the same five inputs the ensemble does: it cannot conjure a signal that is not there, and scoring it as a sixth quantitative predictor would measure "a learned reweighting of inputs we already have," not its actual contribution.

Its real value is **category-level reasoning that no reweighting can reach.** The clearest demonstration: `NLVPMVATV` scored highly across all five layers: 16.6 nM binding, 0.97 presentation, high immunogenicity, strong structure, 754 IEDB assays at a 98% positive response rate. The ensemble said INCLUDE with 0.93 consensus. Claude said **EXCLUDE**, correctly identifying it as the CMV pp65 immunodominant viral epitope, not a tumor neoantigen at all, and useless in a cancer vaccine regardless of how well it scores.

No numerical reweighting of those five layers reaches that conclusion. That is the layer's job.

---

## Repository layout

```
src/
  layer_2_predictors/
    mhc_binding.py              MHCflurry affinity
    mhc_presentation.py         MHCflurry presentation
    immunogenicity_bigmhc.py    BigMHC IM wrapper
    ensemble.py                 Weighted consensus + recommendation
  layer_3_structure/
    neoapred_structure.py       NeoaPred PepConf via Docker
  layer_4_literature/
    iedb_query.py               IEDB Query API (cached, 30d TTL)
    pubmed_search.py            NCBI E-utilities, multi-strategy HLA matching
    literature_evidence.py      Evidence aggregation + scoring
  layer_5_synthesis/
    claude_engine.py            Expert synthesis, PDB-aware
  layer_6_output/
    report_generator.py         HTML report + NGL.js 3D viewer
  orchestrator.py               Single-peptide and batch CSV modes

validation/
  ITSNdb.csv                    Benchmark dataset (199 peptides)
  benchmark_itsndb.py           Ablation harness, dual leakage modes
  structure_features.py         Four interpretable per-PDB features
  bigmhc_weight_sweep.py        Weight sweep (cache replay)
  cv_threshold_tuning.py        5-fold CV, MCC/F1, clinical operating points
```

Every per-peptide layer score is cached to disk, so ablation, weight sweeps, and cross-validation re-run in seconds without recomputing predictions.

---

## Usage

```bash
# Single peptide, auto-open report
python -m src.orchestrator NLVPMVATV HLA-A*02:01 --open

# Batch from CSV
python -m src.orchestrator --batch peptides.csv

# Fast mode (skip Docker + API layers)
python -m src.orchestrator NLVPMVATV HLA-A*02:01 --no-structure --no-literature

# Reproduce the benchmark
python -m validation.benchmark_itsndb
python -m validation.cv_threshold_tuning
```

### Setup

```bash
python3.11 -m venv venv311 && source venv311/bin/activate
pip install -r requirements.txt
mhcflurry-downloads fetch                                  # ~500MB models
git clone https://github.com/KarchinLab/bigmhc.git         # immunogenicity layer
docker pull panda1103/neoapred:1.0.0                       # structure layer
```

`.env`:
```
ANTHROPIC_API_KEY=...
NCBI_API_KEY=...          # optional; raises PubMed rate limit 3→10 req/sec
NCBI_EMAIL=...
```

**Requires Python 3.11** — MHCflurry is incompatible with 3.13+ (removed `pipes` module). Docker Desktop must be running for the structure layer.

---

## Known limitations

- **Dataset size.** 199 peptides across 5 folds gives ~40 test peptides per fold. The ±0.09–0.14 AUROC standard deviations are real; a different 199-peptide dataset could plausibly land anywhere in that band.
- **Weights tuned on the evaluation dataset.** Per-fold threshold tuning was correctly isolated, but the 0.10 immunogenicity weight came from a sweep over the full ITSNdb set. The *direction* of the finding is robust; the exact magnitudes are not certified out of sample.
- **BigMHC training overlap.** Its measured contribution is an upper bound for genuinely novel neoantigens.
- **Structure evaluated as single static PDB.** Wild-type counterparts were not folded, so true mutant-vs-WT foreignness — the most promising structural feature — remains untested.

---

## In progress

**TESLA external validation.** 608 peptides from the Tumor Neoantigen Selection Alliance ([Wells et al., *Cell* 2020](https://doi.org/10.1016/j.cell.2020.09.015)), 37 immunogenic (6.1% prevalence). Running the shipped configuration with **frozen weights and frozen thresholds** -> no further tuning. This is the step that converts internal validation into an external benchmark.

The class balance is a genuine stress test: ITSNdb is 65% immunogenic, TESLA is 6%. These are fundamentally different operating conditions, and performance on both is what real-world deployment looks like. For calibration, the TESLA consortium reported that *no participating team placed more than 20 of the 37 immunogenic peptides in their top 100* — the best pipelines in the field recover roughly half.

---

## References

- Wells et al. (2020) *Cell* — Key Parameters of Tumor Epitope Immunogenicity (TESLA)
- Albert et al. (2023) *Nature Machine Intelligence* — BigMHC
- O'Donnell et al. (2020) *Cell Systems* — MHCflurry 2.0
- Dulab (2024) *Bioinformatics* — NeoaPred
- ITSNdb — https://github.com/elmerfer/ITSNdb

---

## License

MIT
