# Directional coherence and effect magnitude in single-cell CRISPR perturbation responses

<p align="center">
    <a style="text-decoration:none !important;" href="https://arxiv.org/abs/2604.16642" alt="arXiv"><img src="https://img.shields.io/badge/paper-arXiv-blue" /></a>
    <a style="text-decoration:none !important;" href="https://huggingface.co/papers/2604.16642" alt="Hugging Face Papers"><img src="https://img.shields.io/badge/paper-Hugging%20Face-FFD21E?logo=huggingface&logoColor=black" /></a>
</p>

Code to reproduce the analyses and figures. Frozen pipeline: `CONFIG_VERSION=2026-07-29.1`, seed 320, six datasets (n = 2,285 perturbations).

## Setup

```bash
pip install -r requirements.txt
```

Python 3.12 requires the pinned `torch==2.3.1` and `torchtext==0.18.0` (needed for scGPT). Datasets are loaded via [pertpy](https://github.com/scverse/pertpy) into `PERTPY_CACHE_DIR` (default `/tmp/pertpy_data`).

Optional environment variables:

| Variable | Default | Role |
|---|---|---|
| `SHESHA_OUT` | `code/shesha-crispr/` | Output directory for CSVs and figures |
| `PERTPY_CACHE_DIR` | `/tmp/pertpy_data` | h5ad cache |

### scGPT weights

For `embeddings/scgpt_analysis.py`, download the **Whole Human** model (`scGPT_human`) from the [scGPT repository](https://github.com/bowang-lab/scGPT). The folder must contain `best_model.pt`, `vocab.json`, and `args.json`. Pass it as `--model-dir`.

scGPT expects raw counts; the script reloads raw data when the in-memory matrix is already normalized.

## Usage

Run every script from `code/` (a small `paths.py` helper puts the subdirectories on `sys.path` so scripts still import each other by module name):

```bash
cd code

# Frozen Sp table (six datasets)
python utils/run_frozen_main.py
python utils/check_frozen_sp_guard.py --frozen-sp shesha-crispr/frozen_sp_scores.csv

# Competitors and covariates
python competitors/edistance_competitor_analysis.py
python competitors/trade_competitor_analysis.py
python pathways/pathway_analysis.py
python pathways/cell_quality_partial.py
python pathways/attach_stress_markers.py

# Main figures
python figures/magnitude_matched_coherence_illustration.py
python figures/fig2_magnitude_stability_loess.py
python figures/fig5_pathway_forest.py
python figures/fig_revision_new.py
python figures/fig_si_regen.py
python figures/fig_s8_null_model.py
python figures/fig_s9_stress_forest.py
python figures/fig_s10_scgpt_concordance.py
```

GPU / Colab: `embeddings/diffusion_map_robustness.py`, `embeddings/phate_embedding_robustness.py`, `embeddings/scgpt_analysis.py`, and the PCA / leave-one-out ablations in `robustness/geometric_stability_main_analysis.py`.

Outputs are written to `code/shesha-crispr/` (or `SHESHA_OUT`). Generated CSVs and figures are not part of this repository.

## Repository layout

```
LICENSE
README.md
requirements.txt
code/
  paths.py                 import-path helper (loaded by every script)
  utils/                   freeze, pipeline, stats, I/O
  figures/                 main-text and SI figures
  competitors/             E-distance, TRADE, Song PS, related comparisons
  pathways/                Hallmark / stress / CORUM / Adamson UPR
  embeddings/              scGPT, PHATE, diffusion maps
  efficiency/              responder-filter sensitivity
  robustness/              ablations, split-half, synthetic, concordance
  datasets/                Papalexi GEO rebuild, Norman combinatorial
```

### `utils/` — pipeline

Shared freeze and scoring. Start here.

| Script | Role |
|---|---|
| `pipeline_config.py` | Freeze (version, seed, dataset pins, survival gate) |
| `pipeline_core.py` | Load, downsample, PCA/SVD, Sp, digest guard |
| `stats_utils.py` | Rank partial Spearman, bootstrap CIs |
| `revision_io.py` | Output paths, frozen Sp I/O, downloads |
| `fig_style.py` | Shared plot style and dataset colors |
| `remap_modality_labels.py` | Dataset display-name map (KO vs CRISPRi, Adamson UPR vs pilot) |
| `run_frozen_main.py` | Score all six datasets; write `frozen_sp_scores.csv` |
| `check_frozen_sp_guard.py` | Abort on stale `config_version` or digest |
| `check_pipeline_reproducibility.py` | Bit-identical materialize + Sp |
| `inspect_matrix_scale.py` | Pin `matrix_is_log` per dataset |

### `figures/`

| Script | Figure |
|---|---|
| `magnitude_matched_coherence_illustration.py` | Fig 1 (real-data contrast) |
| `new_fig1.py` | Fig 1 helpers / alternate layout |
| `fig_1.py` | Shared density / despine helpers used by other figure scripts |
| `fig2_magnitude_stability_loess.py` | Fig 2, S1 |
| `fig2,4,5.py` | Combined Fig 2 / scGPT companion / combinatorial panel |
| `fig5_pathway_forest.py` | Fig 5 (apoptosis / p53 / DDIT3) |
| `fig7_combinatorial_partial.py` | Combinatorial vs single-gene (magnitude-conditioned) |
| `fig_revision_new.py` | Embeddings, E-distance, efficiency, Approach B |
| `fig_si_regen.py` | S2, S3, S5–S7 |
| `fig_s4_distance_metrics.py` | S4 (distance metrics) |
| `fig_s4_method_comparison_barchart.py` | S4 (method comparison bars) |
| `fig_s8_null_model.py` | S8 |
| `fig_s9_stress_forest.py` | S9 |
| `fig_s10_scgpt_concordance.py` | S10 |
| `fig_stress_markers.py` | Stress-marker companion plot |
| `fig_method_comparison.py` | Method-comparison figure |
| `fig_song_ps.py` | Song PS comparison figure |
| `fig_reproducibility.py` | Split-half / reproducibility figure |
| `fig3.py` | Older three-method comparison |
| `fig_norman.py` | Norman-only companion |
| `fig_replogle.py` | Replogle-only companion |

### `competitors/`

| Script | Role |
|---|---|
| `edistance_competitor_analysis.py` | E-distance vs Sp, partials, QC models |
| `trade_competitor_analysis.py` | TRADE TWI vs Sp / centroid |
| `song_ps_replication.py` | Song et al. PS (scMAGeCK port + Euclidean / Mahalanobis proxies) |
| `papalexi_method_comparison.py` | Papalexi method comparison |
| `nadig.py` | Alternative heterogeneity metrics (η², DE count, spread) |

### `pathways/`

| Script | Role |
|---|---|
| `pathway_analysis.py` | Hallmark pathway scores vs Sp (Approach A / B) |
| `attach_stress_markers.py` | Join DDIT3 / ATF4 / XBP1 / HSPA5 onto frozen Sp |
| `stress_marker_tests.py` | Marker-level partials |
| `cell_quality_partial.py` | Pathway / stress partials also conditioned on mito / n_genes / n_counts |
| `upr_gene_exclusion.py` | Drop Hallmark UPR genes, recompute Sp–stress |
| `corum_systematic_benchmark.py` | CORUM / TRRUST / DepMap vs Sp |
| `corum_loess_discordance.py` | Complex membership vs LOESS discordance |
| `recompute_partial_correlations.py` | Rank-based partial Spearman on key tables |
| `sign_reversal_partial_sp.py` | Sign of Sp–stress / Sp–magnitude across datasets |
| `adamson_upr_spike.py` | Adamson UPR positive-control Sp table |
| `adamson_upr_magnitude_partial.py` | UPR-core Sp deficit conditioned on magnitude |
| `norman_hspa5_quartile_analysis.py` | Norman HSPA5 × discordance by magnitude quartile |
| `go_functional_diversity.py` | GO functional-diversity scores |

### `embeddings/`

| Script | Role |
|---|---|
| `scgpt_analysis.py` | Sp in scGPT space; PCA vs scGPT concordance |
| `phate_embedding_robustness.py` | Sp in PHATE vs frozen PCA |
| `diffusion_map_robustness.py` | Sp in diffusion maps vs frozen PCA |

### `efficiency/`

| Script | Role |
|---|---|
| `efficiency_filter_sp.py` | Recompute Sp on Mixscape / Song-PS responding cells only (sensitivity; does not rewrite the freeze) |

### `robustness/`

| Script | Role |
|---|---|
| `geometric_stability_main_analysis.py` | PCA ablation, leave-one-out, mixed-effects, whitening |
| `geometric_stability_main_analysis_papalexi.py` | Same, Papalexi-only |
| `analysis.py` | Earlier six-dataset geometric analysis |
| `robustness_tests.py` | Nonlinear discordance residuals and related checks |
| `split_half_guide_analysis.py` | Independent-reagent (between-guide) mean-shift cosine vs gene Sp |
| `split_half_reproducibility.py` | Split-half Sp / PS reproducibility |
| `cell_count_downsampling.py` | Sp ranking reliability vs cell number |
| `curved_trajectory_counterexample.py` | Synthetic curved trajectory where mean-shift Sp is low |
| `synthetic_benchmark.py` | scDesign3 efficiency / multi-program benchmark (not used in the manuscript) |
| `cross_dataset_concordance.py` | Shared-gene Sp between Norman and Replogle |

### `datasets/`

| Script | Role |
|---|---|
| `papalexi_geo_full.py` | Rebuild Papalexi from GEO GSE153056 (independent of pertpy) |
| `papalexi_perturbation_audit.py` | Guide / gene-target inventory vs frozen labels |
| `norman_combinatorial_analysis.py` | Norman combinatorial vs single-gene Sp |

## Citation

```bibtex
@article{raju2026crispr,
  title = {Directional coherence and effect magnitude in single-cell CRISPR perturbation responses},
  author = {Raju, Prashant C.},
  journal = {arXiv preprint arXiv:2604.16642},
  year = {2026}
}
```
