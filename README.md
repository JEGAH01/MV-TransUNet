# MV-TransUNet-HVS

Code for **"MV-TransUNet-HVS: Scale-Conditioned Attention and Topology-Aware Supervision for Retinal Vessel Segmentation"** (submitted to *Biomedical Signal Processing and Control*).

A hybrid CNN-Transformer network (ResNet-50 encoder, Vision Transformer bottleneck, U-Net-style decoder with deep supervision) for retinal vessel segmentation, extended with two additions:

- **SC-VAM** (Scale-Conditioned Vessel Attention Module) — a zero-initialized FiLM-style pathway that conditions channel attention on a per-sample vessel-scale ratio.
- **TFFM** (Topology Feature Fusion Module) — a decoder-attached module that jointly predicts vessel segmentation, skeleton, endpoint, and junction maps, with no graph construction or graph-neural-network component.

The topology-supervised model and a matched non-topology baseline are trained on the DRIVE training split only, under an identical optimizer, scheduler, patch-sampling, and reconstruction protocol, and evaluated zero-shot (no retraining) on STARE, CHASE_DB1, and HRF.

## Setup

```bash
conda env create -f environment.yml
conda activate mvtransunet
```

## Repository layout

- `models/` — network definitions (backbone, VAM/SC-VAM, decoder, TFFM, losses).
- `src/` — training, evaluation, dataset, and metrics code.
- `config*.yaml` — experiment configurations (see below).
- `_archive/` — superseded scratch scripts and an earlier, unreported research direction (label-free adaptive-resolution / scale-selection surrogates); kept for the authors' own reference, not part of the reproducibility path for the paper.

## Configurations

The primary comparison reported in the paper (three seeds each) corresponds to:

- `config_baseline_matched_topology_scvam_seed42.yaml`, `config_baseline_matched_topology_scvam_seed7.yaml`, `config_baseline_matched_topology_scvam_seed123.yaml` — matched non-topology baseline (backbone + VAM + SC-VAM), seeds 42, 7, and 123.
- `config_topology_ablation_a8_scvam.yaml`, `config_topology_ablation_a8_scvam_seed7.yaml`, `config_topology_ablation_a8_scvam_seed123.yaml` — full topology model (backbone + VAM + SC-VAM + TFFM), seeds 42, 7, and 123.

Two additional isolating ablations, each run at two seeds under the same matched protocol:

- `config_baseline_matched_topology_novam_seed42.yaml`, `config_baseline_matched_topology_novam_seed7.yaml` — SC-VAM ablation (backbone + VAM only, no scale-conditioning).
- `config_baseline_matched_topology_weighted_seed42.yaml`, `config_baseline_matched_topology_weighted_seed7.yaml` — Weighted-Baseline ablation (plain baseline with the width-adaptive per-pixel loss weighting applied, but no TFFM/topology heads), used to isolate that weighting's contribution from the topology supervision itself.

Other `config_topology_ablation_a*.yaml` files record the intermediate ablation steps (width-adaptive loss weighting, scale-augmentation strength) that led to the final configuration above; they are not all individually reported in the paper but are kept for transparency.

## Training

```bash
# Matched baseline
python -m src.train_baseline_matched --config config_baseline_matched_topology_scvam_seed42.yaml

# Topology-supervised model (any topology config, including ablations)
python run_ablation.py config_topology_ablation_a8_scvam.yaml
```

## Evaluation

Zero-shot evaluation on STARE/CHASE_DB1/HRF requires a precomputed vessel-scale cache (see `src/precompute_scale_ratios.py`) for the SC-VAM scale-conditioning pathway:

```bash
python -m src.precompute_scale_ratios --output-json scale_ratio_cache.json --dataset-name STARE ...
```

Then, for each trained model:

```bash
python -m src.evaluate_topology_v2 \
  --config config_topology_ablation_a8_scvam.yaml \
  --checkpoint checkpoints/topology_a8_scvam_seed42/best_model.pth \
  --output-dir evaluation_outputs/topology_a8_seed42_no_tta \
  --scale-ratio-cache scale_ratio_cache.json

# Repeat with --tta for test-time augmentation
```

`src/evaluate_baseline_v2.py` takes the same flags for the matched baseline.

### Statistical testing, ablation analysis, and complexity

- `paired_significance_baseline_v3_vs_a8.py` — per-image paired Wilcoxon signed-rank test (baseline vs. A8), seeds 42 and 7.
- `paired_significance_baseline_v3_vs_a8_seed123.py` — the same test, scoped to seed 123; writes to its own output file so the seed 42/7 results are untouched.
- `src/evaluate_baseline_structural.py` — post-hoc skeleton/endpoint/junction F1 for the baseline, computed by skeletonizing its saved segmentation output (the baseline has no dedicated structural heads).
- `src/tolerance_sensitivity_analysis.py` — re-scores already-saved structural predictions at tolerance radii of 1px/2px/3px, to check whether the reported structural F1 gap is specific to the paper's chosen tolerances.
- `src/count_model_complexity.py` — trainable-parameter and FLOP comparison between the matched baseline and the full topology model, built with the same `build_model()` functions used by the evaluators above.

## Datasets

DRIVE, STARE, CHASE_DB1, and HRF are publicly available datasets; see the citations in the paper for their original sources. This repository does not redistribute them — `datasets/`, `datasets_processed/`, and `datasets_topology/` are expected locally and are git-ignored.

## Trained checkpoints and cached scale-ratio files

Trained model checkpoints and the precomputed vessel-scale cache used for the reported results are archived on Zenodo: **[DOI to be added]**.

## License

MIT — see `LICENSE`.

## Citation

If you use this code, please cite:

> J. Aliyu, H. Liu. "MV-TransUNet-HVS: Scale-Conditioned Attention and Topology-Aware Supervision for Retinal Vessel Segmentation." *Biomedical Signal Processing and Control* (submitted).
