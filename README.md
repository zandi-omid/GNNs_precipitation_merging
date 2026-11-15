# Gauge-informed graph learning for daily precipitation

This repository contains the code and compact trained-model artifacts used to
merge daily gauge observations with gridded precipitation products over the
Arizona study region. The final framework compares GATGCN, MLP, temporal graph
variants, auxiliary-only models, and an out-of-fold meta-MLP stacker.

## Experiments

- Manual sparse-gauge interpolation
- All-gauges spatial cross-fitting and future-year evaluation
- MSE versus multi-quantile training
- GATGCN, TGATGCN, and temporal-attention ablation
- Auxiliary-only ERA5/IMERG models
- Operational IMERG Early-only models
- Temporal out-of-fold stacking

Training uses 2005-2019 observations, with experiment-specific validation
folds. The untouched 2020-2024 period is used for final evaluation.

## Repository layout

- `GSI/`: graph construction, graph models, training, and inference
- `ANN/`: MLP feature construction, training, and inference
- `configs/`: representative paper experiment configurations
- `evaluation/`: scripts used for paper metrics and figures
- `scripts/`: IMERG Early download and daily aggregation utilities
- `checkpoints/`: compact final trained-model artifacts
- `data/`: external data-bundle documentation

## Installation

Create a Python environment with a CUDA-compatible PyTorch build when training
the graph models, then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
```

## Workflow

1. Obtain the external data bundle described in `data/README.md`.
2. Edit a configuration under `configs/` to point to the local data directory.
3. Build graph sequences with `GSI/build_graph_structure_GSI.py`.
4. Train or infer with `GSI/train.py` and `GSI/inference_global.py`.
5. Build and train the MLP branch with `ANN/build_df.py` and `ANN/train_MLP.py`.
6. Generate out-of-fold predictions and train the meta-MLP stacker.

For example:

```bash
python GSI/train.py --config configs/train_gatgcn_manual_sparse.toml
```

## Three-step example

The repository includes one real study-domain sequence truncated to three time
steps and its graph artifact. Run a checkpoint forward-pass smoke test with:

```bash
python examples/run_three_step_example.py
```

## Data availability

Raw third-party products remain subject to their original providers' licenses.
Processed training inputs and graph artifacts will be distributed separately
with checksums. Large predictions and intermediate checkpoints are intentionally
excluded from Git.

## Citation

Citation metadata is provided in `CITATION.cff`. The paper citation will be
added when its bibliographic details are finalized.
