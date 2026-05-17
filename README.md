# DualForensics Partial Source Release

This folder is a partial source release for the DualForensics training and model code.
It is intended for academic review, reproducibility, and non-commercial research use.

## Included

- Training entry point: `train.py`
- Validation helper: `validate.py`
- Training options: `options/`
- Minimal dataloader code required by `train.py`: `data/`
- Public model scaffolding and selected model modules: `models/`
- Selected smoke tests and hard-list builder scripts: `script/`
- Sanitized example training command: `examples/train_svd_example.sh`
- Example dataset-path template: `dataset_paths.example.py`

## Not Included

- Datasets
- Model checkpoints or pretrained weights
- Private experiment outputs
- Remote execution scripts
- Full paper-experiment orchestration scripts
- Private absolute paths from the original development environment
- Temporary debugging scripts and `__pycache__` files

## Important IP Notice

The authors retain copyright and all intellectual-property rights in the full DualForensics project.
This release exposes only a subset of the code. The absence of a file here does not imply abandonment,
waiver, or public licensing of the corresponding method, implementation detail, data, checkpoint, or experiment pipeline.

Strictly speaking, the default license in this folder is source-available for research use, not an OSI-approved open-source license.
If you need an OSI-approved open-source release, replace `LICENSE` with a standard license such as MIT, BSD-3-Clause,
Apache-2.0, or GPL after confirming the intended rights and obligations.

## Basic Usage

1. Install dependencies from `requirements.txt` or your project environment.
2. Prepare your dataset in a compatible real/fake directory layout.
3. Copy `dataset_paths.example.py` to `dataset_paths.py` if you use `validate.py` dataset-list evaluation.
4. Run the example script after editing paths:

```bash
bash examples/train_svd_example.sh
```

## Citation

If you use this partial release, cite the DualForensics paper/preprint associated with this repository.
