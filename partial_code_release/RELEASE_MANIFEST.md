# Release Manifest

## Root
- `train.py`
- `validate.py`
- `earlystop.py`
- `util.py`
- `requirements.txt`
- `dataset_paths.example.py`

## Data Support
- `data/__init__.py`
- `data/datasets.py`

## Options
- `options/__init__.py`
- `options/base_options.py`
- `options/train_options.py`
- `options/test_options.py`

## Model Files
- `models/__init__.py` (sanitized after copy)
- `models/base_model.py`
- `models/checkpoint_utils.py`
- `models/clip_models.py`
- `models/dinov3_models.py`
- `models/trainer.py`
- `models/modules/__init__.py`
- `models/modules/pixel_mapping.py`
- `models/modules/projector.py`
- `models/modules/mid_frequency.py`
- `models/modules/fire_lite.py`
- `models/modules/losses.py`
- `models/modules/noise_guidance.py`
- `models/modules/evidence_head.py`
- `models/modules/hos_fire_envelope.py`
- `models/modules/real_manifold.py`

## Scripts
- `examples/train_svd_example.sh`
- `script/smoke_test_hos_fire_envelope.py`
- `script/smoke_test_forensic_heads.py`
- `script/smoke_test_fake_hardening.py`
- `script/build_train_hard_lists_from_audit.py`
