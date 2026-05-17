import math
import os
import sys
from types import SimpleNamespace

import torch
import torch.nn as nn
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import models.trainer as trainer_module
from data.datasets import data_augment
from options.train_options import TrainOptions


class DummyHardeningModel(nn.Module):
    def __init__(self, feature_dim=32):
        super().__init__()
        self.feature_dim = feature_dim
        self.stem = nn.Sequential(
            nn.Conv2d(3, feature_dim, kernel_size=3, stride=4, padding=1),
            nn.GELU(),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, stride=4, padding=1),
            nn.GELU(),
        )
        self.fc = nn.Linear(feature_dim, 1)

    def forward(
        self,
        x,
        return_feature=False,
        return_aux_map=False,
        return_patch_tokens=False,
    ):
        aux_full = self.stem(x)
        feat = aux_full.mean(dim=(2, 3))
        logit = self.fc(feat)
        aux_map = aux_full if return_aux_map else None
        patch_tokens = aux_full.flatten(2).transpose(1, 2)
        if return_feature:
            if return_patch_tokens:
                return feat, logit, aux_map, patch_tokens
            return feat, logit, aux_map
        return logit

    def svd_regularization_losses(self):
        zero = self.fc.weight.sum() * 0.0
        return zero, zero


def build_opt():
    argv_backup = list(sys.argv)
    sys.argv = [
        sys.argv[0],
        "--gpu_ids",
        "-1",
        "--arch",
        "DINOv3:ViT-H/16plus_svd",
        "--batch_size",
        "2",
        "--loadSize",
        "224",
        "--cropSize",
        "224",
        "--fix_backbone",
        "--use_svd",
        "--use_hos_fire_envelope",
        "--lambda_hos_evidence",
        "0.01",
        "--lambda_hos_mask",
        "0.0",
        "--lambda_hos_target",
        "0.0",
        "--lambda_hos_boundary",
        "0.0",
        "--lambda_hos_rank",
        "0.0",
        "--lambda_hos_tangent",
        "0.0",
        "--lambda_hos_cdc",
        "0.0",
        "--lambda_hos_anchor",
        "0.0",
        "--lambda_hos_anchor_residual",
        "0.0",
        "--use_fake_hardening",
        "--fake_hard_start_epoch",
        "1",
        "--fake_hard_prob",
        "1.0",
        "--fake_hard_max_batch",
        "2",
        "--lambda_fake_hard",
        "0.1",
        "--lambda_fake_hard_consistency",
        "0.1",
        "--lambda_fake_hard_feat_consistency",
        "0.1",
        "--lambda_fake_hard_margin",
        "0.1",
        "--fake_hard_use_hos",
        "--use_fake_adv_hardening",
        "--fake_adv_start_epoch",
        "1",
        "--fake_adv_prob",
        "1.0",
        "--fake_adv_max_batch",
        "2",
        "--fake_adv_steps",
        "1",
        "--fake_adv_eot_views",
        "1",
        "--fake_adv_eps",
        "2",
        "--fake_adv_alpha",
        "1",
        "--fake_adv_random_start",
        "--fake_adv_on_social_chain",
        "--fake_adv_attack_use_main",
        "--fake_adv_attack_use_hos",
        "--fake_adv_train_use_hos",
        "--lambda_fake_adv",
        "0.1",
        "--lambda_fake_adv_consistency",
        "0.1",
        "--lambda_fake_adv_margin",
        "0.1",
    ]
    opt = TrainOptions().parse(print_options=False)
    sys.argv = argv_backup
    return opt


def assert_finite(name, value):
    if not math.isfinite(float(value)):
        raise AssertionError(f"{name} is not finite: {value}")


def main():
    torch.manual_seed(0)
    opt = build_opt()

    aug_opt = SimpleNamespace(
        color_aug_prob=0.5,
        brightness_range="0.9,1.1",
        contrast_range="0.9,1.1",
        saturation_range="0.9,1.1",
        resize_aug_prob=0.25,
        resize_aug_scale="0.7,1.0",
        rz_interp=["bilinear"],
        blur_prob=0.0,
        blur_sig=[0.0, 1.0],
        jpg_prob=1.0,
        jpg_method=["pil"],
        jpg_qual=[40, 60],
        use_social_chain_aug=True,
        social_chain_prob=1.0,
        social_chain_min_ops=2,
        social_chain_max_ops=4,
        use_fake_dataset_hardening=True,
        fake_dataset_hard_prob=1.0,
        fake_social_chain_min_ops=3,
        fake_social_chain_max_ops=5,
        fake_dataset_use_mid_suppress=True,
    )
    aug_img = Image.new("RGB", (256, 256), color=(127, 127, 127))
    aug_out = data_augment(aug_img, aug_opt, label=1)
    if not isinstance(aug_out, Image.Image):
        raise AssertionError("data_augment did not return a PIL image.")

    original_get_model = trainer_module.get_model
    trainer_module.get_model = lambda _name, _opt: DummyHardeningModel()
    try:
        trainer = trainer_module.Trainer(opt)
    finally:
        trainer_module.get_model = original_get_model

    trainer.train()
    x = torch.rand(2, 3, 224, 224)
    y = torch.tensor([0, 1], dtype=torch.long)
    trainer.set_input((x, y))
    trainer.optimize_parameters(epoch=0)

    required_losses = [
        "total",
        "cls",
        "fake_hard",
        "fake_hard_consistency",
        "fake_hard_feat_consistency",
        "fake_hard_margin",
        "fake_adv",
        "fake_adv_consistency",
        "fake_adv_feat_consistency",
        "fake_adv_margin",
        "hos_evidence",
    ]
    for key in required_losses:
        if key not in trainer.loss_dict:
            raise AssertionError(f"Missing loss key: {key}")
        assert_finite(key, trainer.loss_dict[key])

    if trainer.loss_dict["fake_hard_active"] < 1.0 or trainer.loss_dict["fake_adv_active"] < 1.0:
        raise AssertionError("Fake hardening branches did not activate in the smoke test.")

    trainer.eval()
    with torch.no_grad():
        trainer.opt.infer_mode = "trained_hos_fire"
        logits = trainer.predict_logits(x)
        if logits.shape != (2, 1):
            raise AssertionError(f"Unexpected logits shape: {logits.shape}")
        if not torch.isfinite(logits).all():
            raise AssertionError("trained_hos_fire produced non-finite logits.")

    print("FAKE_HARDENING_SMOKE_PASS")
    print(
        "loss total={total:.6f} cls={cls:.6f} hard={fake_hard:.6f} adv={fake_adv:.6f} "
        "hos={hos_evidence:.6f} hard_active={fake_hard_active:.1f} adv_active={fake_adv_active:.1f}".format(
            **trainer.loss_dict
        )
    )


if __name__ == "__main__":
    main()
