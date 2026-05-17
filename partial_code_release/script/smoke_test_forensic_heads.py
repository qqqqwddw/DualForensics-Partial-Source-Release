import math
import os
import sys

import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import models.trainer as trainer_module
from options.train_options import TrainOptions


class DummyDinoLikeModel(nn.Module):
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
        "--use_svd",
        "--use_three_branch_training",
        "--use_fixed_mapping_branch",
        "--use_random_mapping_branch",
        "--random_branch_warmup_epochs",
        "0",
        "--use_multiview_fusion_head",
        "--multiview_fusion_type",
        "delta",
        "--lambda_multiview_fusion",
        "0.5",
        "--lambda_real_consistency",
        "0.01",
        "--lambda_fake_disagreement",
        "0.01",
        "--fake_disagreement_margin",
        "0.1",
        "--use_evidence_head",
        "--lambda_evidence",
        "0.1",
        "--lambda_evidence_align",
        "0.01",
        "--use_patch_mil_head",
        "--lambda_patch_mil",
        "0.1",
        "--use_cross_view_patch_disagreement",
        "--lambda_patch_disagreement",
        "0.1",
        "--use_fire_error_evidence",
        "--lambda_fire_error",
        "0.1",
        "--lambda_fire_error_align",
        "0.01",
        "--lambda_con",
        "0.01",
        "--lambda_align",
        "0.01",
        "--lambda_orth",
        "0.0",
        "--lambda_ksv",
        "0.0",
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

    original_get_model = trainer_module.get_model
    trainer_module.get_model = lambda _name, _opt: DummyDinoLikeModel()
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
        "multiview_fusion",
        "real_consistency",
        "fake_disagreement",
        "evidence",
        "patch_mil",
        "patch_disagreement",
        "fire_error",
        "fire_error_align",
    ]
    for key in required_losses:
        if key not in trainer.loss_dict:
            raise AssertionError(f"Missing loss key: {key}")
        assert_finite(key, trainer.loss_dict[key])

    trainer.eval()
    infer_modes = [
        "trained_multiview_fusion",
        "trained_patch_mil",
        "trained_delta_patch_mil_avg",
        "trained_forensic_fusion",
    ]
    with torch.no_grad():
        for mode in infer_modes:
            trainer.opt.infer_mode = mode
            logits = trainer.predict_logits(x)
            if logits.shape[0] != x.shape[0]:
                raise AssertionError(f"{mode} batch mismatch: {logits.shape}")
            if not torch.isfinite(logits).all():
                raise AssertionError(f"{mode} produced non-finite logits.")

    print("FORENSIC_HEAD_SMOKE_PASS")
    print(
        "loss total={total:.6f} cls={cls:.6f} mv={multiview_fusion:.6f} "
        "real_cons={real_consistency:.6f} fake_dis={fake_disagreement:.6f} "
        "ev={evidence:.6f} mil={patch_mil:.6f} pd={patch_disagreement:.6f} "
        "fire={fire_error:.6f}".format(**trainer.loss_dict)
    )


if __name__ == "__main__":
    main()
