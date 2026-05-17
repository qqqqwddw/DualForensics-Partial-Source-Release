import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.modules import HOSFireEnvelope


class TinyAuxBackbone(nn.Module):
    def __init__(self, feature_dim=16):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, feature_dim, kernel_size=3, stride=4, padding=1),
            nn.GELU(),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, stride=4, padding=1),
            nn.GELU(),
        )
        self.fc = nn.Linear(feature_dim, 1)

    def forward(self, x_unit):
        aux = self.stem(x_unit)
        feat = F.adaptive_avg_pool2d(aux, output_size=1).flatten(1)
        logit = self.fc(feat)
        return logit, aux


def main():
    torch.manual_seed(3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    module = HOSFireEnvelope(
        feature_dim=16,
        mask_hidden_dim=8,
        score_hidden_dim=8,
        topk_ratio=0.2,
        num_phase_shifts=3,
    ).to(device)
    backbone = TinyAuxBackbone(feature_dim=16).to(device)
    anchor = TinyAuxBackbone(feature_dim=16).to(device).eval()

    x_unit = torch.rand(4, 3, 64, 64, device=device)
    label = torch.tensor([0, 1, 0, 1], dtype=torch.float32, device=device)
    cls_logit, aux_map = backbone(x_unit)

    def aux_forward(z):
        return backbone(z)

    def anchor_forward(z):
        with torch.no_grad():
            _, aux = anchor(z)
        return aux

    result = module(
        x_unit=x_unit,
        aux_map_o=aux_map,
        label=label,
        aux_map_forward=aux_forward,
        cls_logit_o=cls_logit,
        anchor_aux_forward=anchor_forward,
        compute_boundary=True,
        compute_cdc=True,
        compute_anchor=True,
    )
    loss = (
        result["loss_evidence"]
        + result["loss_mask"]
        + result["loss_target"]
        + result["loss_boundary"]
        + result["loss_rank"]
        + result["loss_tangent"]
        + result["loss_cdc"]
        + result["loss_anchor"]
        + result["loss_anchor_res"]
    )
    loss.backward()

    if not torch.isfinite(loss):
        raise RuntimeError("non-finite HOS-FIRE loss")
    if result["logit"].shape != (4, 1):
        raise RuntimeError(f"unexpected logit shape: {tuple(result['logit'].shape)}")
    if module.score_net[0].weight.grad is None:
        raise RuntimeError("score head did not receive gradients")

    stats = result["stats"]
    print(
        "HOS_FIRE_ENVELOPE_SMOKE_PASS "
        f"loss={float(loss.detach().cpu()):.6f} "
        f"ev={float(result['loss_evidence'].detach().cpu()):.6f} "
        f"mask={float(result['loss_mask'].detach().cpu()):.6f} "
        f"target={float(result['loss_target'].detach().cpu()):.6f} "
        f"boundary={float(result['loss_boundary'].detach().cpu()):.6f} "
        f"rank={float(result['loss_rank'].detach().cpu()):.6f} "
        f"tangent={float(result['loss_tangent'].detach().cpu()):.6f} "
        f"cdc={float(result['loss_cdc'].detach().cpu()):.6f} "
        f"mask_density={float(stats['mask_density'].detach().cpu()):.6f}"
    )


if __name__ == "__main__":
    main()
