import torch
import torch.nn.functional as F


def info_nce_loss(anchor: torch.Tensor, positive: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    if anchor.shape != positive.shape:
        raise ValueError("anchor and positive must have the same shape.")
    logits = anchor @ positive.t()
    logits = logits / max(temperature, 1e-8)
    labels = torch.arange(anchor.shape[0], device=anchor.device)
    return F.cross_entropy(logits, labels)


def _binary_or_multiclass_probs(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 1:
        logits = logits.unsqueeze(1)
    if logits.shape[1] == 1:
        p1 = torch.sigmoid(logits)
        return torch.cat([1.0 - p1, p1], dim=1)
    return torch.softmax(logits, dim=1)


def js_divergence(logits_p: torch.Tensor, logits_q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p = _binary_or_multiclass_probs(logits_p)
    q = _binary_or_multiclass_probs(logits_q)
    m = 0.5 * (p + q)
    kl_pm = torch.sum(p * (torch.log(p + eps) - torch.log(m + eps)), dim=1).mean()
    kl_qm = torch.sum(q * (torch.log(q + eps) - torch.log(m + eps)), dim=1).mean()
    return 0.5 * (kl_pm + kl_qm)


def mid_prior_loss(pred: torch.Tensor, target: torch.Tensor, loss_type: str = "mse") -> torch.Tensor:
    if pred.shape[-2:] != target.shape[-2:]:
        target = F.interpolate(target, size=pred.shape[-2:], mode="bilinear", align_corners=False)

    loss_type = loss_type.lower()
    if loss_type == "mse":
        return F.mse_loss(pred, target)
    if loss_type == "l1":
        return F.l1_loss(pred, target)
    if loss_type == "smooth_l1":
        return F.smooth_l1_loss(pred, target)
    raise ValueError(f"Unsupported mid prior loss type: {loss_type}")
