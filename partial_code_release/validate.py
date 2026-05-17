import argparse
import math
import os
import pickle
import random
from copy import deepcopy
from io import BytesIO
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image
from scipy.ndimage import gaussian_filter
from sklearn.metrics import accuracy_score, average_precision_score
from torch.utils.data import Dataset
from tqdm import tqdm

from dataset_paths import DATASET_PATHS
from models import get_model
from models.checkpoint_utils import load_state_dict_with_prefix_compat
from models.modules import (
    CrossViewPatchDisagreementHead,
    DeltaMultiViewFusionHead,
    FireErrorEvidenceHead,
    FixedPixelMapping,
    ForensicEvidenceHead,
    ForensicQueryMILHead,
    HOSFireEnvelope,
    MidBandMaskHead,
    MultiViewFusionHead,
    PatchMILHead,
    RandomPixelMapping,
    RealManifoldDistanceScorer,
    ReconScoreHead,
    build_mid_target_mask,
    fft_band_filter_rgb,
)


SEED = 0


def set_seed():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)


def setup_distributed_from_env():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
    elif torch.cuda.is_available():
        torch.cuda.set_device(0)
    return distributed, rank, local_rank, world_size


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


MEAN = {
    "imagenet": [0.485, 0.456, 0.406],
    "clip": [0.48145466, 0.4578275, 0.40821073],
    "beitv2": [0.485, 0.456, 0.406],
    "siglip": [0.5, 0.5, 0.5],
    "dinov3": [0.485, 0.456, 0.406],
}

STD = {
    "imagenet": [0.229, 0.224, 0.225],
    "clip": [0.26862954, 0.26130258, 0.27577711],
    "beitv2": [0.229, 0.224, 0.225],
    "siglip": [0.5, 0.5, 0.5],
    "dinov3": [0.229, 0.224, 0.225],
}


def translate_duplicate(img, cropSize):
    if min(img.size) < cropSize:
        width, height = img.size

        new_width = width * math.ceil(cropSize / width)
        new_height = height * math.ceil(cropSize / height)

        new_img = Image.new("RGB", (new_width, new_height))
        for i in range(0, new_width, width):
            for j in range(0, new_height, height):
                new_img.paste(img, (i, j))
        return new_img
    else:
        return img


def find_best_threshold(y_true, y_pred, metric="balanced_acc"):
    real_scores = y_pred[y_true == 0]
    fake_scores = y_pred[y_true == 1]

    if len(real_scores) > 0 and len(fake_scores) > 0 and real_scores.max() <= fake_scores.min():
        return float((real_scores.max() + fake_scores.min()) / 2.0)

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    order = np.argsort(y_pred, kind="mergesort")
    scores = y_pred[order]
    labels = y_true[order]

    is_real = labels == 0
    is_fake = labels == 1
    n_real = max(1, int(is_real.sum()))
    n_fake = max(1, int(is_fake.sum()))
    n_total = max(1, int(len(labels)))

    real_le_cum = np.cumsum(is_real)
    fake_le_cum = np.cumsum(is_fake)
    unique_scores, last_indices = np.unique(scores, return_index=True)
    next_indices = np.r_[last_indices[1:], len(scores)]
    last_indices = next_indices - 1

    real_le = real_le_cum[last_indices]
    fake_le = fake_le_cum[last_indices]
    fake_gt = n_fake - fake_le

    r_acc = real_le / float(n_real)
    f_acc = fake_gt / float(n_fake)
    acc = (real_le + fake_gt) / float(n_total)

    if metric == "acc":
        scores_to_max = acc
    elif metric == "fake_acc":
        scores_to_max = f_acc
    else:
        scores_to_max = 0.5 * (r_acc + f_acc)

    best_idx = int(np.argmax(scores_to_max))
    best_thres = float(unique_scores[best_idx])

    return best_thres


def png2jpg(img, quality):
    out = BytesIO()
    img.save(out, format="jpeg", quality=quality)  # ranging from 0-95, 75 is default
    img = Image.open(out)
    img = np.array(img)
    out.close()

    return Image.fromarray(img)


def gaussian_blur(img, sigma):
    img = np.array(img)

    gaussian_filter(img[:, :, 0], output=img[:, :, 0], sigma=sigma)
    gaussian_filter(img[:, :, 1], output=img[:, :, 1], sigma=sigma)
    gaussian_filter(img[:, :, 2], output=img[:, :, 2], sigma=sigma)

    return Image.fromarray(img)


def calculate_acc(y_true, y_pred, thres):
    r_acc = accuracy_score(y_true[y_true == 0], y_pred[y_true == 0] > thres)
    f_acc = accuracy_score(y_true[y_true == 1], y_pred[y_true == 1] > thres)
    acc = accuracy_score(y_true, y_pred > thres)

    return r_acc, f_acc, acc


def _distributed_concat_array(values):
    if not (dist.is_available() and dist.is_initialized()):
        return values
    gathered = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, values)
    if dist.get_rank() != 0:
        return None
    non_empty = [np.asarray(item) for item in gathered if len(item) > 0]
    if len(non_empty) == 0:
        return np.array([])
    return np.concatenate(non_empty, axis=0)


def validate(
    model,
    loader,
    find_thres=False,
    threshold_metric="balanced_acc",
    distributed_collect=False,
):
    with torch.no_grad():
        y_true, y_pred = [], []
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        is_main = rank == 0
        if is_main:
            print("Length of dataset: %d" % (len(loader)))
        if len(loader) == 0:
            if is_main:
                print("Validation loader is empty on this rank.")
        iterator = tqdm(loader, total=len(loader), desc="Eval", leave=False, dynamic_ncols=True) if is_main else loader
        for img, label in iterator:
            in_tens = img.cuda()

            if hasattr(model, "predict_logits"):
                logits = model.predict_logits(in_tens)
            else:
                logits = model(in_tens)
            y_pred.extend(torch.sigmoid(logits).flatten().tolist())
            y_true.extend(label.flatten().tolist())

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    if distributed_collect:
        y_true = _distributed_concat_array(y_true)
        y_pred = _distributed_concat_array(y_pred)
        if y_true is None or y_pred is None:
            if not find_thres:
                return None
            return None

    if len(y_true) == 0:
        if not find_thres:
            return 0.0, 0.0, 0.0, 0.0
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5

    ap = average_precision_score(y_true, y_pred)
    r_acc0, f_acc0, acc0 = calculate_acc(y_true, y_pred, 0.5)
    if not find_thres:
        return ap, r_acc0, f_acc0, acc0

    best_thres = find_best_threshold(y_true, y_pred, metric=threshold_metric)
    r_acc1, f_acc1, acc1 = calculate_acc(y_true, y_pred, best_thres)

    return ap, r_acc0, f_acc0, acc0, r_acc1, f_acc1, acc1, best_thres


def recursively_read(rootdir, must_contain, classes=[], exts=["png", "jpg", "JPEG", "jpeg"]):
    out = []
    for r, d, f in os.walk(rootdir):
        for file in f:
            if (file.split(".")[1] in exts) and (must_contain in os.path.join(r, file)):
                if len(classes) == 0:
                    out.append(os.path.join(r, file))
                elif os.path.join(r, file).split("/")[-3] in classes:
                    out.append(os.path.join(r, file))

    return out


def get_list(path, must_contain="", classes=[]):
    if ".pickle" in path:
        with open(path, "rb") as f:
            image_list = pickle.load(f)
        image_list = [item for item in image_list if must_contain in item]
    else:
        image_list = recursively_read(path, must_contain, classes)

    return image_list


class RealFakeDataset(Dataset):
    def __init__(
        self,
        real_path,
        fake_path,
        data_mode,
        max_sample,
        arch,
        jpeg_quality=None,
        gaussian_sigma=None,
    ):

        assert data_mode in ["wang2020", "ours"]
        self.jpeg_quality = jpeg_quality
        self.gaussian_sigma = gaussian_sigma

        if type(real_path) == str and type(fake_path) == str:
            real_list, fake_list = self.read_path(real_path, fake_path, data_mode, max_sample)
        else:
            real_list = []
            fake_list = []
            for real_p, fake_p in zip(real_path, fake_path):
                real_l, fake_l = self.read_path(real_p, fake_p, data_mode, max_sample)
                real_list += real_l
                fake_list += fake_l

        self.total_list = real_list + fake_list

        self.labels_dict = {}
        for i in real_list:
            self.labels_dict[i] = 0
        for i in fake_list:
            self.labels_dict[i] = 1

        stat_from = get_stat_source(arch)
        self.transform = transforms.Compose(
            [
                transforms.Lambda(lambda img: translate_duplicate(img, 256)),
                transforms.CenterCrop(224) if stat_from != "siglip" else transforms.CenterCrop(256),
                transforms.ToTensor(),
                transforms.Normalize(mean=MEAN[stat_from], std=STD[stat_from]),
            ]
        )

    def read_path(self, real_path, fake_path, data_mode, max_sample):
        if data_mode == "wang2020":
            real_list = get_list(real_path, must_contain="0_real")
            fake_list = get_list(fake_path, must_contain="1_fake")
        else:
            real_list = get_list(real_path)
            fake_list = get_list(fake_path)

        if max_sample is not None and max_sample < 0:
            max_sample = None
        if max_sample is not None:
            if (max_sample > len(real_list)) or (max_sample > len(fake_list)):
                max_sample = 100
                print("not enough images, max_sample falling to 100")
            random.shuffle(real_list)
            random.shuffle(fake_list)
            real_list = real_list[0:max_sample]
            fake_list = fake_list[0:max_sample]

        if len(real_list) == 0 or len(fake_list) == 0:
            raise ValueError(
                f"Both real and fake lists must be non-empty. "
                f"Got real={len(real_list)}, fake={len(fake_list)}."
            )
        if len(real_list) != len(fake_list):
            print(
                f"Warning: unbalanced evaluation set: real={len(real_list)}, "
                f"fake={len(fake_list)}. Overall ACC will be sample-weighted; "
                f"real/fake ACC and balanced-threshold metrics are still reported."
            )

        return real_list, fake_list

    def __len__(self):
        return len(self.total_list)

    def __getitem__(self, idx):
        img_path = self.total_list[idx]

        label = self.labels_dict[img_path]
        img = Image.open(img_path).convert("RGB")

        if self.gaussian_sigma is not None:
            img = gaussian_blur(img, self.gaussian_sigma)
        if self.jpeg_quality is not None:
            img = png2jpg(img, self.jpeg_quality)

        img = self.transform(img)

        return img, label


def get_stat_source(arch: str) -> str:
    arch = arch.lower()
    if arch.startswith("imagenet"):
        return "imagenet"
    if arch.startswith("clip"):
        return "clip"
    if arch.startswith("siglip"):
        return "siglip"
    if arch.startswith("beitv2"):
        return "beitv2"
    if arch.startswith("dinov3"):
        return "dinov3"
    return "clip"


def logits_to_prob(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 1:
        return torch.sigmoid(logits)
    if logits.shape[1] == 1:
        return torch.sigmoid(logits.squeeze(1))
    return torch.softmax(logits, dim=1)[:, 1]


def logits_to_binary_logit(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 1:
        return logits
    if logits.shape[1] == 1:
        return logits.squeeze(1)
    return logits[:, 1] - logits[:, 0]


class FireLiteFusionPredictor:
    def __init__(
        self,
        model: torch.nn.Module,
        arch: str,
        modes: Sequence[str],
        fixed_alpha: float,
        gated_delta: float,
        gated_alpha_max: float,
        or_delta: float,
        or_t_cls: float,
        or_t_aux: float,
        mid_band_mask_head: torch.nn.Module = None,
        recon_score_head: torch.nn.Module = None,
        evidence_head: torch.nn.Module = None,
        multiview_fusion_head: torch.nn.Module = None,
        patch_mil_head: torch.nn.Module = None,
        query_mil_head: torch.nn.Module = None,
        patch_disagreement_head: torch.nn.Module = None,
        fire_error_head: torch.nn.Module = None,
        hos_fire_envelope: torch.nn.Module = None,
        real_manifold_scorer: torch.nn.Module = None,
        infer_mode: str = "original",
        infer_random_views: int = 2,
        infer_uncertain_delta: float = 0.12,
        infer_multiview_alpha: float = 0.5,
        mid_freq_radius_low_ratio: float = 0.15,
        mid_freq_radius_high_ratio: float = 0.45,
        hos_logit_scale: float = 1.0,
        hos_logit_bias: float = 0.0,
        real_manifold_logit_scale: float = 1.0,
        real_manifold_logit_bias: float = 0.0,
        logit_fusion_weight_original: float = 1.0,
        logit_fusion_weight_hos: float = 1.0,
        logit_fusion_weight_manifold: float = 1.0,
        logit_fusion_weight_patch: float = 0.0,
        logit_fusion_weight_evidence: float = 0.0,
        logit_fusion_bias: float = 0.0,
    ):
        self.model = model
        self.modes = list(modes)
        self.infer_mode = infer_mode
        self.infer_random_views = int(infer_random_views)
        self.infer_uncertain_delta = float(infer_uncertain_delta)
        self.infer_multiview_alpha = float(infer_multiview_alpha)
        self.fixed_alpha = float(fixed_alpha)
        self.gated_delta = float(gated_delta)
        self.gated_alpha_max = float(gated_alpha_max)
        self.or_delta = float(or_delta)
        self.or_t_cls = float(or_t_cls)
        self.or_t_aux = float(or_t_aux)

        stat_from = get_stat_source(arch)
        self.norm_mean = torch.tensor(MEAN[stat_from], dtype=torch.float32).view(1, 3, 1, 1).cuda()
        self.norm_std = torch.tensor(STD[stat_from], dtype=torch.float32).view(1, 3, 1, 1).cuda()

        self.mid_band_mask_head = mid_band_mask_head
        self.recon_score_head = recon_score_head
        self.evidence_head = evidence_head
        self.multiview_fusion_head = multiview_fusion_head
        self.patch_mil_head = patch_mil_head
        self.query_mil_head = query_mil_head
        self.patch_disagreement_head = patch_disagreement_head
        self.fire_error_head = fire_error_head
        self.hos_fire_envelope = hos_fire_envelope
        self.real_manifold_scorer = real_manifold_scorer
        self.mid_freq_radius_low_ratio = float(mid_freq_radius_low_ratio)
        self.mid_freq_radius_high_ratio = float(mid_freq_radius_high_ratio)
        self.hos_logit_scale = float(hos_logit_scale)
        self.hos_logit_bias = float(hos_logit_bias)
        self.real_manifold_logit_scale = float(real_manifold_logit_scale)
        self.real_manifold_logit_bias = float(real_manifold_logit_bias)
        self.logit_fusion_weight_original = float(logit_fusion_weight_original)
        self.logit_fusion_weight_hos = float(logit_fusion_weight_hos)
        self.logit_fusion_weight_manifold = float(logit_fusion_weight_manifold)
        self.logit_fusion_weight_patch = float(logit_fusion_weight_patch)
        self.logit_fusion_weight_evidence = float(logit_fusion_weight_evidence)
        self.logit_fusion_bias = float(logit_fusion_bias)
        self.fixed_mapping = FixedPixelMapping().cuda()
        self.random_mapping = RandomPixelMapping().cuda()

    def _to_unit(self, x: torch.Tensor) -> torch.Tensor:
        return (x * self.norm_std.to(x.dtype) + self.norm_mean.to(x.dtype)).clamp(0.0, 1.0)

    def _from_unit(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.norm_mean.to(x.dtype)) / (self.norm_std.to(x.dtype) + 1e-8)

    def _forward_branch(self, x: torch.Tensor, return_aux_map: bool = True):
        feat, logit, aux_map = self.model(x, return_feature=True, return_aux_map=return_aux_map)
        return feat, logit, aux_map

    def _forward_branch_with_tokens(self, x: torch.Tensor, return_aux_map: bool = True):
        feat, logit, aux_map, patch_tokens = self.model(
            x,
            return_feature=True,
            return_aux_map=return_aux_map,
            return_patch_tokens=True,
        )
        return feat, logit, aux_map, patch_tokens

    def _branch_prob(self, x: torch.Tensor) -> torch.Tensor:
        return logits_to_prob(self.model(x))

    def _fire_error_prob(self, x_unit: torch.Tensor, aux_map_o: torch.Tensor) -> torch.Tensor:
        if self.fire_error_head is None:
            raise RuntimeError("fire_error_head is required for FIRE error evidence inference.")
        target_mask = build_mid_target_mask(
            batch_size=x_unit.shape[0],
            height=x_unit.shape[-2],
            width=x_unit.shape[-1],
            low_ratio=self.mid_freq_radius_low_ratio,
            high_ratio=self.mid_freq_radius_high_ratio,
            device=x_unit.device,
            dtype=x_unit.dtype,
        )
        x_mid_unit = fft_band_filter_rgb(x_unit, target_mask)
        x_mid = self._from_unit(x_mid_unit)
        _, _, aux_map_mid = self._forward_branch(x_mid)
        if aux_map_mid is None:
            raise RuntimeError("Backbone did not return aux_map for FIRE error evidence.")
        delta_map = torch.mean(torch.abs(aux_map_o - aux_map_mid), dim=1, keepdim=True)
        return torch.sigmoid(self.fire_error_head(delta_map).squeeze(1))

    def _forward_unit_for_aux(self, x_unit: torch.Tensor):
        x = self._from_unit(x_unit)
        _, logit, aux_map = self._forward_branch(x, return_aux_map=True)
        return logit, aux_map

    def _hos_fire_logit(self, x_unit: torch.Tensor, aux_map_o: torch.Tensor) -> torch.Tensor:
        if self.hos_fire_envelope is None:
            raise RuntimeError("hos_fire_envelope is required for HOS-FIRE inference.")
        result = self.hos_fire_envelope(
            x_unit=x_unit,
            aux_map_o=aux_map_o,
            label=torch.zeros(x_unit.shape[0], device=x_unit.device, dtype=x_unit.dtype),
            aux_map_forward=self._forward_unit_for_aux,
            cls_logit_o=None,
            anchor_aux_forward=None,
            compute_boundary=False,
            compute_cdc=False,
            compute_anchor=False,
        )
        logit = logits_to_binary_logit(result["logit"])
        return self.hos_logit_scale * logit + self.hos_logit_bias

    def _hos_fire_prob(self, x_unit: torch.Tensor, aux_map_o: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self._hos_fire_logit(x_unit, aux_map_o))

    def _real_manifold_logit(self, feat: torch.Tensor, patch_tokens: torch.Tensor) -> torch.Tensor:
        if self.real_manifold_scorer is None:
            raise RuntimeError("real_manifold_scorer is required for real-manifold inference.")
        logit = logits_to_binary_logit(self.real_manifold_scorer(feat, patch_tokens))
        return self.real_manifold_logit_scale * logit + self.real_manifold_logit_bias

    def _weighted_logit_fusion(self, parts: List[torch.Tensor], weights: List[float]) -> torch.Tensor:
        if len(parts) == 0:
            raise RuntimeError("No logits were provided for logit fusion.")
        fused = torch.zeros_like(parts[0])
        norm = 0.0
        for logit, weight in zip(parts, weights):
            if abs(float(weight)) < 1e-12:
                continue
            fused = fused + float(weight) * logit
            norm += abs(float(weight))
        if norm <= 1e-12:
            raise RuntimeError("All logit fusion weights are zero.")
        return fused / norm + self.logit_fusion_bias

    def _predict_multiview_prob(self, x: torch.Tensor) -> torch.Tensor:
        if self.infer_mode == "original":
            return self._branch_prob(x)

        token_modes = (
            "trained_evidence_avg",
            "trained_multiview_evidence_avg",
            "trained_patch_mil",
            "trained_query_mil",
            "trained_delta_patch_mil_avg",
            "trained_forensic_fusion",
            "trained_hos_fire",
            "trained_forensic_hos_fusion",
            "trained_hos_selective_or",
            "trained_real_manifold",
            "trained_hos_manifold_logit_fusion",
            "trained_forensic_hos_logit_fusion",
        )
        if self.infer_mode in token_modes:
            need_aux = (
                self.infer_mode in (
                    "trained_hos_fire",
                    "trained_forensic_hos_fusion",
                    "trained_hos_selective_or",
                    "trained_hos_manifold_logit_fusion",
                    "trained_forensic_hos_logit_fusion",
                )
                or (
                    self.infer_mode in (
                        "trained_forensic_fusion",
                        "trained_forensic_hos_fusion",
                        "trained_forensic_hos_logit_fusion",
                    )
                    and self.fire_error_head is not None
                )
            )
            feat, logit, aux_map_o, patch_tokens = self._forward_branch_with_tokens(
                x,
                return_aux_map=need_aux,
            )
        else:
            feat = logit = aux_map_o = patch_tokens = None

        if self.infer_mode == "trained_hos_fire":
            if aux_map_o is None:
                _, _, aux_map_o = self._forward_branch(x, return_aux_map=True)
            return self._hos_fire_prob(self._to_unit(x), aux_map_o)

        if self.infer_mode == "trained_hos_selective_or":
            x_unit = self._to_unit(x)
            if aux_map_o is None:
                _, _, aux_map_o = self._forward_branch(x, return_aux_map=True)
            p_main = logits_to_prob(logit)
            p_hos = self._hos_fire_prob(x_unit, aux_map_o)
            main_fake = p_main > self.or_t_cls
            override = (~main_fake) & (p_hos >= self.or_t_aux)
            p_final = torch.zeros_like(p_main)
            p_final[main_fake | override] = 1.0
            return p_final

        if self.infer_mode == "trained_real_manifold":
            return torch.sigmoid(self._real_manifold_logit(feat, patch_tokens))

        if self.infer_mode in ("trained_hos_manifold_logit_fusion", "trained_forensic_hos_logit_fusion"):
            x_unit = self._to_unit(x)
            if aux_map_o is None:
                _, _, aux_map_o = self._forward_branch(x, return_aux_map=True)
            parts = [
                logits_to_binary_logit(logit),
                self._hos_fire_logit(x_unit, aux_map_o),
            ]
            weights = [
                self.logit_fusion_weight_original,
                self.logit_fusion_weight_hos,
            ]
            if self.real_manifold_scorer is not None:
                parts.append(self._real_manifold_logit(feat, patch_tokens))
                weights.append(self.logit_fusion_weight_manifold)
            if self.infer_mode == "trained_forensic_hos_logit_fusion":
                if self.patch_mil_head is not None and patch_tokens is not None:
                    parts.append(logits_to_binary_logit(self.patch_mil_head(patch_tokens)))
                    weights.append(self.logit_fusion_weight_patch)
                if self.evidence_head is not None and patch_tokens is not None:
                    parts.append(logits_to_binary_logit(self.evidence_head(feat, patch_tokens, x_unit)))
                    weights.append(self.logit_fusion_weight_evidence)
            return torch.sigmoid(self._weighted_logit_fusion(parts, weights))

        if self.infer_mode == "trained_forensic_hos_fusion":
            x_unit = self._to_unit(x)
            if aux_map_o is None:
                _, _, aux_map_o = self._forward_branch(x, return_aux_map=True)
            probs = [logits_to_prob(logit), self._hos_fire_prob(x_unit, aux_map_o)]
            if self.patch_mil_head is not None and patch_tokens is not None:
                probs.append(torch.sigmoid(self.patch_mil_head(patch_tokens).squeeze(1)))
            if self.evidence_head is not None and patch_tokens is not None:
                probs.append(torch.sigmoid(self.evidence_head(feat, patch_tokens, x_unit).squeeze(1)))
            return torch.stack(probs, dim=0).mean(dim=0)

        if self.infer_mode in ("trained_evidence_avg", "trained_multiview_evidence_avg"):
            if self.evidence_head is None:
                raise RuntimeError(f"infer_mode={self.infer_mode} requires evidence_head in checkpoint.")
            x_unit = self._to_unit(x)
            p_ev = torch.sigmoid(self.evidence_head(feat, patch_tokens, x_unit).squeeze(1))
            if self.infer_mode == "trained_evidence_avg":
                return 0.5 * (logits_to_prob(logit) + p_ev)
        else:
            p_ev = None

        p_orig = self._branch_prob(x)
        x_unit = self._to_unit(x)
        probs = [p_orig]

        if self.infer_mode in ("trained_multiview_fusion", "trained_multiview_evidence_avg"):
            if self.multiview_fusion_head is None:
                raise RuntimeError(f"infer_mode={self.infer_mode} requires multiview_fusion_head in checkpoint.")
            x_fixed = self._from_unit(self.fixed_mapping(x_unit))
            x_random = self._from_unit(self.random_mapping(x_unit))
            feat_o, logit_o, _ = self._forward_branch(x)
            feat_f, logit_f, _ = self._forward_branch(x_fixed)
            feat_r, logit_r, _ = self._forward_branch(x_random)
            p_mv = torch.sigmoid(
                self.multiview_fusion_head(
                    [feat_o, feat_f, feat_r],
                    [logit_o, logit_f, logit_r],
                ).squeeze(1)
            )
            if self.infer_mode == "trained_multiview_fusion":
                return p_mv
            return 0.5 * (p_mv + p_ev)

        if self.infer_mode in ("trained_patch_mil", "trained_delta_patch_mil_avg", "trained_forensic_fusion"):
            if self.patch_mil_head is None:
                raise RuntimeError(f"infer_mode={self.infer_mode} requires patch_mil_head in checkpoint.")
            p_mil = torch.sigmoid(self.patch_mil_head(patch_tokens).squeeze(1))
            if self.infer_mode == "trained_patch_mil":
                return p_mil
            if self.multiview_fusion_head is None:
                raise RuntimeError(f"infer_mode={self.infer_mode} requires multiview_fusion_head in checkpoint.")

            x_unit = self._to_unit(x)
            x_fixed = self._from_unit(self.fixed_mapping(x_unit))
            x_random = self._from_unit(self.random_mapping(x_unit))
            need_pd_tokens = self.infer_mode == "trained_forensic_fusion" and self.patch_disagreement_head is not None
            if need_pd_tokens:
                feat_f, logit_f, _, patch_tokens_f = self._forward_branch_with_tokens(
                    x_fixed,
                    return_aux_map=False,
                )
                feat_r, logit_r, _, patch_tokens_r = self._forward_branch_with_tokens(
                    x_random,
                    return_aux_map=False,
                )
            else:
                feat_f, logit_f, _ = self._forward_branch(x_fixed, return_aux_map=False)
                feat_r, logit_r, _ = self._forward_branch(x_random, return_aux_map=False)
                patch_tokens_f = patch_tokens_r = None
            p_mv = torch.sigmoid(
                self.multiview_fusion_head(
                    [feat, feat_f, feat_r],
                    [logit, logit_f, logit_r],
                ).squeeze(1)
            )
            probs = [p_mv, p_mil]
            if self.infer_mode == "trained_forensic_fusion":
                if self.evidence_head is not None:
                    probs.append(torch.sigmoid(self.evidence_head(feat, patch_tokens, x_unit).squeeze(1)))
                if self.patch_disagreement_head is not None and patch_tokens_f is not None and patch_tokens_r is not None:
                    probs.append(
                        torch.sigmoid(
                            self.patch_disagreement_head(
                                patch_tokens,
                                patch_tokens_f,
                                patch_tokens_r,
                            ).squeeze(1)
                        )
                    )
                if self.fire_error_head is not None:
                    if aux_map_o is None:
                        _, _, aux_map_o = self._forward_branch(x, return_aux_map=True)
                    probs.append(self._fire_error_prob(x_unit, aux_map_o))
            return torch.stack(probs, dim=0).mean(dim=0)

        if self.infer_mode == "trained_query_mil":
            if self.query_mil_head is None:
                raise RuntimeError(f"infer_mode={self.infer_mode} requires query_mil_head in checkpoint.")
            return torch.sigmoid(self.query_mil_head(patch_tokens).squeeze(1))

        x_fixed = self._from_unit(self.fixed_mapping(x_unit))
        probs.append(self._branch_prob(x_fixed))

        if self.infer_mode in ("original_fixed_random_avg", "original_fixed_random_max", "uncertain_multiview"):
            for _ in range(max(1, self.infer_random_views)):
                x_random = self._from_unit(self.random_mapping(x_unit))
                probs.append(self._branch_prob(x_random))

        if self.infer_mode in ("original_fixed_avg", "original_fixed_random_avg"):
            return torch.stack(probs, dim=0).mean(dim=0)
        if self.infer_mode == "original_fixed_random_max":
            return torch.stack(probs, dim=0).amax(dim=0)
        if self.infer_mode == "uncertain_multiview":
            p_multi = torch.stack(probs, dim=0).mean(dim=0)
            uncertain = torch.abs(p_orig - 0.5) < self.infer_uncertain_delta
            p = p_orig.clone()
            p[uncertain] = (1.0 - self.infer_multiview_alpha) * p_orig[uncertain] + self.infer_multiview_alpha * p_multi[uncertain]
            return p
        raise ValueError(f"Unsupported infer_mode: {self.infer_mode}")

    def _compute_p_aux(self, x: torch.Tensor, aux_map_o: torch.Tensor) -> torch.Tensor:
        if self.mid_band_mask_head is None or self.recon_score_head is None:
            raise RuntimeError("FIRE-Lite heads are required for auxiliary fusion modes.")
        if aux_map_o is None:
            raise RuntimeError("Backbone did not return aux_map; cannot compute FIRE-Lite auxiliary score.")

        x_unit = self._to_unit(x)
        m_mid = self.mid_band_mask_head(aux_map_o)
        m_comp = 1.0 - m_mid
        m_comp_up = F.interpolate(m_comp, size=x_unit.shape[-2:], mode="bilinear", align_corners=False)

        x_pse_unit = fft_band_filter_rgb(x_unit, m_comp_up)
        x_pse = self._from_unit(x_pse_unit)

        _, _, aux_map_p = self._forward_branch(x_pse)
        if aux_map_p is None:
            raise RuntimeError("Pseudo branch did not produce aux_map.")

        delta_map = torch.mean(torch.abs(aux_map_o - aux_map_p), dim=1, keepdim=True)
        score_logit = self.recon_score_head(delta_map).squeeze(1)
        return torch.sigmoid(score_logit)

    def predict_batch(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.modes == ["orig"]:
            p_cls = self._predict_multiview_prob(x)
            return {"orig": p_cls}

        _, logit_o, aux_map_o = self._forward_branch(x)
        p_cls = self._predict_multiview_prob(x)

        out: Dict[str, torch.Tensor] = {}
        if "orig" in self.modes:
            out["orig"] = p_cls

        p_aux_full = None
        if "fixed_weight" in self.modes:
            p_aux_full = self._compute_p_aux(x, aux_map_o)
            out["fixed_weight"] = (1.0 - self.fixed_alpha) * p_cls + self.fixed_alpha * p_aux_full

        if "gated_weight" in self.modes:
            conf = torch.abs(p_cls - 0.5)
            uncertain = conf < self.gated_delta
            p_final = p_cls.clone()
            if uncertain.any():
                if p_aux_full is None:
                    p_aux_uncertain = self._compute_p_aux(x[uncertain], aux_map_o[uncertain])
                else:
                    p_aux_uncertain = p_aux_full[uncertain]
                alpha = self.gated_alpha_max * (1.0 - conf[uncertain] / max(self.gated_delta, 1e-6))
                p_final[uncertain] = (1.0 - alpha) * p_cls[uncertain] + alpha * p_aux_uncertain
            out["gated_weight"] = p_final

        if "uncertain_or" in self.modes:
            uncertain = torch.abs(p_cls - 0.5) < self.or_delta
            p_final = p_cls.clone()
            if uncertain.any():
                if p_aux_full is None:
                    p_aux_uncertain = self._compute_p_aux(x[uncertain], aux_map_o[uncertain])
                else:
                    p_aux_uncertain = p_aux_full[uncertain]
                fake_or = (p_cls[uncertain] > self.or_t_cls) | (p_aux_uncertain > self.or_t_aux)
                p_final[uncertain] = fake_or.to(dtype=p_final.dtype)
            out["uncertain_or"] = p_final

        if "selective_or" in self.modes:
            x_unit = self._to_unit(x)
            p_main = logits_to_prob(logit_o)
            if aux_map_o is None:
                _, _, aux_map_o = self._forward_branch(x, return_aux_map=True)
            p_hos = self._hos_fire_prob(x_unit, aux_map_o)
            main_fake = p_main > self.or_t_cls
            override = (~main_fake) & (p_hos >= self.or_t_aux)
            p_final = torch.zeros_like(p_main)
            p_final[main_fake | override] = 1.0
            out["selective_or"] = p_final

        return out


def summarize_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    ap = average_precision_score(y_true, y_pred)
    r_acc0, f_acc0, acc0 = calculate_acc(y_true, y_pred, 0.5)
    best_thres = find_best_threshold(y_true, y_pred)
    r_acc1, f_acc1, acc1 = calculate_acc(y_true, y_pred, best_thres)
    return {
        "ap": float(ap),
        "r_acc0": float(r_acc0),
        "f_acc0": float(f_acc0),
        "acc0": float(acc0),
        "best_thres": float(best_thres),
        "r_acc1": float(r_acc1),
        "f_acc1": float(f_acc1),
        "acc1": float(acc1),
    }


def validate_with_fusion(
    predictor: FireLiteFusionPredictor,
    loader,
    modes: Sequence[str],
    distributed_collect: bool = False,
    is_main: bool = True,
) -> Dict[str, Dict[str, float]]:
    with torch.no_grad():
        if is_main:
            print("Length of dataset: %d" % (len(loader)))
        if len(loader) == 0:
            if is_main:
                print("Validation loader is empty.")
            return {
                mode: {
                    "ap": 0.0,
                    "r_acc0": 0.0,
                    "f_acc0": 0.0,
                    "acc0": 0.0,
                    "best_thres": 0.5,
                    "r_acc1": 0.0,
                    "f_acc1": 0.0,
                    "acc1": 0.0,
                }
                for mode in modes
            }

        y_true: List[float] = []
        y_pred: Dict[str, List[float]] = {mode: [] for mode in modes}

        iterator = tqdm(loader, total=len(loader), desc="Eval", leave=False, dynamic_ncols=True) if is_main else loader
        for img, label in iterator:
            in_tens = img.cuda(non_blocking=True)
            preds = predictor.predict_batch(in_tens)
            for mode in modes:
                y_pred[mode].extend(preds[mode].detach().cpu().flatten().tolist())
            y_true.extend(label.flatten().tolist())

    y_true_np = np.array(y_true)
    if distributed_collect:
        y_true_np = _distributed_concat_array(y_true_np)
        gathered_pred = {}
        for mode in modes:
            gathered_pred[mode] = _distributed_concat_array(np.array(y_pred[mode]))
        if y_true_np is None:
            return {}
        y_pred_np_by_mode = gathered_pred
    else:
        y_pred_np_by_mode = {mode: np.array(y_pred[mode]) for mode in modes}

    metrics = {}
    for mode in modes:
        y_pred_np = y_pred_np_by_mode[mode]
        metrics[mode] = summarize_binary_metrics(y_true_np, y_pred_np)
    return metrics


def get_result_paths(result_folder: str, mode: str, single_orig: bool) -> Dict[str, str]:
    if single_orig and mode == "orig":
        return {
            "ap": os.path.join(result_folder, "ap.txt"),
            "acc0": os.path.join(result_folder, "acc0.txt"),
            "acc1": os.path.join(result_folder, "acc1.txt"),
        }
    suffix = f"_{mode}"
    return {
        "ap": os.path.join(result_folder, f"ap{suffix}.txt"),
        "acc0": os.path.join(result_folder, f"acc0{suffix}.txt"),
        "acc1": os.path.join(result_folder, f"acc1{suffix}.txt"),
    }


def infer_fire_head_dims_from_ckpt(checkpoint, default_mask_hidden: int, default_recon_hidden: int):
    mask_in_channels = None
    mask_hidden = default_mask_hidden
    recon_in_channels = 1
    recon_hidden = default_recon_hidden

    mask_sd = checkpoint.get("mid_band_mask_head", None)
    if isinstance(mask_sd, dict) and "net.0.weight" in mask_sd:
        w = mask_sd["net.0.weight"]
        if w.ndim == 4:
            mask_hidden = int(w.shape[0])
            mask_in_channels = int(w.shape[1])

    recon_sd = checkpoint.get("recon_score_head", None)
    if isinstance(recon_sd, dict) and "conv.0.weight" in recon_sd:
        w = recon_sd["conv.0.weight"]
        if w.ndim == 4:
            recon_hidden = int(w.shape[0])
            recon_in_channels = int(w.shape[1])

    return mask_in_channels, mask_hidden, recon_in_channels, recon_hidden


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--real_path", type=str, default=None, help="dir name or a pickle")
    parser.add_argument("--fake_path", type=str, default=None, help="dir name or a pickle")
    parser.add_argument("--data_mode", type=str, default=None, help="wang2020 or ours")
    parser.add_argument(
        "--max_sample",
        type=int,
        default=1000,
        help="only check this number of images for both fake/real; -1 uses all images",
    )

    parser.add_argument("--arch", type=str, default="res50")
    parser.add_argument("--ckpt", type=str, default="./pretrained_weights/fc_weights.pth")
    parser.add_argument("--dinov3_repo_dir", type=str, default="./dinov3-main/dinov3-main")
    parser.add_argument("--dinov3_weights", type=str, default="")

    parser.add_argument("--result_folder", type=str, default="result", help="")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument(
        "--jpeg_quality",
        type=int,
        default=None,
        help="100, 90, 80, ... 30. Used to test robustness of our model. Not apply if None",
    )
    parser.add_argument(
        "--gaussian_sigma",
        type=int,
        default=None,
        help="0,1,2,3,4. Used to test robustness of our model. Not apply if None",
    )

    parser.add_argument(
        "--infer_mode",
        type=str,
        default="original",
        choices=[
            "original",
            "original_fixed_avg",
            "original_fixed_random_avg",
            "original_fixed_random_max",
            "uncertain_multiview",
            "trained_multiview_fusion",
            "trained_evidence_avg",
            "trained_multiview_evidence_avg",
            "trained_patch_mil",
            "trained_query_mil",
            "trained_delta_patch_mil_avg",
            "trained_forensic_fusion",
            "trained_hos_fire",
            "trained_forensic_hos_fusion",
            "trained_hos_selective_or",
            "trained_real_manifold",
            "trained_hos_manifold_logit_fusion",
            "trained_forensic_hos_logit_fusion",
        ],
    )
    parser.add_argument("--infer_random_views", type=int, default=2)
    parser.add_argument("--infer_uncertain_delta", type=float, default=0.12)
    parser.add_argument("--infer_multiview_alpha", type=float, default=0.5)
    parser.add_argument(
        "--fusion_mode",
        type=str,
        default="orig",
        choices=["orig", "fixed_weight", "gated_weight", "uncertain_or", "selective_or", "all"],
        help="multi-strategy inference mode",
    )
    parser.add_argument("--fusion_fixed_alpha", type=float, default=0.2)
    parser.add_argument("--fusion_gated_delta", type=float, default=0.12)
    parser.add_argument("--fusion_gated_alpha_max", type=float, default=0.35)
    parser.add_argument("--fusion_or_delta", type=float, default=0.10)
    parser.add_argument("--fusion_or_t_cls", type=float, default=0.50)
    parser.add_argument("--fusion_or_t_aux", type=float, default=0.58)

    parser.add_argument("--fire_mask_hidden_dim", type=int, default=128)
    parser.add_argument("--fire_recon_hidden_dim", type=int, default=64)
    parser.add_argument("--evidence_hidden_dim", type=int, default=256)
    parser.add_argument("--multiview_fusion_hidden_dim", type=int, default=256)
    parser.add_argument("--multiview_fusion_type", type=str, default="concat", choices=["concat", "delta"])
    parser.add_argument("--patch_mil_hidden_dim", type=int, default=256)
    parser.add_argument("--patch_mil_topk_ratio", type=float, default=0.10)
    parser.add_argument("--query_mil_hidden_dim", type=int, default=256)
    parser.add_argument("--query_mil_num_queries", type=int, default=4)
    parser.add_argument("--query_mil_num_heads", type=int, default=8)
    parser.add_argument("--query_mil_dropout", type=float, default=0.0)
    parser.add_argument("--query_mil_topk_ratio", type=float, default=0.50)
    parser.add_argument("--patch_disagreement_hidden_dim", type=int, default=256)
    parser.add_argument("--patch_disagreement_topk_ratio", type=float, default=0.10)
    parser.add_argument("--fire_error_hidden_dim", type=int, default=64)
    parser.add_argument("--fire_error_topk_ratio", type=float, default=0.10)
    parser.add_argument("--hos_mask_hidden_dim", type=int, default=64)
    parser.add_argument("--hos_score_hidden_dim", type=int, default=64)
    parser.add_argument("--hos_topk_ratio", type=float, default=0.10)
    parser.add_argument("--hos_num_phase_shifts", type=int, default=6)
    parser.add_argument("--hos_boundary_phase_eps", type=float, default=0.12)
    parser.add_argument("--hos_boundary_margin", type=float, default=0.25)
    parser.add_argument("--hos_cdc_degrade_prob", type=float, default=1.0)
    parser.add_argument("--hos_cdc_noise_std", type=float, default=0.015)
    parser.add_argument("--hos_cdc_min_scale", type=float, default=0.50)
    parser.add_argument("--mid_freq_radius_low_ratio", type=float, default=0.15)
    parser.add_argument("--mid_freq_radius_high_ratio", type=float, default=0.45)
    parser.add_argument("--real_manifold_stats", type=str, default="")
    parser.add_argument("--hos_logit_scale", type=float, default=1.0)
    parser.add_argument("--hos_logit_bias", type=float, default=0.0)
    parser.add_argument("--real_manifold_logit_scale", type=float, default=1.0)
    parser.add_argument("--real_manifold_logit_bias", type=float, default=0.0)
    parser.add_argument("--logit_fusion_weight_original", type=float, default=1.0)
    parser.add_argument("--logit_fusion_weight_hos", type=float, default=1.0)
    parser.add_argument("--logit_fusion_weight_manifold", type=float, default=1.0)
    parser.add_argument("--logit_fusion_weight_patch", type=float, default=0.0)
    parser.add_argument("--logit_fusion_weight_evidence", type=float, default=0.0)
    parser.add_argument("--logit_fusion_bias", type=float, default=0.0)

    parser.add_argument("--use_svd", action="store_true")
    parser.add_argument("--svd_low_rank_forward", action="store_true")
    parser.add_argument("--svd_residual_rank", type=int, default=1)
    parser.add_argument("--svd_keep_rank_ratio", type=float, default=0.999)
    parser.add_argument("--use_lora", action="store_true")

    opt = parser.parse_args()
    distributed, rank, local_rank, world_size = setup_distributed_from_env()
    is_main = rank == 0

    if is_main:
        os.makedirs(opt.result_folder, exist_ok=True)
    if distributed:
        dist.barrier()

    mode_list = (
        ["orig", "fixed_weight", "gated_weight", "uncertain_or", "selective_or"]
        if opt.fusion_mode == "all"
        else [opt.fusion_mode]
    )

    model = get_model(opt.arch, opt)
    checkpoint = torch.load(opt.ckpt, map_location="cpu")
    checkpoint_config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    for key in (
        "evidence_hidden_dim",
        "multiview_fusion_hidden_dim",
        "multiview_fusion_type",
        "patch_mil_hidden_dim",
        "patch_mil_topk_ratio",
        "query_mil_hidden_dim",
        "query_mil_num_queries",
        "query_mil_num_heads",
        "query_mil_dropout",
        "query_mil_topk_ratio",
        "patch_disagreement_hidden_dim",
        "patch_disagreement_topk_ratio",
        "fire_error_hidden_dim",
        "fire_error_topk_ratio",
        "hos_mask_hidden_dim",
        "hos_score_hidden_dim",
        "hos_topk_ratio",
        "hos_num_phase_shifts",
        "hos_boundary_phase_eps",
        "hos_boundary_margin",
        "hos_cdc_degrade_prob",
        "hos_cdc_noise_std",
        "hos_cdc_min_scale",
        "mid_freq_radius_low_ratio",
        "mid_freq_radius_high_ratio",
    ):
        if key in checkpoint_config:
            setattr(opt, key, checkpoint_config[key])
    load_info = load_state_dict_with_prefix_compat(
        model,
        checkpoint,
        strict=bool(opt.use_svd),
    )
    if is_main:
        print(
            "Checkpoint load info | "
            f"source={load_info['source_key']} "
            f"transform={load_info['transform']} "
            f"matched={load_info['matched_keys']} "
            f"coverage={load_info['coverage']:.4f} "
            f"missing={len(load_info['missing_keys'])} "
            f"unexpected={len(load_info['unexpected_keys'])}"
        )

    if is_main:
        print("Model loaded..")
    model.eval()
    model.cuda()

    need_fire_heads = any(mode not in ("orig", "selective_or") for mode in mode_list)
    mid_band_mask_head = None
    recon_score_head = None
    evidence_head = None
    multiview_fusion_head = None
    patch_mil_head = None
    query_mil_head = None
    patch_disagreement_head = None
    fire_error_head = None
    hos_fire_envelope = None
    real_manifold_scorer = None
    if need_fire_heads:
        if "mid_band_mask_head" not in checkpoint or "recon_score_head" not in checkpoint:
            raise RuntimeError(
                "Fusion mode requires FIRE-Lite heads, but checkpoint does not contain "
                "'mid_band_mask_head' and 'recon_score_head'."
            )

        mask_in_channels, mask_hidden_dim, recon_in_channels, recon_hidden_dim = infer_fire_head_dims_from_ckpt(
            checkpoint=checkpoint,
            default_mask_hidden=opt.fire_mask_hidden_dim,
            default_recon_hidden=opt.fire_recon_hidden_dim,
        )
        if mask_in_channels is None:
            mask_in_channels = model.feature_dim

        mid_band_mask_head = MidBandMaskHead(
            in_channels=mask_in_channels,
            hidden_dim=mask_hidden_dim,
        )
        recon_score_head = ReconScoreHead(
            in_channels=recon_in_channels,
            hidden_dim=recon_hidden_dim,
        )
        mid_band_mask_head.load_state_dict(checkpoint["mid_band_mask_head"], strict=True)
        recon_score_head.load_state_dict(checkpoint["recon_score_head"], strict=True)
        mid_band_mask_head.eval().cuda()
        recon_score_head.eval().cuda()

    if opt.infer_mode in ("trained_evidence_avg", "trained_multiview_evidence_avg"):
        if "evidence_head" not in checkpoint:
            raise RuntimeError(f"infer_mode={opt.infer_mode} requires 'evidence_head' in checkpoint.")
        evidence_head = ForensicEvidenceHead(
            feature_dim=model.feature_dim,
            hidden_dim=opt.evidence_hidden_dim,
        )
        evidence_head.load_state_dict(checkpoint["evidence_head"], strict=True)
        evidence_head.eval().cuda()

    if opt.infer_mode in (
        "trained_forensic_fusion",
        "trained_forensic_hos_fusion",
        "trained_forensic_hos_logit_fusion",
    ) and "evidence_head" in checkpoint and evidence_head is None:
        evidence_head = ForensicEvidenceHead(
            feature_dim=model.feature_dim,
            hidden_dim=opt.evidence_hidden_dim,
        )
        evidence_head.load_state_dict(checkpoint["evidence_head"], strict=True)
        evidence_head.eval().cuda()

    if opt.infer_mode in ("trained_multiview_fusion", "trained_multiview_evidence_avg"):
        if "multiview_fusion_head" not in checkpoint:
            raise RuntimeError(f"infer_mode={opt.infer_mode} requires 'multiview_fusion_head' in checkpoint.")
        fusion_type = str(opt.multiview_fusion_type)
        if fusion_type == "delta":
            multiview_fusion_head = DeltaMultiViewFusionHead(
                feature_dim=model.feature_dim,
                hidden_dim=opt.multiview_fusion_hidden_dim,
            )
        else:
            multiview_fusion_head = MultiViewFusionHead(
                feature_dim=model.feature_dim,
                hidden_dim=opt.multiview_fusion_hidden_dim,
                num_views=3,
            )
        multiview_fusion_head.load_state_dict(checkpoint["multiview_fusion_head"], strict=True)
        multiview_fusion_head.eval().cuda()

    if opt.infer_mode in ("trained_delta_patch_mil_avg", "trained_forensic_fusion") and multiview_fusion_head is None:
        if "multiview_fusion_head" not in checkpoint:
            raise RuntimeError(f"infer_mode={opt.infer_mode} requires 'multiview_fusion_head' in checkpoint.")
        fusion_type = str(opt.multiview_fusion_type)
        if fusion_type == "delta":
            multiview_fusion_head = DeltaMultiViewFusionHead(
                feature_dim=model.feature_dim,
                hidden_dim=opt.multiview_fusion_hidden_dim,
            )
        else:
            multiview_fusion_head = MultiViewFusionHead(
                feature_dim=model.feature_dim,
                hidden_dim=opt.multiview_fusion_hidden_dim,
                num_views=3,
            )
        multiview_fusion_head.load_state_dict(checkpoint["multiview_fusion_head"], strict=True)
        multiview_fusion_head.eval().cuda()

    if opt.infer_mode in (
        "trained_patch_mil",
        "trained_delta_patch_mil_avg",
        "trained_forensic_fusion",
        "trained_forensic_hos_fusion",
        "trained_forensic_hos_logit_fusion",
    ):
        if "patch_mil_head" not in checkpoint:
            if opt.infer_mode not in ("trained_forensic_hos_fusion", "trained_forensic_hos_logit_fusion"):
                raise RuntimeError(f"infer_mode={opt.infer_mode} requires 'patch_mil_head' in checkpoint.")
        else:
            patch_mil_head = PatchMILHead(
                feature_dim=model.feature_dim,
                hidden_dim=opt.patch_mil_hidden_dim,
                topk_ratio=opt.patch_mil_topk_ratio,
            )
            patch_mil_head.load_state_dict(checkpoint["patch_mil_head"], strict=True)
            patch_mil_head.eval().cuda()

    if opt.infer_mode == "trained_query_mil":
        if "query_mil_head" not in checkpoint:
            raise RuntimeError(f"infer_mode={opt.infer_mode} requires 'query_mil_head' in checkpoint.")
        query_mil_head = ForensicQueryMILHead(
            feature_dim=model.feature_dim,
            hidden_dim=opt.query_mil_hidden_dim,
            num_queries=opt.query_mil_num_queries,
            num_heads=opt.query_mil_num_heads,
            dropout=opt.query_mil_dropout,
            topk_ratio=opt.query_mil_topk_ratio,
        )
        query_mil_head.load_state_dict(checkpoint["query_mil_head"], strict=True)
        query_mil_head.eval().cuda()

    if opt.infer_mode == "trained_forensic_fusion" and "patch_disagreement_head" in checkpoint:
        patch_disagreement_head = CrossViewPatchDisagreementHead(
            feature_dim=model.feature_dim,
            hidden_dim=opt.patch_disagreement_hidden_dim,
            topk_ratio=opt.patch_disagreement_topk_ratio,
        )
        patch_disagreement_head.load_state_dict(checkpoint["patch_disagreement_head"], strict=True)
        patch_disagreement_head.eval().cuda()

    if opt.infer_mode == "trained_forensic_fusion" and "fire_error_head" in checkpoint:
        fire_error_head = FireErrorEvidenceHead(
            in_channels=1,
            hidden_dim=opt.fire_error_hidden_dim,
            topk_ratio=opt.fire_error_topk_ratio,
        )
        fire_error_head.load_state_dict(checkpoint["fire_error_head"], strict=True)
        fire_error_head.eval().cuda()

    if opt.infer_mode in (
        "trained_hos_fire",
        "trained_forensic_hos_fusion",
        "trained_hos_selective_or",
        "trained_hos_manifold_logit_fusion",
        "trained_forensic_hos_logit_fusion",
    ):
        if "hos_fire_envelope" not in checkpoint:
            raise RuntimeError(f"infer_mode={opt.infer_mode} requires 'hos_fire_envelope' in checkpoint.")
        hos_fire_envelope = HOSFireEnvelope(
            feature_dim=model.feature_dim,
            mask_hidden_dim=opt.hos_mask_hidden_dim,
            score_hidden_dim=opt.hos_score_hidden_dim,
            topk_ratio=opt.hos_topk_ratio,
            mid_low_ratio=opt.mid_freq_radius_low_ratio,
            mid_high_ratio=opt.mid_freq_radius_high_ratio,
            num_phase_shifts=opt.hos_num_phase_shifts,
            boundary_phase_eps=opt.hos_boundary_phase_eps,
            boundary_margin=opt.hos_boundary_margin,
            cdc_degrade_prob=opt.hos_cdc_degrade_prob,
            cdc_noise_std=opt.hos_cdc_noise_std,
            cdc_min_scale=opt.hos_cdc_min_scale,
        )
        hos_fire_envelope.load_state_dict(checkpoint["hos_fire_envelope"], strict=True)
        hos_fire_envelope.eval().cuda()

    if opt.infer_mode in (
        "trained_real_manifold",
        "trained_hos_manifold_logit_fusion",
        "trained_forensic_hos_logit_fusion",
    ):
        if not opt.real_manifold_stats:
            raise RuntimeError(f"infer_mode={opt.infer_mode} requires --real_manifold_stats.")
        real_manifold_scorer = RealManifoldDistanceScorer(opt.real_manifold_stats)
        real_manifold_scorer.eval().cuda()

    predictor = FireLiteFusionPredictor(
        model=model,
        arch=opt.arch,
        modes=mode_list,
        fixed_alpha=opt.fusion_fixed_alpha,
        gated_delta=opt.fusion_gated_delta,
        gated_alpha_max=opt.fusion_gated_alpha_max,
        or_delta=opt.fusion_or_delta,
        or_t_cls=opt.fusion_or_t_cls,
        or_t_aux=opt.fusion_or_t_aux,
        mid_band_mask_head=mid_band_mask_head,
        recon_score_head=recon_score_head,
        evidence_head=evidence_head,
        multiview_fusion_head=multiview_fusion_head,
        patch_mil_head=patch_mil_head,
        query_mil_head=query_mil_head,
        patch_disagreement_head=patch_disagreement_head,
        fire_error_head=fire_error_head,
        hos_fire_envelope=hos_fire_envelope,
        real_manifold_scorer=real_manifold_scorer,
        infer_mode=opt.infer_mode,
        infer_random_views=opt.infer_random_views,
        infer_uncertain_delta=opt.infer_uncertain_delta,
        infer_multiview_alpha=opt.infer_multiview_alpha,
        mid_freq_radius_low_ratio=opt.mid_freq_radius_low_ratio,
        mid_freq_radius_high_ratio=opt.mid_freq_radius_high_ratio,
        hos_logit_scale=opt.hos_logit_scale,
        hos_logit_bias=opt.hos_logit_bias,
        real_manifold_logit_scale=opt.real_manifold_logit_scale,
        real_manifold_logit_bias=opt.real_manifold_logit_bias,
        logit_fusion_weight_original=opt.logit_fusion_weight_original,
        logit_fusion_weight_hos=opt.logit_fusion_weight_hos,
        logit_fusion_weight_manifold=opt.logit_fusion_weight_manifold,
        logit_fusion_weight_patch=opt.logit_fusion_weight_patch,
        logit_fusion_weight_evidence=opt.logit_fusion_weight_evidence,
        logit_fusion_bias=opt.logit_fusion_bias,
    )
    if is_main:
        print(f"Inference setup | infer_mode={opt.infer_mode} | fusion_mode={opt.fusion_mode}")

    if (opt.real_path is None) or (opt.fake_path is None) or (opt.data_mode is None):
        dataset_paths = DATASET_PATHS
    else:
        dataset_paths = [
            dict(real_path=opt.real_path, fake_path=opt.fake_path, data_mode=opt.data_mode, key="custom")
        ]

    single_orig = len(mode_list) == 1 and mode_list[0] == "orig"
    result_paths = {mode: get_result_paths(opt.result_folder, mode, single_orig=single_orig) for mode in mode_list}

    if is_main:
        for mode in mode_list:
            for key in ("ap", "acc0", "acc1"):
                with open(result_paths[mode][key], "a") as f:
                    f.write("-----------------------------------------\n")

    metrics_sum = {
        mode: {
            "ap": 0.0,
            "r_acc0": 0.0,
            "f_acc0": 0.0,
            "acc0": 0.0,
            "r_acc1": 0.0,
            "f_acc1": 0.0,
            "acc1": 0.0,
        }
        for mode in mode_list
    }

    for dataset_path in dataset_paths:
        set_seed()

        dataset = RealFakeDataset(
            dataset_path["real_path"],
            dataset_path["fake_path"],
            dataset_path["data_mode"],
            opt.max_sample,
            opt.arch,
            jpeg_quality=opt.jpeg_quality,
            gaussian_sigma=opt.gaussian_sigma,
        )
        eval_dataset = dataset
        if distributed:
            eval_dataset = torch.utils.data.Subset(
                dataset,
                list(range(rank, len(dataset), world_size)),
            )

        loader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        metrics_by_mode = validate_with_fusion(
            predictor,
            loader,
            mode_list,
            distributed_collect=distributed,
            is_main=is_main,
        )

        if not is_main:
            continue

        for mode in mode_list:
            m = metrics_by_mode[mode]
            with open(result_paths[mode]["acc0"], "a") as f:
                f.write("thres: 0.5\n")
            with open(result_paths[mode]["acc1"], "a") as f:
                f.write(f"thres: {m['best_thres']}\n")

            metrics_sum[mode]["ap"] += m["ap"]
            metrics_sum[mode]["r_acc0"] += m["r_acc0"]
            metrics_sum[mode]["f_acc0"] += m["f_acc0"]
            metrics_sum[mode]["acc0"] += m["acc0"]
            metrics_sum[mode]["r_acc1"] += m["r_acc1"]
            metrics_sum[mode]["f_acc1"] += m["f_acc1"]
            metrics_sum[mode]["acc1"] += m["acc1"]

            print(
                f"[{dataset_path['key']}] mode={mode} "
                f"AP={m['ap']*100:.2f} | "
                f"ACC@0.5(real/fake/all)={m['r_acc0']*100:.2f}/{m['f_acc0']*100:.2f}/{m['acc0']*100:.2f} | "
                f"best_thres={m['best_thres']:.6f} | "
                f"ACC@best(real/fake/all)={m['r_acc1']*100:.2f}/{m['f_acc1']*100:.2f}/{m['acc1']*100:.2f}"
            )

            with open(result_paths[mode]["ap"], "a") as f:
                f.write(dataset_path["key"] + ": " + str(round(m["ap"] * 100, 2)) + "\n")
            with open(result_paths[mode]["acc0"], "a") as f:
                f.write(
                    dataset_path["key"]
                    + ": "
                    + str(round(m["r_acc0"] * 100, 2))
                    + "  "
                    + str(round(m["f_acc0"] * 100, 2))
                    + "  "
                    + str(round(m["acc0"] * 100, 2))
                    + "\n"
                )
            with open(result_paths[mode]["acc1"], "a") as f:
                f.write(
                    dataset_path["key"]
                    + ": "
                    + str(round(m["r_acc1"] * 100, 2))
                    + "  "
                    + str(round(m["f_acc1"] * 100, 2))
                    + "  "
                    + str(round(m["acc1"] * 100, 2))
                    + "\n"
                )

    if is_main:
        print("========== Validation Summary ==========")
        num_sets = len(dataset_paths)
        for mode in mode_list:
            avg = {k: v / num_sets for k, v in metrics_sum[mode].items()}

            with open(result_paths[mode]["ap"], "a") as f:
                f.write("avg: " + str(round(avg["ap"] * 100, 2)) + "\n")
                f.write("-----------------------------------------\n")
            with open(result_paths[mode]["acc0"], "a") as f:
                f.write(
                    "avg: "
                    + str(round(avg["r_acc0"] * 100, 2))
                    + "  "
                    + str(round(avg["f_acc0"] * 100, 2))
                    + "  "
                    + str(round(avg["acc0"] * 100, 2))
                    + "\n"
                )
                f.write("-----------------------------------------\n")
            with open(result_paths[mode]["acc1"], "a") as f:
                f.write(
                    "avg: "
                    + str(round(avg["r_acc1"] * 100, 2))
                    + "  "
                    + str(round(avg["f_acc1"] * 100, 2))
                    + "  "
                    + str(round(avg["acc1"] * 100, 2))
                    + "\n"
                )
                f.write("-----------------------------------------\n")

            print(
                f"mode={mode} | "
                f"AP avg: {avg['ap']*100:.2f} | "
                f"ACC@0.5 avg (real/fake/all): {avg['r_acc0']*100:.2f}/{avg['f_acc0']*100:.2f}/{avg['acc0']*100:.2f} | "
                f"ACC@best avg (real/fake/all): {avg['r_acc1']*100:.2f}/{avg['f_acc1']*100:.2f}/{avg['acc1']*100:.2f}"
            )

    cleanup_distributed()
