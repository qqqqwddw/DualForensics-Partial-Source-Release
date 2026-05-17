import torch
import math
from contextlib import nullcontext
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from .base_model import BaseModel
from models import get_model
from models.modules import (
    FixedPixelMapping,
    CrossViewPatchDisagreementHead,
    DeltaMultiViewFusionHead,
    FireErrorEvidenceHead,
    ForensicQueryMILHead,
    HOSFireEnvelope,
    RandomPixelMapping,
    MLPProjector,
    MidFrequencyPrior,
    MidPriorPredictor,
    MidBandMaskHead,
    ReconScoreHead,
    ForensicEvidenceHead,
    build_mid_target_mask,
    fft_band_filter_rgb,
    feature_l2_distance,
    info_nce_loss,
    js_divergence,
    mid_prior_loss,
    MultiViewFusionHead,
    MultiScaleNoiseGuidance,
    PatchMILHead,
    parse_stage_scales,
)


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


class Trainer(BaseModel):
    def name(self):
        return "Trainer"

    def __init__(self, opt):
        super(Trainer, self).__init__(opt)
        self.opt = opt
        self.multi_res_sizes = self._parse_multi_res_sizes(getattr(opt, "multi_res_sizes", ""))
        self.use_multi_resolution_training = bool(
            getattr(opt, "use_multi_resolution_training", False) and len(self.multi_res_sizes) > 0
        )
        self.dual_res_high_size = int(getattr(opt, "dual_res_high_size", 384))
        self.use_dual_resolution_consistency = bool(
            getattr(opt, "use_dual_resolution_consistency", False) and self.dual_res_high_size > 0
        )
        self.amp_dtype_str = str(getattr(opt, "amp_dtype", "fp16")).lower()
        if self.amp_dtype_str not in ("fp16", "bf16"):
            raise ValueError("amp_dtype should be [fp16, bf16]")
        self.use_amp = bool(
            getattr(opt, "use_amp", False)
            and torch.cuda.is_available()
            and self.device.type == "cuda"
        )
        self.amp_dtype = torch.float16 if self.amp_dtype_str == "fp16" else torch.bfloat16
        amp_scaler_enabled = self.use_amp and self.amp_dtype == torch.float16
        try:
            self.amp_scaler = torch.amp.GradScaler(
                "cuda",
                enabled=amp_scaler_enabled,
            )
        except (AttributeError, TypeError):
            self.amp_scaler = torch.cuda.amp.GradScaler(enabled=amp_scaler_enabled)
        self.model = get_model(opt.arch, opt)
        self.base_lr = float(opt.lr)
        self.lr = self.base_lr
        self.lr_warmup_origin_step = 0
        torch.nn.init.normal_(self.model.fc.weight.data, 0.0, opt.init_gain)

        stat_from = self._get_stat_source(opt.arch)
        self.register_buffer("norm_mean", torch.tensor(MEAN[stat_from], dtype=torch.float32).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("norm_std", torch.tensor(STD[stat_from], dtype=torch.float32).view(1, 3, 1, 1), persistent=False)

        self.fixed_mapping = FixedPixelMapping(remap_to_unit=opt.pixel_mapping_remap_to_unit)
        self.random_mapping = RandomPixelMapping(
            remap_to_unit=opt.pixel_mapping_remap_to_unit,
            random_range=opt.pixel_mapping_random_range,
            deterministic=opt.pixel_mapping_deterministic,
            seed=opt.pixel_mapping_seed,
        )
        self.projector = MLPProjector(in_dim=self.model.feature_dim, out_dim=opt.proj_dim)

        if opt.use_mid_frequency_prior:
            self.mid_frequency_prior = MidFrequencyPrior(
                use_grayscale=opt.mid_prior_use_grayscale,
                low_ratio=opt.mid_freq_radius_low_ratio,
                high_ratio=opt.mid_freq_radius_high_ratio,
                use_absolute_response=opt.mid_freq_use_absolute_response,
            )
            self.mid_prior_predictor = MidPriorPredictor(
                in_channels=self.model.feature_dim,
                hidden_dim=opt.mid_prior_predictor_hidden_dim,
                out_channels=opt.mid_prior_out_channels,
                depth=opt.mid_prior_predictor_depth,
            )
        else:
            self.mid_frequency_prior = None
            self.mid_prior_predictor = None

        if getattr(opt, "use_evidence_head", False):
            self.evidence_head = ForensicEvidenceHead(
                feature_dim=self.model.feature_dim,
                hidden_dim=opt.evidence_hidden_dim,
            )
        else:
            self.evidence_head = None

        if getattr(opt, "use_multiview_fusion_head", False):
            fusion_type = getattr(opt, "multiview_fusion_type", "concat")
            if fusion_type == "delta":
                self.multiview_fusion_head = DeltaMultiViewFusionHead(
                    feature_dim=self.model.feature_dim,
                    hidden_dim=opt.multiview_fusion_hidden_dim,
                )
            else:
                self.multiview_fusion_head = MultiViewFusionHead(
                    feature_dim=self.model.feature_dim,
                    hidden_dim=opt.multiview_fusion_hidden_dim,
                    num_views=3,
                )
        else:
            self.multiview_fusion_head = None

        if getattr(opt, "use_patch_mil_head", False):
            self.patch_mil_head = PatchMILHead(
                feature_dim=self.model.feature_dim,
                hidden_dim=opt.patch_mil_hidden_dim,
                topk_ratio=opt.patch_mil_topk_ratio,
            )
        else:
            self.patch_mil_head = None

        if getattr(opt, "use_query_mil_head", False):
            self.query_mil_head = ForensicQueryMILHead(
                feature_dim=self.model.feature_dim,
                hidden_dim=opt.query_mil_hidden_dim,
                num_queries=opt.query_mil_num_queries,
                num_heads=opt.query_mil_num_heads,
                dropout=opt.query_mil_dropout,
                topk_ratio=opt.query_mil_topk_ratio,
            )
        else:
            self.query_mil_head = None

        if getattr(opt, "use_cross_view_patch_disagreement", False):
            self.patch_disagreement_head = CrossViewPatchDisagreementHead(
                feature_dim=self.model.feature_dim,
                hidden_dim=opt.patch_disagreement_hidden_dim,
                topk_ratio=opt.patch_disagreement_topk_ratio,
            )
        else:
            self.patch_disagreement_head = None

        if getattr(opt, "use_fire_error_evidence", False):
            self.fire_error_head = FireErrorEvidenceHead(
                in_channels=1,
                hidden_dim=opt.fire_error_hidden_dim,
                topk_ratio=opt.fire_error_topk_ratio,
            )
        else:
            self.fire_error_head = None

        if getattr(opt, "use_hos_fire_envelope", False):
            self.hos_fire_envelope = HOSFireEnvelope(
                feature_dim=self.model.feature_dim,
                mask_hidden_dim=opt.hos_mask_hidden_dim,
                score_hidden_dim=opt.hos_score_hidden_dim,
                topk_ratio=opt.hos_topk_ratio,
                mid_low_ratio=opt.mid_freq_radius_low_ratio,
                mid_high_ratio=opt.mid_freq_radius_high_ratio,
                num_phase_shifts=opt.hos_num_phase_shifts,
                boundary_phase_eps=opt.hos_boundary_phase_eps,
                boundary_margin=opt.hos_boundary_margin,
                rank_margin=opt.hos_rank_margin,
                evidence_fake_weight=opt.hos_evidence_fake_weight,
                boundary_cls_weight=opt.hos_boundary_cls_weight,
                tangent_rank=opt.hos_tangent_rank,
                detach_aux_maps=opt.hos_detach_aux,
                cdc_degrade_prob=opt.hos_cdc_degrade_prob,
                cdc_noise_std=opt.hos_cdc_noise_std,
                cdc_min_scale=opt.hos_cdc_min_scale,
            )
        else:
            self.hos_fire_envelope = None

        if opt.use_fire_lite:
            self.mid_band_mask_head = MidBandMaskHead(
                in_channels=self.model.feature_dim,
                hidden_dim=opt.fire_mask_hidden_dim,
            )
            self.recon_score_head = ReconScoreHead(
                in_channels=1,
                hidden_dim=opt.fire_recon_hidden_dim,
            )
        else:
            self.mid_band_mask_head = None
            self.recon_score_head = None

        if opt.use_noise_guidance:
            self.noise_guidance = MultiScaleNoiseGuidance(
                in_channels=self.model.feature_dim,
                embed_dim=opt.noise_guidance_embed_dim,
                stage_scales=parse_stage_scales(opt.noise_guidance_scales),
                topk_ratio=opt.noise_guidance_topk_ratio,
                detach_mask=opt.noise_guidance_detach_mask,
                use_grayscale=opt.noise_guidance_use_grayscale,
                max_tokens=opt.noise_guidance_max_tokens,
            )
            self.noise_guidance_head = nn.Sequential(
                nn.LayerNorm(self.model.feature_dim * 2),
                nn.Linear(self.model.feature_dim * 2, opt.noise_guidance_head_hidden_dim),
                nn.GELU(),
                nn.Dropout(float(getattr(opt, "noise_guidance_head_dropout", 0.10))),
                nn.Linear(opt.noise_guidance_head_hidden_dim, 1),
            )
        else:
            self.noise_guidance = None
            self.noise_guidance_head = None

        params = []
        train_hos_only = bool(getattr(opt, "train_hos_only", False))
        if train_hos_only and self.hos_fire_envelope is None:
            raise ValueError("--train_hos_only requires --use_hos_fire_envelope")

        if train_hos_only:
            for module in (
                self.model,
                self.projector,
                self.mid_prior_predictor,
                self.evidence_head,
                self.multiview_fusion_head,
                self.patch_mil_head,
                self.query_mil_head,
                self.patch_disagreement_head,
                self.fire_error_head,
                self.mid_band_mask_head,
                self.recon_score_head,
                self.noise_guidance,
                self.noise_guidance_head,
            ):
                if module is None:
                    continue
                for p in module.parameters():
                    p.requires_grad = False
            params.extend(self.hos_fire_envelope.parameters())
            print("HOS-only training enabled. Only HOSFireEnvelope parameters are trainable.")
        elif opt.fix_backbone:
            for name, p in self.model.named_parameters():
                if name == "fc.weight" or name == "fc.bias":
                    params.append(p)
                else:
                    p.requires_grad = False
        else:
            print(
                "Your backbone is not fixed. Are you sure you want to proceed? "
                "If this is a mistake, enable the --fix_backbone command during training and rerun"
            )
            params.extend([p for p in self.model.parameters() if p.requires_grad])

        if not train_hos_only:
            params.extend(self.projector.parameters())
            if self.mid_prior_predictor is not None:
                params.extend(self.mid_prior_predictor.parameters())
            if self.evidence_head is not None:
                params.extend(self.evidence_head.parameters())
            if self.multiview_fusion_head is not None:
                params.extend(self.multiview_fusion_head.parameters())
            if self.patch_mil_head is not None:
                params.extend(self.patch_mil_head.parameters())
            if self.query_mil_head is not None:
                params.extend(self.query_mil_head.parameters())
            if self.patch_disagreement_head is not None:
                params.extend(self.patch_disagreement_head.parameters())
            if self.fire_error_head is not None:
                params.extend(self.fire_error_head.parameters())
            if self.hos_fire_envelope is not None:
                params.extend(self.hos_fire_envelope.parameters())
            if self.mid_band_mask_head is not None:
                params.extend(self.mid_band_mask_head.parameters())
            if self.recon_score_head is not None:
                params.extend(self.recon_score_head.parameters())
            if self.noise_guidance is not None:
                params.extend(self.noise_guidance.parameters())
            if self.noise_guidance_head is not None:
                params.extend(self.noise_guidance_head.parameters())

        if opt.optim in ("adam", "adamw"):
            self.optimizer = torch.optim.AdamW(
                params,
                lr=opt.lr,
                betas=(opt.beta1, 0.999),
                weight_decay=opt.weight_decay,
            )
        elif opt.optim == "sgd":
            self.optimizer = torch.optim.SGD(params, lr=opt.lr, momentum=0.0, weight_decay=opt.weight_decay)
        else:
            raise ValueError("optim should be [adam, adamw, sgd]")

        self.loss_fn = nn.BCEWithLogitsLoss()
        self.loss_dict = {}
        if self.use_multi_resolution_training:
            print(f"Multi-resolution training enabled. sizes={self.multi_res_sizes}")
        if self.use_dual_resolution_consistency:
            print(f"Dual-resolution consistency enabled. high_size={self.dual_res_high_size}")
        if self.use_amp:
            scaler_state = "on" if self.amp_scaler.is_enabled() else "off"
            print(f"AMP enabled. dtype={self.amp_dtype_str}, scaler={scaler_state}")

        self.model.to(self.device)
        if getattr(opt, "distributed", False):
            print(f"Using DDP on rank {opt.rank}, local_rank {opt.local_rank}, gpu {opt.gpu_ids[0]}")
            self.model = DDP(
                self.model,
                device_ids=[opt.gpu_ids[0]],
                output_device=opt.gpu_ids[0],
                broadcast_buffers=False,
                find_unused_parameters=False,
            )
        elif len(opt.gpu_ids) > 1:
            print(f"Using DataParallel on GPUs: {opt.gpu_ids}")
            self.model = nn.DataParallel(self.model, device_ids=opt.gpu_ids, output_device=opt.gpu_ids[0])
        self.projector.to(self.device)
        self.fixed_mapping.to(self.device)
        self.random_mapping.to(self.device)
        if self.mid_frequency_prior is not None:
            self.mid_frequency_prior.to(self.device)
        if self.mid_prior_predictor is not None:
            self.mid_prior_predictor.to(self.device)
        if self.evidence_head is not None:
            self.evidence_head.to(self.device)
        if self.multiview_fusion_head is not None:
            self.multiview_fusion_head.to(self.device)
        if self.patch_mil_head is not None:
            self.patch_mil_head.to(self.device)
        if self.query_mil_head is not None:
            self.query_mil_head.to(self.device)
        if self.patch_disagreement_head is not None:
            self.patch_disagreement_head.to(self.device)
        if self.fire_error_head is not None:
            self.fire_error_head.to(self.device)
        if self.hos_fire_envelope is not None:
            self.hos_fire_envelope.to(self.device)
        if self.mid_band_mask_head is not None:
            self.mid_band_mask_head.to(self.device)
        if self.recon_score_head is not None:
            self.recon_score_head.to(self.device)
        if self.noise_guidance is not None:
            self.noise_guidance.to(self.device)
        if self.noise_guidance_head is not None:
            self.noise_guidance_head.to(self.device)
        self._broadcast_auxiliary_modules()

    def _get_stat_source(self, arch):
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

    def _parse_multi_res_sizes(self, raw):
        if raw is None:
            return []
        if isinstance(raw, str):
            tokens = [tok.strip() for tok in raw.split(",")]
        else:
            tokens = [str(v).strip() for v in raw]

        sizes = []
        for tok in tokens:
            if not tok:
                continue
            value = int(tok)
            if value > 0:
                sizes.append(value)
        return sorted(set(sizes))

    def _sample_train_resolution(self):
        if not (self.training and self.use_multi_resolution_training):
            return int(self.input.shape[-1])
        if len(self.multi_res_sizes) == 1:
            return int(self.multi_res_sizes[0])

        if getattr(self.opt, "distributed", False) and dist.is_available() and dist.is_initialized():
            if self.opt.rank == 0:
                idx = torch.randint(
                    low=0,
                    high=len(self.multi_res_sizes),
                    size=(1,),
                    device=self.device,
                    dtype=torch.long,
                )
            else:
                idx = torch.zeros(1, device=self.device, dtype=torch.long)
            dist.broadcast(idx, src=0)
            return int(self.multi_res_sizes[int(idx.item())])

        idx = int(torch.randint(low=0, high=len(self.multi_res_sizes), size=(1,)).item())
        return int(self.multi_res_sizes[idx])

    def _autocast_ctx(self):
        if not self.use_amp:
            return nullcontext()
        try:
            return torch.amp.autocast(device_type="cuda", dtype=self.amp_dtype)
        except AttributeError:
            return torch.cuda.amp.autocast(dtype=self.amp_dtype)

    def _backbone_for_forward(self):
        if (not self.training) and hasattr(self.model, "module"):
            return self.model.module
        return self.model

    def adjust_learning_rate(self, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] *= 0.8
            self.lr = param_group["lr"]
            if param_group["lr"] < min_lr:
                return False
        return True

    def reset_lr_warmup(self):
        self.lr_warmup_origin_step = int(self.total_steps)

    def update_learning_rate(self):
        warmup_steps = int(getattr(self.opt, "lr_warmup_steps", 0))
        if warmup_steps <= 0:
            target_lr = self.base_lr
        else:
            start_factor = float(getattr(self.opt, "lr_warmup_start_factor", 0.1))
            start_factor = max(0.0, min(1.0, start_factor))
            schedule_step = int(self.total_steps)
            if getattr(self.opt, "lr_warmup_reset_on_resume", False):
                schedule_step = max(0, int(self.total_steps) - int(self.lr_warmup_origin_step))
            progress = min(1.0, float(schedule_step) / float(warmup_steps))
            lr_factor = start_factor + (1.0 - start_factor) * progress
            target_lr = self.base_lr * lr_factor

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = target_lr
        self.lr = target_lr
        return target_lr

    def set_input(self, input_data):
        self.input = input_data[0].to(self.device)
        self.label = input_data[1].to(self.device).float()

    def _to_unit(self, x):
        norm_mean = self.norm_mean.to(device=x.device, dtype=x.dtype)
        norm_std = self.norm_std.to(device=x.device, dtype=x.dtype)
        return (x * norm_std + norm_mean).clamp(0.0, 1.0)

    def _from_unit(self, x):
        norm_mean = self.norm_mean.to(device=x.device, dtype=x.dtype)
        norm_std = self.norm_std.to(device=x.device, dtype=x.dtype)
        return (x - norm_mean) / (norm_std + 1e-8)

    def _warmup_random_weight(self, epoch):
        warmup_epochs = self.opt.random_branch_warmup_epochs
        if warmup_epochs <= 0:
            return 1.0
        epoch_1based = 1 if epoch is None else (epoch + 1)
        return min(1.0, float(epoch_1based) / float(warmup_epochs))

    def _forward_branch(self, x, return_aux_map=False):
        feat, logit, aux_map = self._backbone_for_forward()(
            x,
            return_feature=True,
            return_aux_map=return_aux_map,
        )
        return feat, logit.view(-1, 1), aux_map

    def _forward_branch_with_tokens(self, x, return_aux_map=False):
        feat, logit, aux_map, patch_tokens = self._backbone_for_forward()(
            x,
            return_feature=True,
            return_aux_map=return_aux_map,
            return_patch_tokens=True,
        )
        return feat, logit.view(-1, 1), aux_map, patch_tokens

    def _classification_loss(self, logits, label):
        logits = logits.squeeze(1) if logits.ndim == 2 and logits.shape[1] == 1 else logits
        bce = F.binary_cross_entropy_with_logits(logits, label, reduction="none")
        fake_weight = float(getattr(self.opt, "fake_loss_weight", 1.0))
        if fake_weight != 1.0:
            weights = torch.where(label >= 0.5, torch.full_like(label, fake_weight), torch.ones_like(label))
            bce = bce * weights
        if getattr(self.opt, "use_focal_loss", False):
            prob = torch.sigmoid(logits)
            pt = torch.where(label >= 0.5, prob, 1.0 - prob)
            gamma = float(getattr(self.opt, "focal_gamma", 2.0))
            bce = ((1.0 - pt).clamp(min=1e-6) ** gamma) * bce
        return bce.mean()

    @staticmethod
    def _zero_like_label(label):
        return label.sum() * 0.0

    def _multiview_logit_disagreement(self, logits):
        if len(logits) < 2:
            return self._zero_like_label(self.label)
        cols = [logit.squeeze(1) if logit.ndim == 2 and logit.shape[1] == 1 else logit for logit in logits]
        pairs = []
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                pairs.append(torch.abs(cols[i] - cols[j]))
        return torch.stack(pairs, dim=0).mean(dim=0)

    def _real_consistency_loss(self, logits, label):
        weight = float(getattr(self.opt, "lambda_real_consistency", 0.0))
        if weight <= 0:
            return self._zero_like_label(label)
        real_mask = label < 0.5
        if not real_mask.any():
            return self._zero_like_label(label)
        disagreement = self._multiview_logit_disagreement(logits)
        return disagreement[real_mask].mean()

    def _fake_disagreement_loss(self, logits, label):
        weight = float(getattr(self.opt, "lambda_fake_disagreement", 0.0))
        if weight <= 0:
            return self._zero_like_label(label)
        fake_mask = label >= 0.5
        if not fake_mask.any():
            return self._zero_like_label(label)
        disagreement = self._multiview_logit_disagreement(logits)
        margin = float(getattr(self.opt, "fake_disagreement_margin", 0.15))
        return F.relu(margin - disagreement[fake_mask]).mean()

    def _fake_margin_loss(self, logits, label):
        margin_weight = float(getattr(self.opt, "lambda_fake_margin", 0.0))
        if margin_weight <= 0:
            return logits.sum() * 0.0
        logits = logits.squeeze(1) if logits.ndim == 2 and logits.shape[1] == 1 else logits
        fake_mask = label >= 0.5
        if not fake_mask.any():
            return logits.sum() * 0.0
        margin = float(getattr(self.opt, "fake_margin", 1.0))
        return F.relu(margin - logits[fake_mask]).mean()

    @staticmethod
    def _unit_eps(value):
        value = float(value)
        return value / 255.0 if value > 1.0 else value

    @staticmethod
    def _range_pair(raw, default_low, default_high):
        if raw is None:
            return float(default_low), float(default_high)
        if isinstance(raw, str):
            tokens = [tok.strip() for tok in raw.split(",") if tok.strip()]
        else:
            tokens = [str(v).strip() for v in raw]
        if not tokens:
            return float(default_low), float(default_high)
        if len(tokens) == 1:
            value = float(tokens[0])
            return value, value
        return float(tokens[0]), float(tokens[1])

    def _epoch_ready(self, epoch, start_epoch):
        start_epoch = int(start_epoch)
        if start_epoch <= 0:
            return True
        epoch_1based = 1 if epoch is None else int(epoch) + 1
        return epoch_1based >= start_epoch

    def _sample_fake_indices(self, label, prob, max_items):
        fake_mask = label >= 0.5
        idx = torch.nonzero(fake_mask, as_tuple=False).flatten()
        if idx.numel() == 0:
            return idx
        prob = float(prob)
        if prob <= 0.0:
            return idx[:0]
        if prob < 1.0:
            keep = torch.rand(idx.numel(), device=idx.device) < prob
            idx = idx[keep]
        if idx.numel() == 0:
            return idx
        max_items = int(max_items)
        if max_items > 0 and idx.numel() > max_items:
            perm = torch.randperm(idx.numel(), device=idx.device)[:max_items]
            idx = idx[perm]
        return idx

    @staticmethod
    def _gaussian_kernel2d(sigma, device, dtype):
        sigma = float(max(float(sigma), 1e-3))
        radius = max(1, int(math.ceil(3.0 * sigma)))
        coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
        kernel1d = torch.exp(-(coords ** 2) / (2.0 * sigma * sigma))
        kernel1d = kernel1d / kernel1d.sum().clamp_min(1e-6)
        kernel2d = torch.outer(kernel1d, kernel1d)
        kernel2d = kernel2d / kernel2d.sum().clamp_min(1e-6)
        return kernel2d, radius

    def _gaussian_blur_torch(self, x, sigma):
        sigma = float(sigma)
        if sigma <= 0.0:
            return x
        kernel2d, radius = self._gaussian_kernel2d(sigma, x.device, x.dtype)
        c = x.shape[1]
        kernel = kernel2d.view(1, 1, kernel2d.shape[0], kernel2d.shape[1]).repeat(c, 1, 1, 1)
        x_pad = F.pad(x, (radius, radius, radius, radius), mode="reflect")
        return F.conv2d(x_pad, kernel, groups=c)

    def _resize_roundtrip_torch(self, x, scale):
        scale = float(max(0.1, min(1.0, scale)))
        h, w = x.shape[-2:]
        nh = max(8, int(round(h * scale)))
        nw = max(8, int(round(w * scale)))
        x = F.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False)
        return F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)

    @staticmethod
    def _jpeg_ste_torch(x, levels):
        levels = float(max(2.0, levels))
        x_quant = torch.round(x.clamp(0.0, 1.0) * levels) / levels
        return x + (x_quant - x).detach()

    @staticmethod
    def _gray_from_rgb(x):
        return 0.2989 * x[:, 0:1] + 0.5870 * x[:, 1:2] + 0.1140 * x[:, 2:3]

    def _color_adjust_torch(self, x, brightness, contrast, saturation, gamma, bias):
        gray = self._gray_from_rgb(x)
        x = gray + (x - gray) * float(saturation)
        mean = x.mean(dim=(2, 3), keepdim=True)
        x = (x - mean) * float(contrast) + mean
        x = x * float(brightness) + float(bias)
        x = x.clamp(0.0, 1.0).pow(float(gamma))
        return x.clamp(0.0, 1.0)

    def _mid_suppress_torch(self, x, low_ratio, high_ratio, strength):
        x = x.float().clamp(0.0, 1.0)
        b, c, h, w = x.shape
        mask = build_mid_target_mask(
            batch_size=b,
            height=h,
            width=w,
            low_ratio=float(low_ratio),
            high_ratio=float(high_ratio),
            device=x.device,
            dtype=torch.float32,
        )
        attenuation = 1.0 - float(strength) * mask
        freq = torch.fft.fftshift(torch.fft.fft2(x * 255.0, dim=(-2, -1)), dim=(-2, -1))
        freq = freq * attenuation
        freq = torch.fft.ifftshift(freq, dim=(-2, -1))
        x_rec = torch.fft.ifft2(freq, dim=(-2, -1)).real
        flat = x_rec.flatten(2)
        x_min = flat.min(dim=2, keepdim=True).values.view(b, c, 1, 1)
        x_max = flat.max(dim=2, keepdim=True).values.view(b, c, 1, 1)
        x_rec = (x_rec - x_min) / (x_max - x_min + 1e-6)
        return x_rec.clamp(0.0, 1.0).to(dtype=x.dtype)

    def _apply_social_chain_torch(self, x_unit, hard=False):
        x = x_unit.float().clamp(0.0, 1.0)
        hard = bool(hard)
        if hard:
            resize_low, resize_high = 0.35, 0.90
            jpeg_low, jpeg_high = 12.0, 48.0
            blur_low, blur_high = 0.60, 2.00
            bright_low, bright_high = 0.82, 1.18
            contrast_low, contrast_high = 0.80, 1.20
            sat_low, sat_high = 0.80, 1.20
            gamma_low, gamma_high = 0.82, 1.25
            bias_low, bias_high = -0.06, 0.06
            noise_low, noise_high = 0.0, 0.020
        else:
            resize_low, resize_high = 0.60, 0.95
            jpeg_low, jpeg_high = 24.0, 64.0
            blur_low, blur_high = 0.0, 1.0
            bright_low, bright_high = 0.90, 1.10
            contrast_low, contrast_high = 0.90, 1.12
            sat_low, sat_high = 0.90, 1.12
            gamma_low, gamma_high = 0.90, 1.15
            bias_low, bias_high = -0.04, 0.04
            noise_low, noise_high = 0.0, 0.010

        ops = ["resize", "jpeg", "blur", "color", "noise"]
        min_ops, max_ops = (3, 5) if hard else (2, 4)
        num_ops = int(torch.randint(low=min_ops, high=max_ops + 1, size=(1,), device=x.device).item())
        num_ops = max(1, min(num_ops, len(ops)))
        chosen = torch.randperm(len(ops), device=x.device)[:num_ops].tolist()

        for idx in chosen:
            op = ops[idx]
            if op == "resize":
                scale = torch.empty(1, device=x.device).uniform_(resize_low, resize_high).item()
                x = self._resize_roundtrip_torch(x, scale)
            elif op == "jpeg":
                levels = torch.empty(1, device=x.device).uniform_(jpeg_low, jpeg_high).item()
                x = self._jpeg_ste_torch(x, levels)
            elif op == "blur":
                sigma = torch.empty(1, device=x.device).uniform_(blur_low, blur_high).item()
                if sigma > 0:
                    x = self._gaussian_blur_torch(x, sigma)
            elif op == "color":
                brightness = torch.empty(1, device=x.device).uniform_(bright_low, bright_high).item()
                contrast = torch.empty(1, device=x.device).uniform_(contrast_low, contrast_high).item()
                saturation = torch.empty(1, device=x.device).uniform_(sat_low, sat_high).item()
                gamma = torch.empty(1, device=x.device).uniform_(gamma_low, gamma_high).item()
                bias = torch.empty(1, device=x.device).uniform_(bias_low, bias_high).item()
                x = self._color_adjust_torch(x, brightness, contrast, saturation, gamma, bias)
            else:
                noise_std = torch.empty(1, device=x.device).uniform_(noise_low, noise_high).item()
                if noise_std > 0:
                    x = (x + torch.randn_like(x) * noise_std).clamp(0.0, 1.0)

        if hard:
            x = self._mid_suppress_torch(x, 0.15, 0.45, torch.empty(1, device=x.device).uniform_(0.25, 0.70).item())
        return x.clamp(0.0, 1.0).to(dtype=x_unit.dtype)

    def _fake_attack_loss(self, x_unit_view, attack_use_main=True, attack_use_hos=True):
        attack_terms = []
        feat_view, logit_view, aux_view = self._forward_branch(
            self._from_unit(x_unit_view),
            return_aux_map=bool(attack_use_hos and self.hos_fire_envelope is not None),
        )
        zero_target = torch.zeros(logit_view.shape[0], device=logit_view.device, dtype=logit_view.dtype)
        if attack_use_main:
            attack_terms.append(self._classification_loss(logit_view, zero_target))
        if attack_use_hos and self.hos_fire_envelope is not None and aux_view is not None:
            hos_logit = self._hos_fire_logit_from_aux(x_unit_view, aux_view)
            attack_terms.append(self._classification_loss(hos_logit, zero_target))
        if not attack_terms:
            return None
        return torch.stack(attack_terms, dim=0).mean()

    def _fake_hard_view_losses(self, x_unit_view, feat_clean, logit_clean, use_hos=True):
        feat_view, logit_view, aux_view = self._forward_branch(
            self._from_unit(x_unit_view),
            return_aux_map=bool(use_hos and self.hos_fire_envelope is not None),
        )
        fake_target = torch.ones(logit_view.shape[0], device=logit_view.device, dtype=logit_view.dtype)
        loss_cls = self._classification_loss(logit_view, fake_target)
        hos_logit = None
        if use_hos and self.hos_fire_envelope is not None and aux_view is not None:
            hos_logit = self._hos_fire_logit_from_aux(x_unit_view, aux_view)
            loss_cls = loss_cls + float(getattr(self.opt, "fake_hard_hos_weight", 1.0)) * self._classification_loss(
                hos_logit,
                fake_target,
            )

        clean_logit = self._binary_logit(logit_clean).detach()
        view_logit = self._binary_logit(logit_view)
        loss_consistency = js_divergence(clean_logit, view_logit)
        loss_feat_consistency = 1.0 - F.cosine_similarity(
            F.normalize(feat_clean.detach(), dim=1),
            F.normalize(feat_view, dim=1),
            dim=1,
        ).mean()

        margin = float(getattr(self.opt, "fake_hard_margin", 1.0))
        margin_terms = [F.relu(margin - view_logit.view(-1)).mean()]
        if hos_logit is not None:
            margin_terms.append(F.relu(margin - self._binary_logit(hos_logit).view(-1)).mean())
        loss_margin = torch.stack(margin_terms, dim=0).mean()
        return feat_view, logit_view, hos_logit, loss_cls, loss_consistency, loss_feat_consistency, loss_margin

    def _fake_adv_view(self, x_unit_base, attack_use_main=True, attack_use_hos=True):
        eps = self._unit_eps(getattr(self.opt, "fake_adv_eps", 4.0))
        steps = max(1, int(getattr(self.opt, "fake_adv_steps", 2)))
        alpha_raw = float(getattr(self.opt, "fake_adv_alpha", 0.0))
        alpha = self._unit_eps(alpha_raw) if alpha_raw > 0.0 else eps / float(steps)
        views = max(1, int(getattr(self.opt, "fake_adv_eot_views", 1)))

        x_base = x_unit_base.detach().float()
        x_adv = x_base.clone()
        if bool(getattr(self.opt, "fake_adv_random_start", True)):
            x_adv = (x_adv + torch.empty_like(x_adv).uniform_(-eps, eps)).clamp(0.0, 1.0)

        for _ in range(steps):
            x_adv = x_adv.detach().requires_grad_(True)
            loss = None
            for _view in range(views):
                x_view = x_adv
                if bool(getattr(self.opt, "fake_adv_on_social_chain", True)):
                    x_view = self._apply_social_chain_torch(x_view, hard=True)
                attack_loss = self._fake_attack_loss(
                    x_view,
                    attack_use_main=attack_use_main,
                    attack_use_hos=attack_use_hos,
                )
                if attack_loss is None:
                    continue
                loss = attack_loss if loss is None else loss + attack_loss
            if loss is None:
                return x_base.to(dtype=x_unit_base.dtype)
            loss = loss / float(views)
            grad = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]
            x_adv = x_adv - alpha * grad.sign()
            delta = torch.clamp(x_adv - x_base, min=-eps, max=eps)
            x_adv = torch.clamp(x_base + delta, 0.0, 1.0).detach()
        return x_adv.to(dtype=x_unit_base.dtype)

    def _branch_prob(self, x):
        _, logit, _ = self._forward_branch(x, return_aux_map=False)
        return torch.sigmoid(logit)

    @staticmethod
    def _binary_logit(logit):
        if logit.ndim == 1:
            return logit.view(-1, 1)
        if logit.ndim == 2 and logit.shape[1] == 1:
            return logit
        if logit.ndim == 2 and logit.shape[1] == 2:
            return (logit[:, 1] - logit[:, 0]).view(-1, 1)
        return logit.view(logit.shape[0], -1)[:, :1]

    def _weighted_logit_fusion(self, parts, weights):
        if len(parts) == 0:
            raise RuntimeError("No logits were provided for logit fusion.")
        fused = torch.zeros_like(parts[0])
        norm = 0.0
        for logit, weight in zip(parts, weights):
            weight = float(weight)
            if abs(weight) < 1e-12:
                continue
            fused = fused + weight * self._binary_logit(logit)
            norm += abs(weight)
        if norm <= 1e-12:
            raise RuntimeError("All logit fusion weights are zero.")
        return fused / norm + float(getattr(self.opt, "logit_fusion_bias", 0.0))

    def _fire_error_logit_from_aux(self, x_unit, aux_map_o):
        target_mask = build_mid_target_mask(
            batch_size=x_unit.shape[0],
            height=x_unit.shape[-2],
            width=x_unit.shape[-1],
            low_ratio=self.opt.mid_freq_radius_low_ratio,
            high_ratio=self.opt.mid_freq_radius_high_ratio,
            device=x_unit.device,
            dtype=x_unit.dtype,
        )
        x_mid_unit = fft_band_filter_rgb(x_unit, target_mask)
        x_mid = self._from_unit(x_mid_unit)
        _, _, aux_map_mid = self._forward_branch(x_mid, return_aux_map=True)
        if aux_map_mid is None:
            raise RuntimeError("Backbone did not return aux_map for FIRE error evidence.")
        delta_map = torch.mean(torch.abs(aux_map_o - aux_map_mid), dim=1, keepdim=True)
        return self.fire_error_head(delta_map)

    def _forward_unit_for_aux(self, x_unit):
        x = self._from_unit(x_unit)
        logit, aux_map = None, None
        _, logit, aux_map = self._forward_branch(x, return_aux_map=True)
        return logit, aux_map

    def _anchor_aux_from_unit(self, x_unit):
        model = self._unwrap_model()
        if not hasattr(model, "extract_anchor_aux_map"):
            return None
        x = self._from_unit(x_unit)
        return model.extract_anchor_aux_map(x)

    def _hos_fire_logit_from_aux(self, x_unit, aux_map_o):
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
        logit = self._binary_logit(result["logit"])
        scale = float(getattr(self.opt, "hos_logit_scale", 1.0))
        bias = float(getattr(self.opt, "hos_logit_bias", 0.0))
        return scale * logit + bias

    def predict_logits(self, x):
        mode = getattr(self.opt, "infer_mode", "original")
        if mode == "original":
            return self._backbone_for_forward()(x)

        was_training = self.training
        self.eval()
        with torch.no_grad():
            feat_o = None
            logit_o = None
            aux_map_o = None
            patch_tokens_o = None
            x_unit = self._to_unit(x)

            token_modes = (
                "trained_evidence_avg",
                "trained_multiview_evidence_avg",
                "trained_patch_mil",
                "trained_query_mil",
                "trained_delta_patch_mil_avg",
                "trained_forensic_fusion",
                "trained_hos_fire",
                "trained_forensic_hos_fusion",
                "trained_hos_manifold_logit_fusion",
                "trained_forensic_hos_logit_fusion",
            )
            if mode in token_modes:
                need_aux = (
                    mode in (
                        "trained_hos_fire",
                        "trained_forensic_hos_fusion",
                        "trained_hos_manifold_logit_fusion",
                        "trained_forensic_hos_logit_fusion",
                    )
                    or (
                        mode in (
                            "trained_forensic_fusion",
                            "trained_forensic_hos_fusion",
                            "trained_forensic_hos_logit_fusion",
                        )
                        and self.fire_error_head is not None
                    )
                    or (mode == "trained_forensic_hos_fusion" and self.hos_fire_envelope is not None)
                )
                feat_o, logit_o, aux_map_o, patch_tokens_o = self._forward_branch_with_tokens(
                    x,
                    return_aux_map=need_aux,
                )

            if mode == "trained_hos_fire":
                if self.hos_fire_envelope is None:
                    raise RuntimeError(f"infer_mode={mode} requires --use_hos_fire_envelope checkpoint.")
                if aux_map_o is None:
                    _, _, aux_map_o = self._forward_branch(x, return_aux_map=True)
                p = torch.sigmoid(self._hos_fire_logit_from_aux(x_unit, aux_map_o))
            elif mode == "trained_forensic_hos_fusion":
                if self.hos_fire_envelope is None:
                    raise RuntimeError(f"infer_mode={mode} requires --use_hos_fire_envelope checkpoint.")
                if aux_map_o is None:
                    _, _, aux_map_o = self._forward_branch(x, return_aux_map=True)
                probs_for_fusion = [torch.sigmoid(logit_o)]
                probs_for_fusion.append(torch.sigmoid(self._hos_fire_logit_from_aux(x_unit, aux_map_o)))
                if self.patch_mil_head is not None and patch_tokens_o is not None:
                    probs_for_fusion.append(torch.sigmoid(self.patch_mil_head(patch_tokens_o)))
                if self.evidence_head is not None and patch_tokens_o is not None:
                    probs_for_fusion.append(torch.sigmoid(self.evidence_head(feat_o, patch_tokens_o, x_unit)))
                p = torch.stack(probs_for_fusion, dim=0).mean(dim=0)
            elif mode in ("trained_hos_manifold_logit_fusion", "trained_forensic_hos_logit_fusion"):
                if self.hos_fire_envelope is None:
                    raise RuntimeError(f"infer_mode={mode} requires --use_hos_fire_envelope checkpoint.")
                if aux_map_o is None:
                    _, _, aux_map_o = self._forward_branch(x, return_aux_map=True)
                parts = [
                    self._binary_logit(logit_o),
                    self._hos_fire_logit_from_aux(x_unit, aux_map_o),
                ]
                weights = [
                    float(getattr(self.opt, "logit_fusion_weight_original", 1.0)),
                    float(getattr(self.opt, "logit_fusion_weight_hos", 1.0)),
                ]
                if mode == "trained_forensic_hos_logit_fusion":
                    if self.patch_mil_head is not None and patch_tokens_o is not None:
                        parts.append(self.patch_mil_head(patch_tokens_o))
                        weights.append(float(getattr(self.opt, "logit_fusion_weight_patch", 0.0)))
                    if self.evidence_head is not None and patch_tokens_o is not None:
                        parts.append(self.evidence_head(feat_o, patch_tokens_o, x_unit))
                        weights.append(float(getattr(self.opt, "logit_fusion_weight_evidence", 0.0)))
                p = torch.sigmoid(self._weighted_logit_fusion(parts, weights))

            if mode in ("trained_evidence_avg", "trained_multiview_evidence_avg"):
                if self.evidence_head is None:
                    raise RuntimeError(f"infer_mode={mode} requires --use_evidence_head checkpoint.")
                logit_ev = self.evidence_head(feat_o, patch_tokens_o, x_unit)
                if mode == "trained_evidence_avg":
                    p = 0.5 * (torch.sigmoid(logit_o) + torch.sigmoid(logit_ev))
                else:
                    p_ev = torch.sigmoid(logit_ev)
                    p = None
            else:
                p_ev = None

            if feat_o is None or logit_o is None:
                feat_o, logit_o, _ = self._forward_branch(x, return_aux_map=False)
            p_orig = torch.sigmoid(logit_o)
            probs = [p_orig]

            if mode == "trained_multiview_fusion" or mode == "trained_multiview_evidence_avg":
                if self.multiview_fusion_head is None:
                    raise RuntimeError(f"infer_mode={mode} requires --use_multiview_fusion_head checkpoint.")
                x_fixed = self._from_unit(self.fixed_mapping(x_unit))
                x_random = self._from_unit(self.random_mapping(x_unit))
                feat_f, logit_f, _ = self._forward_branch(x_fixed, return_aux_map=False)
                feat_r, logit_r, _ = self._forward_branch(x_random, return_aux_map=False)
                logit_mv = self.multiview_fusion_head(
                    [feat_o, feat_f, feat_r],
                    [logit_o, logit_f, logit_r],
                )
                if mode == "trained_multiview_fusion":
                    p = torch.sigmoid(logit_mv)
                else:
                    p = 0.5 * (torch.sigmoid(logit_mv) + p_ev)
            elif mode in ("trained_patch_mil", "trained_delta_patch_mil_avg", "trained_forensic_fusion"):
                if self.patch_mil_head is None:
                    raise RuntimeError(f"infer_mode={mode} requires --use_patch_mil_head checkpoint.")
                probs_for_fusion = []
                logit_mil = self.patch_mil_head(patch_tokens_o)
                if mode == "trained_patch_mil":
                    p = torch.sigmoid(logit_mil)
                else:
                    if self.multiview_fusion_head is None:
                        raise RuntimeError(f"infer_mode={mode} requires --use_multiview_fusion_head checkpoint.")
                    x_fixed = self._from_unit(self.fixed_mapping(x_unit))
                    x_random = self._from_unit(self.random_mapping(x_unit))
                    need_pd_tokens = self.patch_disagreement_head is not None and mode == "trained_forensic_fusion"
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
                        patch_tokens_f = None
                        patch_tokens_r = None
                    logit_mv = self.multiview_fusion_head(
                        [feat_o, feat_f, feat_r],
                        [logit_o, logit_f, logit_r],
                    )
                    probs_for_fusion.extend([torch.sigmoid(logit_mv), torch.sigmoid(logit_mil)])
                    if mode == "trained_forensic_fusion":
                        if self.evidence_head is not None:
                            probs_for_fusion.append(torch.sigmoid(self.evidence_head(feat_o, patch_tokens_o, x_unit)))
                        if self.patch_disagreement_head is not None and patch_tokens_f is not None and patch_tokens_r is not None:
                            probs_for_fusion.append(
                                torch.sigmoid(
                                    self.patch_disagreement_head(
                                        patch_tokens_o,
                                        patch_tokens_f,
                                        patch_tokens_r,
                                    )
                                )
                            )
                        if self.fire_error_head is not None:
                            if aux_map_o is None:
                                _, _, aux_map_o = self._forward_branch(x, return_aux_map=True)
                            probs_for_fusion.append(torch.sigmoid(self._fire_error_logit_from_aux(x_unit, aux_map_o)))
                    p = torch.stack(probs_for_fusion, dim=0).mean(dim=0)
            elif mode == "trained_query_mil":
                if self.query_mil_head is None:
                    raise RuntimeError(f"infer_mode={mode} requires --use_query_mil_head checkpoint.")
                p = torch.sigmoid(self.query_mil_head(patch_tokens_o))
            elif mode in (
                "original_fixed_avg",
                "original_fixed_random_avg",
                "original_fixed_random_max",
                "uncertain_multiview",
            ):
                x_fixed = self._from_unit(self.fixed_mapping(x_unit))
                probs.append(self._branch_prob(x_fixed))

            if mode in ("original_fixed_random_avg", "original_fixed_random_max", "uncertain_multiview"):
                random_views = max(1, int(getattr(self.opt, "infer_random_views", 2)))
                for _ in range(random_views):
                    x_random = self._from_unit(self.random_mapping(x_unit))
                    probs.append(self._branch_prob(x_random))

            if mode in ("original_fixed_avg", "original_fixed_random_avg"):
                p = torch.stack(probs, dim=0).mean(dim=0)
            elif mode == "original_fixed_random_max":
                p = torch.stack(probs, dim=0).amax(dim=0)
            elif mode == "uncertain_multiview":
                p_multi = torch.stack(probs, dim=0).mean(dim=0)
                uncertain = torch.abs(p_orig - 0.5) < float(getattr(self.opt, "infer_uncertain_delta", 0.12))
                alpha = float(getattr(self.opt, "infer_multiview_alpha", 0.5))
                p = p_orig.clone()
                p[uncertain] = (1.0 - alpha) * p_orig[uncertain] + alpha * p_multi[uncertain]
            elif mode not in (
                "trained_multiview_fusion",
                "trained_evidence_avg",
                "trained_multiview_evidence_avg",
                "trained_patch_mil",
                "trained_query_mil",
                "trained_delta_patch_mil_avg",
                "trained_forensic_fusion",
                "trained_hos_fire",
                "trained_forensic_hos_fusion",
                "trained_hos_manifold_logit_fusion",
                "trained_forensic_hos_logit_fusion",
            ):
                raise ValueError(f"Unsupported infer_mode: {mode}")

            eps = torch.finfo(p.dtype).eps
            logits = torch.logit(p.clamp(min=eps, max=1.0 - eps))

        if was_training:
            self.train()
        return logits

    def _unwrap_model(self):
        return self.model.module if hasattr(self.model, "module") else self.model

    def _broadcast_module_state(self, module):
        if module is None:
            return
        if not getattr(self.opt, "distributed", False):
            return
        if not (dist.is_available() and dist.is_initialized()):
            return
        for tensor in list(module.parameters()) + list(module.buffers()):
            dist.broadcast(tensor.data, src=0)

    def _broadcast_auxiliary_modules(self):
        modules = [
            self.projector,
            self.mid_prior_predictor,
            self.evidence_head,
            self.multiview_fusion_head,
            self.patch_mil_head,
            self.query_mil_head,
            self.patch_disagreement_head,
            self.fire_error_head,
            self.hos_fire_envelope,
            self.mid_band_mask_head,
            self.recon_score_head,
            self.noise_guidance,
            self.noise_guidance_head,
        ]
        for module in modules:
            self._broadcast_module_state(module)

    def forward(self, epoch=None):
        random_weight = self._warmup_random_weight(epoch)
        zero = self.label.sum() * 0.0

        x_o = self.input
        train_res = int(x_o.shape[-1])
        sampled_res = self._sample_train_resolution()
        if sampled_res > 0:
            train_res = int(sampled_res)
            if x_o.shape[-2] != train_res or x_o.shape[-1] != train_res:
                x_o = F.interpolate(
                    x_o,
                    size=(train_res, train_res),
                    mode="bilinear",
                    align_corners=False,
                )

        x_unit = self._to_unit(x_o)
        need_aux_map = (
            self.training
            and (
                (self.opt.use_mid_frequency_prior and self.opt.mid_prior_scope in ("original", "original_fixed", "all"))
                or self.opt.use_fire_lite
                or self.fire_error_head is not None
                or self.hos_fire_envelope is not None
                or self.opt.use_noise_guidance
                or self.opt.use_fake_hardening
                or self.opt.use_fake_adv_hardening
                or self.evidence_head is not None
            )
        )
        need_patch_tokens = self.training and (
            self.evidence_head is not None
            or self.patch_mil_head is not None
            or self.query_mil_head is not None
            or self.patch_disagreement_head is not None
        )
        if need_patch_tokens:
            feat_o, logit_o, aux_map_o, patch_tokens_o = self._forward_branch_with_tokens(
                x_o,
                return_aux_map=need_aux_map,
            )
        else:
            feat_o, logit_o, aux_map_o = self._forward_branch(x_o, return_aux_map=need_aux_map)
            patch_tokens_o = None
        self.output = logit_o

        loss_cls = self.opt.branch_weight_original * self._classification_loss(logit_o, self.label)
        loss_con = zero
        loss_align = zero
        loss_mid = zero
        loss_mask = zero
        loss_rec = zero
        loss_rank = zero
        loss_noise = zero
        loss_noise_align = zero
        loss_mid_consistency = zero
        loss_evidence = zero
        loss_evidence_align = zero
        loss_multiview_fusion = zero
        loss_real_consistency = zero
        loss_fake_disagreement = zero
        loss_patch_mil = zero
        loss_query_mil = zero
        loss_query_diversity = zero
        loss_patch_disagreement = zero
        loss_fire_error = zero
        loss_fire_error_align = zero
        loss_hos_evidence = zero
        loss_hos_mask = zero
        loss_hos_target = zero
        loss_hos_boundary = zero
        loss_hos_rank = zero
        loss_hos_tangent = zero
        loss_hos_cdc = zero
        loss_hos_anchor = zero
        loss_hos_anchor_residual = zero
        loss_fake_margin = zero
        loss_fake_hard = zero
        loss_fake_hard_consistency = zero
        loss_fake_hard_feat_consistency = zero
        loss_fake_hard_margin = zero
        loss_fake_adv = zero
        loss_fake_adv_consistency = zero
        loss_fake_adv_feat_consistency = zero
        loss_fake_adv_margin = zero
        loss_dual_res_cls = zero
        loss_dual_res_align = zero
        d_real_mean = zero
        d_fake_mean = zero
        d_gap = zero
        fire_active = 0.0
        hos_active = 0.0
        noise_mask_density = zero
        hos_mask_density = zero
        hos_target_density = zero
        hos_delta_mean = zero
        fake_hard_active = 0.0
        fake_adv_active = 0.0
        fake_hard_count = zero
        fake_adv_count = zero

        z_o = None
        feat_f, logit_f = None, None
        feat_r, logit_r = None, None
        patch_tokens_f = None
        patch_tokens_r = None

        if self.training and self.opt.use_three_branch_training and self.opt.use_pixel_mapping:
            if self.opt.use_fixed_mapping_branch:
                x_f = self.fixed_mapping(x_unit)
                x_f = self._from_unit(x_f)
                if self.patch_disagreement_head is not None:
                    feat_f, logit_f, _, patch_tokens_f = self._forward_branch_with_tokens(
                        x_f,
                        return_aux_map=False,
                    )
                else:
                    feat_f, logit_f, _ = self._forward_branch(x_f, return_aux_map=False)
                loss_cls = loss_cls + self.opt.branch_weight_fixed * self._classification_loss(logit_f, self.label)

            if self.opt.use_random_mapping_branch:
                x_r = self.random_mapping(x_unit)
                x_r = self._from_unit(x_r)
                if self.patch_disagreement_head is not None:
                    feat_r, logit_r, _, patch_tokens_r = self._forward_branch_with_tokens(
                        x_r,
                        return_aux_map=False,
                    )
                else:
                    feat_r, logit_r, _ = self._forward_branch(x_r, return_aux_map=False)
                loss_cls = loss_cls + (
                    self.opt.branch_weight_random
                    * random_weight
                    * self._classification_loss(logit_r, self.label)
                )

            if feat_f is not None or feat_r is not None:
                z_o = self.projector(feat_o)

            if feat_f is not None:
                z_f = self.projector(feat_f)
                loss_con = loss_con + info_nce_loss(
                    z_o,
                    z_f,
                    temperature=self.opt.contrastive_temperature,
                )
                loss_align = loss_align + js_divergence(logit_o, logit_f)

            if feat_r is not None:
                z_r = self.projector(feat_r)
                loss_con = loss_con + random_weight * info_nce_loss(
                    z_o,
                    z_r,
                    temperature=self.opt.contrastive_temperature,
                )
                loss_align = loss_align + random_weight * js_divergence(logit_o, logit_r)

            if (
                self.multiview_fusion_head is not None
                and feat_f is not None
                and feat_r is not None
                and random_weight > 0
            ):
                logit_mv = self.multiview_fusion_head(
                    [feat_o, feat_f, feat_r],
                    [logit_o, logit_f, logit_r],
                )
                loss_multiview_fusion = self._classification_loss(logit_mv, self.label)

            multiview_logits = [logit_o]
            if logit_f is not None:
                multiview_logits.append(logit_f)
            if logit_r is not None:
                multiview_logits.append(logit_r)
            if len(multiview_logits) >= 2:
                loss_real_consistency = self._real_consistency_loss(multiview_logits, self.label)
                loss_fake_disagreement = self._fake_disagreement_loss(multiview_logits, self.label)

            if (
                self.patch_disagreement_head is not None
                and patch_tokens_o is not None
                and patch_tokens_f is not None
                and patch_tokens_r is not None
                and random_weight > 0
            ):
                logit_pd = self.patch_disagreement_head(
                    patch_tokens_o,
                    patch_tokens_f,
                    patch_tokens_r,
                )
                loss_patch_disagreement = self._classification_loss(logit_pd, self.label)

        if self.training and self.use_dual_resolution_consistency:
            high_size = max(int(train_res), int(self.dual_res_high_size))
            if x_o.shape[-2] != high_size or x_o.shape[-1] != high_size:
                x_high = F.interpolate(
                    x_o,
                    size=(high_size, high_size),
                    mode="bilinear",
                    align_corners=False,
                )
            else:
                x_high = x_o
            _, logit_high, _ = self._forward_branch(x_high, return_aux_map=False)
            loss_dual_res_cls = self._classification_loss(logit_high, self.label)
            loss_dual_res_align = js_divergence(logit_o, logit_high)

        if self.training and self.opt.use_mid_frequency_prior and aux_map_o is not None and self.mid_prior_predictor is not None:
            use_multiband_mid = int(getattr(self.opt, "mid_prior_out_channels", 1)) > 1
            m_mid = self.mid_frequency_prior(x_unit, return_multiband=use_multiband_mid)
            m_hat_mid = self.mid_prior_predictor(aux_map_o)
            loss_mid = mid_prior_loss(m_hat_mid, m_mid, loss_type=self.opt.mid_prior_loss_type)
            if float(getattr(self.opt, "lambda_mid_consistency", 0.0)) > 0.0:
                with torch.no_grad():
                    x_mid_deg_unit = self.hos_fire_envelope._degrade(x_unit) if self.hos_fire_envelope is not None else x_unit
                _, _, aux_mid_deg = self._forward_branch(self._from_unit(x_mid_deg_unit), return_aux_map=True)
                if aux_mid_deg is not None:
                    m_hat_mid_deg = self.mid_prior_predictor(aux_mid_deg)
                    loss_mid_consistency = F.smooth_l1_loss(m_hat_mid_deg, m_hat_mid.detach())

        if self.training and self.opt.use_fire_lite and aux_map_o is not None and self.mid_band_mask_head is not None and self.recon_score_head is not None:
            interval = max(1, int(self.opt.fire_pseudo_interval))
            if interval <= 1 or (self.total_steps % interval == 0):
                fire_active = 1.0
                fire_m_mid = self.mid_band_mask_head(aux_map_o)
                fire_m_comp = 1.0 - fire_m_mid

                fire_m_comp_input = F.interpolate(
                    fire_m_comp,
                    size=x_unit.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                x_pse_unit = fft_band_filter_rgb(x_unit, fire_m_comp_input)
                x_pse = self._from_unit(x_pse_unit)

                feat_p, _, aux_map_p = self._forward_branch(x_pse, return_aux_map=True)
                if aux_map_p is not None:
                    delta_map = torch.mean(torch.abs(aux_map_o - aux_map_p), dim=1, keepdim=True)
                    score_logit = self.recon_score_head(delta_map).squeeze(1)

                    target_mask = build_mid_target_mask(
                        batch_size=fire_m_mid.shape[0],
                        height=fire_m_mid.shape[-2],
                        width=fire_m_mid.shape[-1],
                        low_ratio=self.opt.mid_freq_radius_low_ratio,
                        high_ratio=self.opt.mid_freq_radius_high_ratio,
                        device=fire_m_mid.device,
                        dtype=fire_m_mid.dtype,
                    )
                    loss_mask = F.l1_loss(fire_m_mid, target_mask) + torch.mean(torch.abs(1.0 - fire_m_mid - fire_m_comp))

                    rec_target = 1.0 - self.label
                    loss_rec = F.binary_cross_entropy_with_logits(score_logit, rec_target)

                    d = feature_l2_distance(feat_o, feat_p)
                    real_mask = self.label < 0.5
                    fake_mask = self.label >= 0.5
                    if real_mask.any() and fake_mask.any():
                        d_real_mean = d[real_mask].mean()
                        d_fake_mean = d[fake_mask].mean()
                        d_gap = d_real_mean - d_fake_mean
                        loss_rank = F.relu(self.opt.fire_rank_margin - d_gap)

        if (
            self.training
            and self.opt.use_noise_guidance
            and aux_map_o is not None
            and self.noise_guidance is not None
            and self.noise_guidance_head is not None
        ):
            guided_map, ng_stats = self.noise_guidance(aux_map_o, x_unit)
            guided_avg = F.adaptive_avg_pool2d(guided_map, output_size=1).flatten(1)
            guided_max = F.adaptive_max_pool2d(guided_map, output_size=1).flatten(1)
            guided_feat = torch.cat([guided_avg, guided_max], dim=1)
            logit_ng = self.noise_guidance_head(guided_feat)
            loss_noise = self._classification_loss(logit_ng, self.label)
            loss_noise_align = js_divergence(logit_o, logit_ng)
            noise_mask_density = ng_stats["mask_density"]

        if self.training and self.evidence_head is not None and patch_tokens_o is not None:
            logit_ev = self.evidence_head(feat_o, patch_tokens_o, x_unit)
            loss_evidence = self._classification_loss(logit_ev, self.label)
            loss_evidence_align = js_divergence(logit_o, logit_ev)

        if self.training and self.patch_mil_head is not None and patch_tokens_o is not None:
            logit_mil = self.patch_mil_head(patch_tokens_o)
            loss_patch_mil = self._classification_loss(logit_mil, self.label)

        if self.training and self.query_mil_head is not None and patch_tokens_o is not None:
            logit_query_mil = self.query_mil_head(patch_tokens_o)
            loss_query_mil = self._classification_loss(logit_query_mil, self.label)
            loss_query_diversity = self.query_mil_head.diversity_loss()

        if self.training and self.fire_error_head is not None and aux_map_o is not None:
            target_mask = build_mid_target_mask(
                batch_size=x_unit.shape[0],
                height=x_unit.shape[-2],
                width=x_unit.shape[-1],
                low_ratio=self.opt.mid_freq_radius_low_ratio,
                high_ratio=self.opt.mid_freq_radius_high_ratio,
                device=x_unit.device,
                dtype=x_unit.dtype,
            )
            x_mid_unit = fft_band_filter_rgb(x_unit, target_mask)
            x_mid = self._from_unit(x_mid_unit)
            _, logit_mid, aux_map_mid = self._forward_branch(x_mid, return_aux_map=True)
            if aux_map_mid is not None:
                delta_map = torch.mean(torch.abs(aux_map_o - aux_map_mid), dim=1, keepdim=True)
                logit_fire_error = self.fire_error_head(delta_map)
                loss_fire_error = self._classification_loss(logit_fire_error, self.label)
                loss_fire_error_align = js_divergence(logit_o, logit_fire_error)

        if self.training and self.hos_fire_envelope is not None and aux_map_o is not None:
            interval = max(1, int(getattr(self.opt, "hos_forward_interval", 1)))
            if interval <= 1 or (self.total_steps % interval == 0):
                hos_active = 1.0
                hos_result = self.hos_fire_envelope(
                    x_unit=x_unit,
                    aux_map_o=aux_map_o,
                    label=self.label,
                    aux_map_forward=self._forward_unit_for_aux,
                    cls_logit_o=logit_o,
                    anchor_aux_forward=self._anchor_aux_from_unit,
                    compute_boundary=(
                        float(getattr(self.opt, "lambda_hos_boundary", 0.0)) > 0.0
                        or float(getattr(self.opt, "lambda_hos_rank", 0.0)) > 0.0
                        or float(getattr(self.opt, "lambda_hos_tangent", 0.0)) > 0.0
                    ),
                    compute_cdc=(
                        float(getattr(self.opt, "lambda_hos_cdc", 0.0)) > 0.0
                        or float(getattr(self.opt, "lambda_hos_anchor_residual", 0.0)) > 0.0
                    ),
                    compute_anchor=float(getattr(self.opt, "lambda_hos_anchor", 0.0)) > 0.0,
                )
                loss_hos_evidence = hos_result["loss_evidence"]
                loss_hos_mask = hos_result["loss_mask"]
                loss_hos_target = hos_result["loss_target"]
                loss_hos_boundary = hos_result["loss_boundary"]
                loss_hos_rank = hos_result["loss_rank"]
                loss_hos_tangent = hos_result["loss_tangent"]
                loss_hos_cdc = hos_result["loss_cdc"]
                loss_hos_anchor = hos_result["loss_anchor"]
                loss_hos_anchor_residual = hos_result["loss_anchor_res"]
                hos_stats = hos_result["stats"]
                hos_mask_density = hos_stats["mask_density"]
                hos_target_density = hos_stats["target_density"]
                hos_delta_mean = hos_stats["delta_mean"]

        if self.training and self.opt.use_fake_hardening and self._epoch_ready(epoch, self.opt.fake_hard_start_epoch):
            fake_idx = self._sample_fake_indices(self.label, self.opt.fake_hard_prob, self.opt.fake_hard_max_batch)
            if fake_idx.numel() > 0:
                fake_hard_active = 1.0
                fake_hard_count = feat_o.new_tensor(float(fake_idx.numel()))
                x_fake_clean = x_unit[fake_idx]
                feat_fake_clean = feat_o[fake_idx]
                logit_fake_clean = logit_o[fake_idx]
                x_fake_hard = self._apply_social_chain_torch(x_fake_clean, hard=True)
                (
                    _feat_hard,
                    _logit_hard,
                    _hos_logit_hard,
                    loss_fake_hard_view,
                    loss_fake_hard_cons_view,
                    loss_fake_hard_feat_view,
                    loss_fake_hard_margin_view,
                ) = self._fake_hard_view_losses(
                    x_fake_hard,
                    feat_fake_clean,
                    logit_fake_clean,
                    use_hos=bool(getattr(self.opt, "fake_hard_use_hos", True)),
                )
                loss_fake_hard = loss_fake_hard + loss_fake_hard_view
                loss_fake_hard_consistency = loss_fake_hard_consistency + loss_fake_hard_cons_view
                loss_fake_hard_feat_consistency = loss_fake_hard_feat_consistency + loss_fake_hard_feat_view
                loss_fake_hard_margin = loss_fake_hard_margin + loss_fake_hard_margin_view

        if self.training and self.opt.use_fake_adv_hardening and self._epoch_ready(epoch, self.opt.fake_adv_start_epoch):
            fake_idx = self._sample_fake_indices(self.label, self.opt.fake_adv_prob, self.opt.fake_adv_max_batch)
            if fake_idx.numel() > 0:
                fake_adv_active = 1.0
                fake_adv_count = feat_o.new_tensor(float(fake_idx.numel()))
                x_fake_clean = x_unit[fake_idx]
                feat_fake_clean = feat_o[fake_idx]
                logit_fake_clean = logit_o[fake_idx]
                x_fake_adv = self._fake_adv_view(
                    x_fake_clean,
                    attack_use_main=bool(getattr(self.opt, "fake_adv_attack_use_main", True)),
                    attack_use_hos=bool(getattr(self.opt, "fake_adv_attack_use_hos", True)),
                )
                (
                    _feat_adv,
                    _logit_adv,
                    _hos_logit_adv,
                    loss_fake_adv_view,
                    loss_fake_adv_cons_view,
                    loss_fake_adv_feat_view,
                    loss_fake_adv_margin_view,
                ) = self._fake_hard_view_losses(
                    x_fake_adv,
                    feat_fake_clean,
                    logit_fake_clean,
                    use_hos=bool(getattr(self.opt, "fake_adv_train_use_hos", True)),
                )
                loss_fake_adv = loss_fake_adv + loss_fake_adv_view
                loss_fake_adv_consistency = loss_fake_adv_consistency + loss_fake_adv_cons_view
                loss_fake_adv_feat_consistency = loss_fake_adv_feat_consistency + loss_fake_adv_feat_view
                loss_fake_adv_margin = loss_fake_adv_margin + loss_fake_adv_margin_view

        loss_fake_margin = self._fake_margin_loss(logit_o, self.label)

        loss_orth = zero
        loss_ksv = zero
        if self.opt.use_svd:
            loss_orth, loss_ksv = self._unwrap_model().svd_regularization_losses()

        self.loss = (
            loss_cls
            + self.opt.lambda_con * loss_con
            + self.opt.lambda_align * loss_align
            + self.opt.lambda_mid * loss_mid
            + self.opt.lambda_mid_consistency * loss_mid_consistency
            + self.opt.lambda_mask * loss_mask
            + self.opt.lambda_rec * loss_rec
            + self.opt.lambda_rank * loss_rank
            + self.opt.lambda_noise * loss_noise
            + self.opt.lambda_noise_align * loss_noise_align
            + self.opt.lambda_evidence * loss_evidence
            + self.opt.lambda_evidence_align * loss_evidence_align
            + self.opt.lambda_multiview_fusion * loss_multiview_fusion
            + self.opt.lambda_real_consistency * loss_real_consistency
            + self.opt.lambda_fake_disagreement * loss_fake_disagreement
            + self.opt.lambda_patch_mil * loss_patch_mil
            + self.opt.lambda_query_mil * loss_query_mil
            + self.opt.lambda_query_diversity * loss_query_diversity
            + self.opt.lambda_patch_disagreement * loss_patch_disagreement
            + self.opt.lambda_fire_error * loss_fire_error
            + self.opt.lambda_fire_error_align * loss_fire_error_align
            + self.opt.lambda_hos_evidence * loss_hos_evidence
            + self.opt.lambda_hos_mask * loss_hos_mask
            + self.opt.lambda_hos_target * loss_hos_target
            + self.opt.lambda_hos_boundary * loss_hos_boundary
            + self.opt.lambda_hos_rank * loss_hos_rank
            + self.opt.lambda_hos_tangent * loss_hos_tangent
            + self.opt.lambda_hos_cdc * loss_hos_cdc
            + self.opt.lambda_hos_anchor * loss_hos_anchor
            + self.opt.lambda_hos_anchor_residual * loss_hos_anchor_residual
            + self.opt.lambda_fake_margin * loss_fake_margin
            + self.opt.lambda_fake_hard * loss_fake_hard
            + self.opt.lambda_fake_hard_consistency * loss_fake_hard_consistency
            + self.opt.lambda_fake_hard_feat_consistency * loss_fake_hard_feat_consistency
            + self.opt.lambda_fake_hard_margin * loss_fake_hard_margin
            + self.opt.lambda_fake_adv * loss_fake_adv
            + self.opt.lambda_fake_adv_consistency * loss_fake_adv_consistency
            + self.opt.lambda_fake_adv_feat_consistency * loss_fake_adv_feat_consistency
            + self.opt.lambda_fake_adv_margin * loss_fake_adv_margin
            + self.opt.lambda_dual_res_cls * loss_dual_res_cls
            + self.opt.lambda_dual_res_align * loss_dual_res_align
            + self.opt.lambda_orth * loss_orth
            + self.opt.lambda_ksv * loss_ksv
        )

        self.loss_dict = {
            "total": float(self.loss.detach().item()),
            "cls": float(loss_cls.detach().item()),
            "con": float(loss_con.detach().item()),
            "align": float(loss_align.detach().item()),
            "mid": float(loss_mid.detach().item()),
            "mid_consistency": float(loss_mid_consistency.detach().item()),
            "mask": float(loss_mask.detach().item()),
            "rec": float(loss_rec.detach().item()),
            "rank": float(loss_rank.detach().item()),
            "noise": float(loss_noise.detach().item()),
            "noise_align": float(loss_noise_align.detach().item()),
            "evidence": float(loss_evidence.detach().item()),
            "evidence_align": float(loss_evidence_align.detach().item()),
            "multiview_fusion": float(loss_multiview_fusion.detach().item()),
            "real_consistency": float(loss_real_consistency.detach().item()),
            "fake_disagreement": float(loss_fake_disagreement.detach().item()),
            "patch_mil": float(loss_patch_mil.detach().item()),
            "query_mil": float(loss_query_mil.detach().item()),
            "query_diversity": float(loss_query_diversity.detach().item()),
            "patch_disagreement": float(loss_patch_disagreement.detach().item()),
            "fire_error": float(loss_fire_error.detach().item()),
            "fire_error_align": float(loss_fire_error_align.detach().item()),
            "hos_evidence": float(loss_hos_evidence.detach().item()),
            "hos_mask": float(loss_hos_mask.detach().item()),
            "hos_target": float(loss_hos_target.detach().item()),
            "hos_boundary": float(loss_hos_boundary.detach().item()),
            "hos_rank": float(loss_hos_rank.detach().item()),
            "hos_tangent": float(loss_hos_tangent.detach().item()),
            "hos_cdc": float(loss_hos_cdc.detach().item()),
            "hos_anchor": float(loss_hos_anchor.detach().item()),
            "hos_anchor_residual": float(loss_hos_anchor_residual.detach().item()),
            "fake_margin": float(loss_fake_margin.detach().item()),
            "fake_hard": float(loss_fake_hard.detach().item()),
            "fake_hard_consistency": float(loss_fake_hard_consistency.detach().item()),
            "fake_hard_feat_consistency": float(loss_fake_hard_feat_consistency.detach().item()),
            "fake_hard_margin": float(loss_fake_hard_margin.detach().item()),
            "fake_adv": float(loss_fake_adv.detach().item()),
            "fake_adv_consistency": float(loss_fake_adv_consistency.detach().item()),
            "fake_adv_feat_consistency": float(loss_fake_adv_feat_consistency.detach().item()),
            "fake_adv_margin": float(loss_fake_adv_margin.detach().item()),
            "dual_res_cls": float(loss_dual_res_cls.detach().item()),
            "dual_res_align": float(loss_dual_res_align.detach().item()),
            "orth": float(loss_orth.detach().item()),
            "ksv": float(loss_ksv.detach().item()),
            "d_real": float(d_real_mean.detach().item()),
            "d_fake": float(d_fake_mean.detach().item()),
            "d_gap": float(d_gap.detach().item()),
            "fire_active": float(fire_active),
            "hos_active": float(hos_active),
            "noise_mask_density": float(noise_mask_density.detach().item()),
            "hos_mask_density": float(hos_mask_density.detach().item()),
            "hos_target_density": float(hos_target_density.detach().item()),
            "hos_delta_mean": float(hos_delta_mean.detach().item()),
            "fake_hard_active": float(fake_hard_active),
            "fake_adv_active": float(fake_adv_active),
            "fake_hard_count": float(fake_hard_count.detach().item()),
            "fake_adv_count": float(fake_adv_count.detach().item()),
            "random_warmup": float(random_weight),
            "train_res": float(train_res),
        }

    def get_loss(self):
        return self.loss

    def _sync_module_grads(self, module):
        if not getattr(self.opt, "distributed", False):
            return
        if not (dist.is_available() and dist.is_initialized()):
            return
        for p in module.parameters():
            if p.grad is None:
                continue
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad /= float(self.opt.world_size)

    @staticmethod
    def _module_grad_norm(module):
        if module is None:
            return 0.0
        total = 0.0
        for p in module.parameters():
            if p.grad is None:
                continue
            total += float(p.grad.detach().float().pow(2).sum().item())
        return total ** 0.5

    def optimize_parameters(self, epoch=None):
        self.update_learning_rate()
        self.optimizer.zero_grad()
        with self._autocast_ctx():
            self.forward(epoch=epoch)

        if self.use_amp and self.amp_scaler.is_enabled():
            self.amp_scaler.scale(self.loss).backward()
            self.amp_scaler.unscale_(self.optimizer)
        else:
            self.loss.backward()

        self._sync_module_grads(self.projector)
        if self.mid_prior_predictor is not None:
            self._sync_module_grads(self.mid_prior_predictor)
        if self.evidence_head is not None:
            self._sync_module_grads(self.evidence_head)
        if self.multiview_fusion_head is not None:
            self._sync_module_grads(self.multiview_fusion_head)
        if self.patch_mil_head is not None:
            self._sync_module_grads(self.patch_mil_head)
        if self.query_mil_head is not None:
            self._sync_module_grads(self.query_mil_head)
        if self.patch_disagreement_head is not None:
            self._sync_module_grads(self.patch_disagreement_head)
        if self.fire_error_head is not None:
            self._sync_module_grads(self.fire_error_head)
        if self.hos_fire_envelope is not None:
            self._sync_module_grads(self.hos_fire_envelope)
        if self.mid_band_mask_head is not None:
            self._sync_module_grads(self.mid_band_mask_head)
        if self.recon_score_head is not None:
            self._sync_module_grads(self.recon_score_head)
        if self.noise_guidance is not None:
            self._sync_module_grads(self.noise_guidance)
        if self.noise_guidance_head is not None:
            self._sync_module_grads(self.noise_guidance_head)

        if self.loss_dict is not None:
            self.loss_dict["hos_grad_norm"] = self._module_grad_norm(self.hos_fire_envelope)
            self.loss_dict["backbone_grad_norm"] = self._module_grad_norm(self.model)
            self.loss_dict["score_net_grad_norm"] = (
                self._module_grad_norm(self.hos_fire_envelope.score_net)
                if self.hos_fire_envelope is not None
                else 0.0
            )
            self.loss_dict["mask_net_grad_norm"] = (
                self._module_grad_norm(self.hos_fire_envelope.mask_net)
                if self.hos_fire_envelope is not None
                else 0.0
            )

        if self.use_amp and self.amp_scaler.is_enabled():
            self.amp_scaler.step(self.optimizer)
            self.amp_scaler.update()
        else:
            self.optimizer.step()

