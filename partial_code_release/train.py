import os
import time
import random
import re
from datetime import timedelta
from tensorboardX import SummaryWriter
from tqdm import tqdm

from validate import validate
from data import create_dataloader
from earlystop import EarlyStopping
from models.trainer import Trainer
from options.train_options import TrainOptions
from models.checkpoint_utils import load_partial_state_dict_with_prefix_compat
import torch
import torch.distributed as dist
import numpy as np


SEED = 0
def set_seed(rank_offset=0):
    seed = SEED + int(rank_offset)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_distributed(opt):
    if not getattr(opt, "distributed", False):
        return
    timeout_minutes = max(1, int(getattr(opt, "ddp_timeout_minutes", 180)))
    init_kwargs = dict(
        backend="nccl",
        init_method="env://",
        timeout=timedelta(minutes=timeout_minutes),
    )
    try:
        dist.init_process_group(
            **init_kwargs,
            device_id=torch.device(f"cuda:{opt.gpu_ids[0]}"),
        )
    except TypeError:
        # Older PyTorch versions do not support device_id.
        dist.init_process_group(**init_kwargs)
    torch.cuda.set_device(opt.gpu_ids[0])


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(opt):
    return (not getattr(opt, "distributed", False)) or opt.rank == 0


def maybe_set_sampler_epoch(data_loader, epoch):
    sampler = getattr(data_loader, "sampler", None)
    if sampler is not None and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)


def get_eval_opt(opt, data_label):
    val_opt = TrainOptions().parse(print_options=False)
    val_opt.isTrain = False
    val_opt.no_resize = False
    val_opt.no_crop = False
    val_opt.serial_batches = True
    val_opt.data_label = data_label
    val_opt.distributed = getattr(opt, "distributed", False)
    val_opt.world_size = getattr(opt, "world_size", 1)
    val_opt.rank = getattr(opt, "rank", 0)
    val_opt.local_rank = getattr(opt, "local_rank", 0)
    val_opt.distributed_eval = getattr(opt, "distributed", False)
    if getattr(opt, "eval_batch_size", 0) > 0:
        val_opt.batch_size = opt.eval_batch_size

    return val_opt


def log_training_setup(opt, log_path):
    setup_lines = [
        "===== Three-Branch / Mid-Prior Setup =====",
        f"lr: {opt.lr}",
        f"lr_warmup_steps: {opt.lr_warmup_steps}",
        f"lr_warmup_start_factor: {opt.lr_warmup_start_factor}",
        f"lr_warmup_reset_on_resume: {opt.lr_warmup_reset_on_resume}",
        f"use_three_branch_training: {opt.use_three_branch_training}",
        f"data_aug: {opt.data_aug}",
        f"color_aug_prob: {opt.color_aug_prob}",
        f"brightness_range: {opt.brightness_range}",
        f"contrast_range: {opt.contrast_range}",
        f"saturation_range: {opt.saturation_range}",
        f"resize_aug_prob: {opt.resize_aug_prob}",
        f"resize_aug_scale: {opt.resize_aug_scale}",
        f"train_max_per_class: {opt.train_max_per_class}",
        f"eval_max_per_class: {opt.eval_max_per_class}",
        f"debug_no_save: {opt.debug_no_save}",
        f"use_pixel_mapping: {opt.use_pixel_mapping}",
        f"use_fixed_mapping_branch: {opt.use_fixed_mapping_branch}",
        f"use_random_mapping_branch: {opt.use_random_mapping_branch}",
        f"proj_dim: {opt.proj_dim}",
        f"contrastive_temperature: {opt.contrastive_temperature}",
        f"lambda_con: {opt.lambda_con}",
        f"lambda_align: {opt.lambda_align}",
        f"branch_weight_original: {opt.branch_weight_original}",
        f"branch_weight_fixed: {opt.branch_weight_fixed}",
        f"branch_weight_random: {opt.branch_weight_random}",
        f"use_multiview_fusion_head: {opt.use_multiview_fusion_head}",
        f"lambda_multiview_fusion: {opt.lambda_multiview_fusion}",
        f"multiview_fusion_type: {opt.multiview_fusion_type}",
        f"use_evidence_head: {opt.use_evidence_head}",
        f"lambda_evidence: {opt.lambda_evidence}",
        f"lambda_evidence_align: {opt.lambda_evidence_align}",
        f"lambda_real_consistency: {opt.lambda_real_consistency}",
        f"lambda_fake_disagreement: {opt.lambda_fake_disagreement}",
        f"fake_disagreement_margin: {opt.fake_disagreement_margin}",
        f"use_patch_mil_head: {opt.use_patch_mil_head}",
        f"lambda_patch_mil: {opt.lambda_patch_mil}",
        f"patch_mil_topk_ratio: {opt.patch_mil_topk_ratio}",
        f"use_query_mil_head: {opt.use_query_mil_head}",
        f"lambda_query_mil: {opt.lambda_query_mil}",
        f"lambda_query_diversity: {opt.lambda_query_diversity}",
        f"query_mil_num_queries: {opt.query_mil_num_queries}",
        f"query_mil_topk_ratio: {opt.query_mil_topk_ratio}",
        f"use_cross_view_patch_disagreement: {opt.use_cross_view_patch_disagreement}",
        f"lambda_patch_disagreement: {opt.lambda_patch_disagreement}",
        f"patch_disagreement_topk_ratio: {opt.patch_disagreement_topk_ratio}",
        f"use_fire_error_evidence: {opt.use_fire_error_evidence}",
        f"lambda_fire_error: {opt.lambda_fire_error}",
        f"lambda_fire_error_align: {opt.lambda_fire_error_align}",
        f"fire_error_topk_ratio: {opt.fire_error_topk_ratio}",
        f"use_hos_fire_envelope: {opt.use_hos_fire_envelope}",
        f"lambda_hos_evidence: {opt.lambda_hos_evidence}",
        f"lambda_hos_mask: {opt.lambda_hos_mask}",
        f"lambda_hos_target: {opt.lambda_hos_target}",
        f"lambda_hos_boundary: {opt.lambda_hos_boundary}",
        f"lambda_hos_rank: {opt.lambda_hos_rank}",
        f"lambda_hos_tangent: {opt.lambda_hos_tangent}",
        f"lambda_hos_cdc: {opt.lambda_hos_cdc}",
        f"lambda_hos_anchor: {opt.lambda_hos_anchor}",
        f"lambda_hos_anchor_residual: {opt.lambda_hos_anchor_residual}",
        f"train_hos_only: {opt.train_hos_only}",
        f"hos_topk_ratio: {opt.hos_topk_ratio}",
        f"hos_num_phase_shifts: {opt.hos_num_phase_shifts}",
        f"hos_boundary_phase_eps: {opt.hos_boundary_phase_eps}",
        f"hos_rank_margin: {opt.hos_rank_margin}",
        f"hos_evidence_fake_weight: {opt.hos_evidence_fake_weight}",
        f"hos_boundary_cls_weight: {opt.hos_boundary_cls_weight}",
        f"hos_tangent_rank: {opt.hos_tangent_rank}",
        f"hos_detach_aux: {opt.hos_detach_aux}",
        f"hos_forward_interval: {opt.hos_forward_interval}",
        f"use_focal_loss: {opt.use_focal_loss}",
        f"focal_gamma: {opt.focal_gamma}",
        f"fake_loss_weight: {opt.fake_loss_weight}",
        f"lambda_fake_margin: {opt.lambda_fake_margin}",
        f"fake_margin: {opt.fake_margin}",
        f"use_domain_balanced_sampler: {opt.use_domain_balanced_sampler}",
        f"svd_residual_rank: {opt.svd_residual_rank}",
        f"svd_keep_rank_ratio: {opt.svd_keep_rank_ratio}",
        f"random_branch_warmup_epochs: {opt.random_branch_warmup_epochs}",
        f"infer_mode: {opt.infer_mode}",
        f"use_mid_frequency_prior: {opt.use_mid_frequency_prior}",
        f"mid_prior_scope: {opt.mid_prior_scope}",
        f"lambda_mid: {opt.lambda_mid}",
        f"mid_prior_loss_type: {opt.mid_prior_loss_type}",
        f"mid_prior_use_grayscale: {opt.mid_prior_use_grayscale}",
        f"mid_freq_radius_low_ratio: {opt.mid_freq_radius_low_ratio}",
        f"mid_freq_radius_high_ratio: {opt.mid_freq_radius_high_ratio}",
        f"mid_prior_from_layer: {opt.mid_prior_from_layer}",
        f"mid_prior_predictor_hidden_dim: {opt.mid_prior_predictor_hidden_dim}",
        f"mid_prior_predictor_depth: {opt.mid_prior_predictor_depth}",
        f"mid_prior_out_channels: {opt.mid_prior_out_channels}",
        f"lambda_mid_consistency: {opt.lambda_mid_consistency}",
        f"use_fire_lite: {opt.use_fire_lite}",
        f"lambda_mask: {opt.lambda_mask}",
        f"lambda_rec: {opt.lambda_rec}",
        f"lambda_rank: {opt.lambda_rank}",
        f"fire_rank_margin: {opt.fire_rank_margin}",
        f"fire_mask_hidden_dim: {opt.fire_mask_hidden_dim}",
        f"fire_recon_hidden_dim: {opt.fire_recon_hidden_dim}",
        f"fire_pseudo_interval: {opt.fire_pseudo_interval}",
        f"use_noise_guidance: {opt.use_noise_guidance}",
        f"noise_guidance_scales: {opt.noise_guidance_scales}",
        f"noise_guidance_embed_dim: {opt.noise_guidance_embed_dim}",
        f"noise_guidance_head_hidden_dim: {opt.noise_guidance_head_hidden_dim}",
        f"noise_guidance_head_dropout: {opt.noise_guidance_head_dropout}",
        f"noise_guidance_topk_ratio: {opt.noise_guidance_topk_ratio}",
        f"noise_guidance_max_tokens: {opt.noise_guidance_max_tokens}",
        f"noise_guidance_detach_mask: {opt.noise_guidance_detach_mask}",
        f"noise_guidance_use_grayscale: {opt.noise_guidance_use_grayscale}",
        f"lambda_noise: {opt.lambda_noise}",
        f"lambda_noise_align: {opt.lambda_noise_align}",
        f"use_social_chain_aug: {getattr(opt, 'use_social_chain_aug', False)}",
        f"social_chain_prob: {getattr(opt, 'social_chain_prob', 1.0)}",
        f"social_chain_min_ops: {getattr(opt, 'social_chain_min_ops', 2)}",
        f"social_chain_max_ops: {getattr(opt, 'social_chain_max_ops', 4)}",
        f"use_fake_dataset_hardening: {getattr(opt, 'use_fake_dataset_hardening', False)}",
        f"fake_dataset_hard_prob: {getattr(opt, 'fake_dataset_hard_prob', 0.35)}",
        f"fake_social_chain_min_ops: {getattr(opt, 'fake_social_chain_min_ops', 3)}",
        f"fake_social_chain_max_ops: {getattr(opt, 'fake_social_chain_max_ops', 5)}",
        f"fake_dataset_use_mid_suppress: {getattr(opt, 'fake_dataset_use_mid_suppress', True)}",
        f"use_fake_hardening: {getattr(opt, 'use_fake_hardening', False)}",
        f"fake_hard_start_epoch: {getattr(opt, 'fake_hard_start_epoch', 2)}",
        f"fake_hard_prob: {getattr(opt, 'fake_hard_prob', 0.35)}",
        f"fake_hard_max_batch: {getattr(opt, 'fake_hard_max_batch', 32)}",
        f"lambda_fake_hard: {getattr(opt, 'lambda_fake_hard', 0.3)}",
        f"lambda_fake_hard_consistency: {getattr(opt, 'lambda_fake_hard_consistency', 0.05)}",
        f"lambda_fake_hard_feat_consistency: {getattr(opt, 'lambda_fake_hard_feat_consistency', 0.05)}",
        f"lambda_fake_hard_margin: {getattr(opt, 'lambda_fake_hard_margin', 0.1)}",
        f"fake_hard_margin: {getattr(opt, 'fake_hard_margin', 1.0)}",
        f"fake_hard_use_hos: {getattr(opt, 'fake_hard_use_hos', True)}",
        f"fake_hard_hos_weight: {getattr(opt, 'fake_hard_hos_weight', 1.0)}",
        f"use_fake_adv_hardening: {getattr(opt, 'use_fake_adv_hardening', False)}",
        f"fake_adv_start_epoch: {getattr(opt, 'fake_adv_start_epoch', 3)}",
        f"fake_adv_prob: {getattr(opt, 'fake_adv_prob', 0.25)}",
        f"fake_adv_max_batch: {getattr(opt, 'fake_adv_max_batch', 8)}",
        f"fake_adv_eps: {getattr(opt, 'fake_adv_eps', 4.0)}",
        f"fake_adv_alpha: {getattr(opt, 'fake_adv_alpha', 2.0)}",
        f"fake_adv_steps: {getattr(opt, 'fake_adv_steps', 2)}",
        f"fake_adv_eot_views: {getattr(opt, 'fake_adv_eot_views', 1)}",
        f"fake_adv_random_start: {getattr(opt, 'fake_adv_random_start', True)}",
        f"fake_adv_on_social_chain: {getattr(opt, 'fake_adv_on_social_chain', True)}",
        f"fake_adv_attack_use_main: {getattr(opt, 'fake_adv_attack_use_main', True)}",
        f"fake_adv_attack_use_hos: {getattr(opt, 'fake_adv_attack_use_hos', True)}",
        f"fake_adv_train_use_hos: {getattr(opt, 'fake_adv_train_use_hos', True)}",
        f"lambda_fake_adv: {getattr(opt, 'lambda_fake_adv', 0.2)}",
        f"lambda_fake_adv_consistency: {getattr(opt, 'lambda_fake_adv_consistency', 0.05)}",
        f"lambda_fake_adv_feat_consistency: {getattr(opt, 'lambda_fake_adv_feat_consistency', 0.05)}",
        f"lambda_fake_adv_margin: {getattr(opt, 'lambda_fake_adv_margin', 0.1)}",
        f"fake_adv_margin: {getattr(opt, 'fake_adv_margin', 1.0)}",
        f"use_multi_resolution_training: {opt.use_multi_resolution_training}",
        f"multi_res_sizes: {opt.multi_res_sizes}",
        f"use_dual_resolution_consistency: {opt.use_dual_resolution_consistency}",
        f"dual_res_high_size: {opt.dual_res_high_size}",
        f"lambda_dual_res_cls: {opt.lambda_dual_res_cls}",
        f"lambda_dual_res_align: {opt.lambda_dual_res_align}",
        f"val_split: {opt.val_split}",
        f"test_split: {opt.test_split}",
        f"test_freq: {opt.test_freq}",
        f"eval_batch_size: {opt.eval_batch_size}",
        f"ddp_timeout_minutes: {opt.ddp_timeout_minutes}",
        f"resume_ckpt: {opt.resume_ckpt}",
        f"resume_epoch: {opt.resume_epoch}",
        f"new_optim: {opt.new_optim}",
        f"use_amp: {opt.use_amp}",
        f"amp_dtype: {opt.amp_dtype}",
        f"dinov3_repo_dir: {opt.dinov3_repo_dir}",
        f"dinov3_weights: {opt.dinov3_weights}",
        "===========================================",
    ]
    with open(log_path, "a") as f:
        for line in setup_lines:
            f.write(line + "\n")


def unwrap_model(module):
    return module.module if hasattr(module, "module") else module


def infer_completed_epoch_from_path(path):
    match = re.search(r"model_epoch_(\d+)\.pth$", os.path.basename(path))
    if match:
        return int(match.group(1))
    return -1


def resolve_resume_path(opt):
    resume_path = opt.resume_ckpt
    if not resume_path:
        return ""
    if os.path.isabs(resume_path) or os.path.exists(resume_path):
        return resume_path
    return os.path.join(opt.checkpoints_dir, opt.name, resume_path)


def load_resume_checkpoint(model, ckpt_path, load_optimizer=True):
    checkpoint = torch.load(ckpt_path, map_location=model.device)
    model_to_load = unwrap_model(model.model)
    model_to_load.load_state_dict(checkpoint["model"], strict=True)

    optional_modules = [
        "projector",
        "mid_prior_predictor",
        "evidence_head",
        "multiview_fusion_head",
        "patch_mil_head",
        "query_mil_head",
        "patch_disagreement_head",
        "fire_error_head",
        "hos_fire_envelope",
        "mid_band_mask_head",
        "recon_score_head",
        "noise_guidance",
        "noise_guidance_head",
    ]
    for name in optional_modules:
        module = getattr(model, name, None)
        if module is not None and name in checkpoint:
            try:
                module.load_state_dict(checkpoint[name], strict=True)
            except RuntimeError as exc:
                partial = load_partial_state_dict_with_prefix_compat(module, checkpoint[name])
                if is_main_process(model.opt):
                    print(
                        f"Partially loaded optional module '{name}' after strict load failed: "
                        f"loaded={partial['loaded_keys']}, skipped_shape={len(partial['skipped_shape_keys'])}. "
                        f"Original error: {exc}"
                    )

    if load_optimizer and "optimizer" in checkpoint:
        try:
            model.optimizer.load_state_dict(checkpoint["optimizer"])
        except (RuntimeError, ValueError) as exc:
            if is_main_process(model.opt):
                print(f"Skip optimizer state from resume checkpoint because it is incompatible: {exc}")

    model.total_steps = int(checkpoint.get("total_steps", 0))
    return checkpoint


if __name__ == '__main__':
    opt = TrainOptions().parse()
    setup_distributed(opt)
    set_seed(opt.rank if getattr(opt, "distributed", False) else 0)

    model = Trainer(opt)
    start_epoch = 0
    if getattr(opt, "resume_ckpt", ""):
        resume_path = resolve_resume_path(opt)
        if not os.path.exists(resume_path):
            raise FileNotFoundError("resume_ckpt not found: %s" % resume_path)
        load_resume_checkpoint(model, resume_path, load_optimizer=not opt.new_optim)
        completed_epoch = opt.resume_epoch
        if completed_epoch < 0:
            completed_epoch = infer_completed_epoch_from_path(resume_path)
        start_epoch = completed_epoch + 1 if completed_epoch >= 0 else 0
        if getattr(opt, "lr_warmup_reset_on_resume", False):
            model.reset_lr_warmup()
        if is_main_process(opt):
            optim_msg = "with optimizer" if not opt.new_optim else "without optimizer"
            print(
                "Resumed from %s (%s). total_steps=%s, start_epoch=%s"
                % (resume_path, optim_msg, model.total_steps, start_epoch)
            )

    data_loader = create_dataloader(opt)
    main_process = is_main_process(opt)

    val_opt = get_eval_opt(opt, opt.val_split)
    val_loader = create_dataloader(val_opt)
    test_loader = None
    if opt.test_freq > 0:
        test_opt = get_eval_opt(opt, opt.test_split)
        test_loader = create_dataloader(test_opt)
    train_writer = None
    val_writer = None
    test_writer = None
    early_stopping = None
    start_time = time.time()
    log_file = os.path.join(opt.checkpoints_dir, opt.name, 'log.txt')
    if main_process:
        train_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "train"))
        val_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "val"))
        if test_loader is not None:
            test_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "test"))
        early_stopping = EarlyStopping(patience=opt.earlystop_epoch, delta=-0.001, verbose=True)
        print("Length of data loader: %d" % (len(data_loader)))
        with open(log_file, 'a') as f:
            f.write("Length of data loader: %d \n" % (len(data_loader)))
        log_training_setup(opt, log_file)

    for epoch in range(start_epoch, opt.niter):
        maybe_set_sampler_epoch(data_loader, epoch)
        if main_process and not getattr(opt, "debug_no_save", False):
            model.save_networks('model_epoch_init.pth')

        if main_process:
            epoch_pbar = tqdm(
                data_loader,
                total=len(data_loader),
                desc=f"Epoch {epoch + 1}/{opt.niter}",
                leave=True,
                dynamic_ncols=True,
            )
        else:
            epoch_pbar = data_loader

        for i, data in enumerate(epoch_pbar):
            model.total_steps += 1

            model.set_input(data)
            model.optimize_parameters(epoch=epoch)

            if main_process and model.loss_dict:
                epoch_pbar.set_postfix(
                    loss=f"{model.loss_dict['total']:.4f}",
                    cls=f"{model.loss_dict['cls']:.4f}",
                    rw=f"{model.loss_dict['random_warmup']:.2f}",
                    res=f"{int(model.loss_dict['train_res'])}",
                )

            if main_process and model.total_steps % opt.loss_freq == 0:
                print("Train loss: {} at step: {}".format(model.loss_dict["total"], model.total_steps))
                train_writer.add_scalar('loss_total', model.loss_dict["total"], model.total_steps)
                train_writer.add_scalar('loss_cls', model.loss_dict["cls"], model.total_steps)
                train_writer.add_scalar('loss_con', model.loss_dict["con"], model.total_steps)
                train_writer.add_scalar('loss_align', model.loss_dict["align"], model.total_steps)
                train_writer.add_scalar('loss_mid', model.loss_dict["mid"], model.total_steps)
                train_writer.add_scalar('loss_mask', model.loss_dict["mask"], model.total_steps)
                train_writer.add_scalar('loss_rec', model.loss_dict["rec"], model.total_steps)
                train_writer.add_scalar('loss_rank', model.loss_dict["rank"], model.total_steps)
                train_writer.add_scalar('loss_orth', model.loss_dict["orth"], model.total_steps)
                train_writer.add_scalar('loss_ksv', model.loss_dict["ksv"], model.total_steps)
                train_writer.add_scalar('loss_noise', model.loss_dict["noise"], model.total_steps)
                train_writer.add_scalar('loss_noise_align', model.loss_dict["noise_align"], model.total_steps)
                train_writer.add_scalar('loss_evidence', model.loss_dict.get("evidence", 0.0), model.total_steps)
                train_writer.add_scalar('loss_evidence_align', model.loss_dict.get("evidence_align", 0.0), model.total_steps)
                train_writer.add_scalar('loss_multiview_fusion', model.loss_dict.get("multiview_fusion", 0.0), model.total_steps)
                train_writer.add_scalar('loss_real_consistency', model.loss_dict.get("real_consistency", 0.0), model.total_steps)
                train_writer.add_scalar('loss_fake_disagreement', model.loss_dict.get("fake_disagreement", 0.0), model.total_steps)
                train_writer.add_scalar('loss_patch_mil', model.loss_dict.get("patch_mil", 0.0), model.total_steps)
                train_writer.add_scalar('loss_query_mil', model.loss_dict.get("query_mil", 0.0), model.total_steps)
                train_writer.add_scalar('loss_query_diversity', model.loss_dict.get("query_diversity", 0.0), model.total_steps)
                train_writer.add_scalar('loss_patch_disagreement', model.loss_dict.get("patch_disagreement", 0.0), model.total_steps)
                train_writer.add_scalar('loss_fire_error', model.loss_dict.get("fire_error", 0.0), model.total_steps)
                train_writer.add_scalar('loss_fire_error_align', model.loss_dict.get("fire_error_align", 0.0), model.total_steps)
                train_writer.add_scalar('loss_hos_evidence', model.loss_dict.get("hos_evidence", 0.0), model.total_steps)
                train_writer.add_scalar('loss_hos_boundary', model.loss_dict.get("hos_boundary", 0.0), model.total_steps)
                train_writer.add_scalar('loss_hos_rank', model.loss_dict.get("hos_rank", 0.0), model.total_steps)
                train_writer.add_scalar('loss_hos_tangent', model.loss_dict.get("hos_tangent", 0.0), model.total_steps)
                train_writer.add_scalar('loss_hos_cdc', model.loss_dict.get("hos_cdc", 0.0), model.total_steps)
                train_writer.add_scalar('loss_hos_anchor_residual', model.loss_dict.get("hos_anchor_residual", 0.0), model.total_steps)
                train_writer.add_scalar('hos_grad_norm', model.loss_dict.get("hos_grad_norm", 0.0), model.total_steps)
                train_writer.add_scalar('hos_score_net_grad_norm', model.loss_dict.get("score_net_grad_norm", 0.0), model.total_steps)
                train_writer.add_scalar('hos_mask_net_grad_norm', model.loss_dict.get("mask_net_grad_norm", 0.0), model.total_steps)
                train_writer.add_scalar('backbone_grad_norm', model.loss_dict.get("backbone_grad_norm", 0.0), model.total_steps)
                train_writer.add_scalar('loss_fake_margin', model.loss_dict.get("fake_margin", 0.0), model.total_steps)
                train_writer.add_scalar('loss_fake_hard', model.loss_dict.get("fake_hard", 0.0), model.total_steps)
                train_writer.add_scalar('loss_fake_hard_consistency', model.loss_dict.get("fake_hard_consistency", 0.0), model.total_steps)
                train_writer.add_scalar('loss_fake_hard_feat_consistency', model.loss_dict.get("fake_hard_feat_consistency", 0.0), model.total_steps)
                train_writer.add_scalar('loss_fake_hard_margin', model.loss_dict.get("fake_hard_margin", 0.0), model.total_steps)
                train_writer.add_scalar('loss_fake_adv', model.loss_dict.get("fake_adv", 0.0), model.total_steps)
                train_writer.add_scalar('loss_fake_adv_consistency', model.loss_dict.get("fake_adv_consistency", 0.0), model.total_steps)
                train_writer.add_scalar('loss_fake_adv_feat_consistency', model.loss_dict.get("fake_adv_feat_consistency", 0.0), model.total_steps)
                train_writer.add_scalar('loss_fake_adv_margin', model.loss_dict.get("fake_adv_margin", 0.0), model.total_steps)
                train_writer.add_scalar('fake_hard_active', model.loss_dict.get("fake_hard_active", 0.0), model.total_steps)
                train_writer.add_scalar('fake_adv_active', model.loss_dict.get("fake_adv_active", 0.0), model.total_steps)
                train_writer.add_scalar('fake_hard_count', model.loss_dict.get("fake_hard_count", 0.0), model.total_steps)
                train_writer.add_scalar('fake_adv_count', model.loss_dict.get("fake_adv_count", 0.0), model.total_steps)
                train_writer.add_scalar('loss_dual_res_cls', model.loss_dict["dual_res_cls"], model.total_steps)
                train_writer.add_scalar('loss_dual_res_align', model.loss_dict["dual_res_align"], model.total_steps)
                train_writer.add_scalar('d_real', model.loss_dict["d_real"], model.total_steps)
                train_writer.add_scalar('d_fake', model.loss_dict["d_fake"], model.total_steps)
                train_writer.add_scalar('d_gap', model.loss_dict["d_gap"], model.total_steps)
                train_writer.add_scalar('fire_active', model.loss_dict["fire_active"], model.total_steps)
                train_writer.add_scalar('noise_mask_density', model.loss_dict["noise_mask_density"], model.total_steps)
                train_writer.add_scalar('random_warmup', model.loss_dict["random_warmup"], model.total_steps)
                train_writer.add_scalar('train_res', model.loss_dict["train_res"], model.total_steps)
                print("Iter time: ", ((time.time()-start_time)/model.total_steps) )
                with open(log_file, 'a') as f:
                    f.write(
                        f"Iter time: {(time.time()-start_time)/model.total_steps}, "
                        f"Lr: {model.lr}, "
                        f"total: {model.loss_dict['total']}, cls: {model.loss_dict['cls']}, "
                        f"con: {model.loss_dict['con']}, align: {model.loss_dict['align']}, "
                        f"mid: {model.loss_dict['mid']}, mask: {model.loss_dict['mask']}, "
                        f"rec: {model.loss_dict['rec']}, rank: {model.loss_dict['rank']}, "
                        f"orth: {model.loss_dict['orth']}, ksv: {model.loss_dict['ksv']}, "
                        f"noise: {model.loss_dict['noise']}, noise_align: {model.loss_dict['noise_align']}, "
                        f"evidence: {model.loss_dict.get('evidence', 0.0)}, "
                        f"evidence_align: {model.loss_dict.get('evidence_align', 0.0)}, "
                        f"multiview_fusion: {model.loss_dict.get('multiview_fusion', 0.0)}, "
                        f"real_consistency: {model.loss_dict.get('real_consistency', 0.0)}, "
                        f"fake_disagreement: {model.loss_dict.get('fake_disagreement', 0.0)}, "
                        f"patch_mil: {model.loss_dict.get('patch_mil', 0.0)}, "
                        f"query_mil: {model.loss_dict.get('query_mil', 0.0)}, "
                        f"query_diversity: {model.loss_dict.get('query_diversity', 0.0)}, "
                        f"patch_disagreement: {model.loss_dict.get('patch_disagreement', 0.0)}, "
                        f"fire_error: {model.loss_dict.get('fire_error', 0.0)}, "
                        f"fire_error_align: {model.loss_dict.get('fire_error_align', 0.0)}, "
                        f"hos_evidence: {model.loss_dict.get('hos_evidence', 0.0)}, "
                        f"hos_boundary: {model.loss_dict.get('hos_boundary', 0.0)}, "
                        f"hos_rank: {model.loss_dict.get('hos_rank', 0.0)}, "
                        f"hos_tangent: {model.loss_dict.get('hos_tangent', 0.0)}, "
                        f"hos_cdc: {model.loss_dict.get('hos_cdc', 0.0)}, "
                        f"hos_anchor_residual: {model.loss_dict.get('hos_anchor_residual', 0.0)}, "
                        f"hos_grad_norm: {model.loss_dict.get('hos_grad_norm', 0.0)}, "
                        f"hos_score_grad: {model.loss_dict.get('score_net_grad_norm', 0.0)}, "
                        f"hos_mask_grad: {model.loss_dict.get('mask_net_grad_norm', 0.0)}, "
                        f"backbone_grad: {model.loss_dict.get('backbone_grad_norm', 0.0)}, "
                        f"fake_margin: {model.loss_dict.get('fake_margin', 0.0)}, "
                        f"fake_hard: {model.loss_dict.get('fake_hard', 0.0)}, "
                        f"fake_hard_cons: {model.loss_dict.get('fake_hard_consistency', 0.0)}, "
                        f"fake_hard_feat: {model.loss_dict.get('fake_hard_feat_consistency', 0.0)}, "
                        f"fake_hard_margin: {model.loss_dict.get('fake_hard_margin', 0.0)}, "
                        f"fake_adv: {model.loss_dict.get('fake_adv', 0.0)}, "
                        f"fake_adv_cons: {model.loss_dict.get('fake_adv_consistency', 0.0)}, "
                        f"fake_adv_feat: {model.loss_dict.get('fake_adv_feat_consistency', 0.0)}, "
                        f"fake_adv_margin: {model.loss_dict.get('fake_adv_margin', 0.0)}, "
                        f"fake_hard_active: {model.loss_dict.get('fake_hard_active', 0.0)}, "
                        f"fake_adv_active: {model.loss_dict.get('fake_adv_active', 0.0)}, "
                        f"fake_hard_count: {model.loss_dict.get('fake_hard_count', 0.0)}, "
                        f"fake_adv_count: {model.loss_dict.get('fake_adv_count', 0.0)}, "
                        f"dual_res_cls: {model.loss_dict['dual_res_cls']}, dual_res_align: {model.loss_dict['dual_res_align']}, "
                        f"d_real: {model.loss_dict['d_real']}, d_fake: {model.loss_dict['d_fake']}, "
                        f"d_gap: {model.loss_dict['d_gap']}, fire_active: {model.loss_dict['fire_active']}, "
                        f"noise_mask_density: {model.loss_dict['noise_mask_density']}, "
                        f"random_warmup: {model.loss_dict['random_warmup']}, "
                        f"train_res: {model.loss_dict['train_res']} "
                        f"at step: {model.total_steps}\n"
                    )

            if (
                main_process
                and not getattr(opt, "debug_no_save", False)
                and model.total_steps in [50,100,500,550,600,650,700,800,900,1000,1200,1500,2000,3000,5000,8000,10000,12000,18000,20000,23000,25000]
            ): # save models at these iters
                model.train()
                model.save_networks('model_iters_%s.pth' % model.total_steps)
            
            # if model.total_steps % 500 == 0:
            #     model.adjust_learning_rate()
        if main_process:
            epoch_pbar.close()

        stop_training = False
        if getattr(opt, "distributed", False):
            dist.barrier()

        if main_process and not getattr(opt, "debug_no_save", False) and epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d' % (epoch))
            model.train()
            model.save_networks('model_epoch_%s.pth' % epoch)

        if getattr(opt, "distributed", False):
            dist.barrier()

        model.eval()
        if getattr(opt, "infer_mode", "original") == "original":
            model_for_val = model.model.module if hasattr(model.model, "module") else model.model
        else:
            model_for_val = model

        val_metrics = validate(
            model_for_val,
            val_loader,
            distributed_collect=getattr(opt, "distributed", False),
        )
        if main_process:
            ap, r_acc, f_acc, acc = val_metrics
            val_writer.add_scalar('accuracy', acc, model.total_steps)
            val_writer.add_scalar('ap', ap, model.total_steps)
            print("(Val @ epoch {}) acc: {}; ap: {}".format(epoch, acc, ap))
            with open(log_file, "a") as f:
                f.write(
                    f"(Val @ epoch {epoch}) acc: {acc}; ap: {ap}; "
                    f"r_acc: {r_acc}; f_acc: {f_acc}\n"
                )

        if test_loader is not None and opt.test_freq > 0 and (epoch + 1) % opt.test_freq == 0:
            test_metrics = validate(
                model_for_val,
                test_loader,
                find_thres=True,
                threshold_metric="balanced_acc",
                distributed_collect=getattr(opt, "distributed", False),
            )
            if main_process:
                (
                    test_ap,
                    test_r_acc,
                    test_f_acc,
                    test_acc,
                    test_r_acc_best,
                    test_f_acc_best,
                    test_acc_best,
                    test_best_thres,
                ) = test_metrics
                test_writer.add_scalar('accuracy@0.5', test_acc, model.total_steps)
                test_writer.add_scalar('ap', test_ap, model.total_steps)
                test_writer.add_scalar('real_accuracy@0.5', test_r_acc, model.total_steps)
                test_writer.add_scalar('fake_accuracy@0.5', test_f_acc, model.total_steps)
                test_writer.add_scalar('accuracy@best_threshold', test_acc_best, model.total_steps)
                test_writer.add_scalar('real_accuracy@best_threshold', test_r_acc_best, model.total_steps)
                test_writer.add_scalar('fake_accuracy@best_threshold', test_f_acc_best, model.total_steps)
                test_writer.add_scalar('best_threshold', test_best_thres, model.total_steps)
                print(
                    "(Test @ epoch {}) ap: {}; "
                    "thr0.5 acc: {}; r_acc: {}; f_acc: {}; "
                    "best_thr: {}; best_acc: {}; best_r_acc: {}; best_f_acc: {}".format(
                        epoch,
                        test_ap,
                        test_acc,
                        test_r_acc,
                        test_f_acc,
                        test_best_thres,
                        test_acc_best,
                        test_r_acc_best,
                        test_f_acc_best,
                    )
                )
                with open(log_file, "a") as f:
                    f.write(
                        f"(Test @ epoch {epoch}) ap: {test_ap}; "
                        f"thr0.5_acc: {test_acc}; thr0.5_r_acc: {test_r_acc}; "
                        f"thr0.5_f_acc: {test_f_acc}; best_thr: {test_best_thres}; "
                        f"best_acc: {test_acc_best}; best_r_acc: {test_r_acc_best}; "
                        f"best_f_acc: {test_f_acc_best}\n"
                    )

        if main_process and not getattr(opt, "debug_no_save", False):
            early_stopping(acc, model)
            if early_stopping.early_stop:
                print("Early stopping triggered.")
                stop_training = True
        model.train()

        if getattr(opt, "distributed", False):
            stop_tensor = torch.tensor([1 if stop_training else 0], dtype=torch.int32, device=model.device)
            dist.broadcast(stop_tensor, src=0)
            stop_training = bool(stop_tensor.item())

        if stop_training:
            break

    if main_process:
        train_writer.close()
        val_writer.close()
        if test_writer is not None:
            test_writer.close()
    cleanup_distributed()
