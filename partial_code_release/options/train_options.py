from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--earlystop_epoch', type=int, default=5)
        parser.add_argument('--data_aug', action='store_true', help='if specified, perform additional data augmentation (photometric, blurring, jpegging)')
        parser.add_argument('--color_aug_prob', type=float, default=0.5)
        parser.add_argument('--brightness_range', type=str, default='0.85,1.15')
        parser.add_argument('--contrast_range', type=str, default='0.85,1.15')
        parser.add_argument('--saturation_range', type=str, default='0.85,1.15')
        parser.add_argument('--resize_aug_prob', type=float, default=0.25)
        parser.add_argument('--resize_aug_scale', type=str, default='0.5,1.0')
        parser.add_argument('--use_social_chain_aug', action='store_true',
                            help='apply a label-aware social-chain augmentation before the standard train augmentations')
        parser.add_argument('--social_chain_prob', type=float, default=1.0,
                            help='probability of applying the shared social-chain augmentation')
        parser.add_argument('--social_chain_min_ops', type=int, default=2)
        parser.add_argument('--social_chain_max_ops', type=int, default=4)
        parser.add_argument('--use_fake_dataset_hardening', action='store_true',
                            help='apply an extra label-preserving hardening chain to fake samples during data loading')
        parser.add_argument('--fake_dataset_hard_prob', type=float, default=0.35,
                            help='probability of applying the fake-only dataset hardening chain')
        parser.add_argument('--fake_social_chain_min_ops', type=int, default=3)
        parser.add_argument('--fake_social_chain_max_ops', type=int, default=5)
        parser.add_argument('--fake_dataset_use_mid_suppress', dest='fake_dataset_use_mid_suppress', action='store_true',
                            help='include a mid-frequency suppression step in the fake-only dataset hardening chain')
        parser.add_argument('--no_fake_dataset_use_mid_suppress', dest='fake_dataset_use_mid_suppress', action='store_false')
        parser.set_defaults(fake_dataset_use_mid_suppress=True)
        parser.add_argument('--train_max_per_class', type=int, default=-1,
                            help='debug/ablation only: limit training real and fake samples per class when > 0')
        parser.add_argument('--eval_max_per_class', type=int, default=-1,
                            help='debug/ablation only: limit validation/test real and fake samples per class when > 0')
        parser.add_argument('--hard_fake_list', type=str, default='',
                            help='optional txt/csv/pickle list of hard fake paths to replay during training')
        parser.add_argument('--hard_real_list', type=str, default='',
                            help='optional txt/csv/pickle list of hard real paths to replay during training')
        parser.add_argument('--hard_fake_repeats', type=int, default=1,
                            help='number of times to append hard fake paths to the training list')
        parser.add_argument('--hard_real_repeats', type=int, default=1,
                            help='number of times to append hard real paths to the training list')
        parser.add_argument('--debug_no_save', action='store_true',
                            help='debug/ablation only: skip checkpoint writes and early-stopping best saves')
        parser.add_argument('--optim', type=str, default='adam', help='optim to use [sgd, adam, adamw]; adam and adamw both use AdamW')
        parser.add_argument('--new_optim', action='store_true', help='new optimizer instead of loading the optim state')
        parser.add_argument('--resume_ckpt', type=str, default='', help='checkpoint path to resume training from')
        parser.add_argument('--resume_epoch', type=int, default=-1,
                            help='completed epoch index in resume_ckpt; -1 tries to infer from model_epoch_N.pth')
        parser.add_argument('--resume_allow_partial_model', action='store_true',
                            help='skip model checkpoint tensors whose shapes do not match the current architecture')
        parser.add_argument('--loss_freq', type=int, default=500, help='frequency of showing loss on tensorboard')
        parser.add_argument('--save_epoch_freq', type=int, default=1, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--last_epoch', type=int, default=-1, help='starting epoch count for scheduler intialization')
        parser.add_argument('--train_split', type=str, default='train', help='train, val, test, etc')
        parser.add_argument('--val_split', type=str, default='val', help='train, val, test, etc')
        parser.add_argument('--test_split', type=str, default='test', help='test split name')
        parser.add_argument('--test_freq', type=int, default=0, help='run test every N epochs; 0 disables in-training test')
        parser.add_argument('--eval_batch_size', type=int, default=0,
                            help='batch size for validation/test; 0 reuses --batch_size')
        parser.add_argument('--ddp_timeout_minutes', type=int, default=180,
                            help='DDP/NCCL collective timeout; increase when rank-0 validation/test is long')
        parser.add_argument('--use_domain_balanced_sampler', action='store_true',
                            help='sample images with weights balanced by real/fake label and path-derived domain')
        parser.add_argument('--niter', type=int, default=100, help='total epoches')
        parser.add_argument('--beta1', type=float, default=0.9, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
        parser.add_argument('--lr_warmup_steps', type=int, default=0,
                            help='linearly warm up learning rate to --lr over this many optimizer steps; 0 disables')
        parser.add_argument('--lr_warmup_start_factor', type=float, default=0.1,
                            help='warmup starts from lr * this factor')
        parser.add_argument('--lr_warmup_reset_on_resume', action='store_true',
                            help='when resuming, start warmup from the checkpoint total_steps instead of global step 0')

        # three-branch training and pixel mapping
        parser.add_argument('--use_three_branch_training', action='store_true')

        parser.add_argument('--use_pixel_mapping', dest='use_pixel_mapping', action='store_true')
        parser.add_argument('--no_use_pixel_mapping', dest='use_pixel_mapping', action='store_false')
        parser.set_defaults(use_pixel_mapping=True)

        parser.add_argument('--use_fixed_mapping_branch', dest='use_fixed_mapping_branch', action='store_true')
        parser.add_argument('--no_use_fixed_mapping_branch', dest='use_fixed_mapping_branch', action='store_false')
        parser.set_defaults(use_fixed_mapping_branch=True)

        parser.add_argument('--use_random_mapping_branch', dest='use_random_mapping_branch', action='store_true')
        parser.add_argument('--no_use_random_mapping_branch', dest='use_random_mapping_branch', action='store_false')
        parser.set_defaults(use_random_mapping_branch=True)

        parser.add_argument('--pixel_mapping_remap_to_unit', dest='pixel_mapping_remap_to_unit', action='store_true')
        parser.add_argument('--no_pixel_mapping_remap_to_unit', dest='pixel_mapping_remap_to_unit', action='store_false')
        parser.set_defaults(pixel_mapping_remap_to_unit=True)
        parser.add_argument('--pixel_mapping_random_range', type=float, default=1.0)
        parser.add_argument('--pixel_mapping_deterministic', action='store_true')
        parser.add_argument('--pixel_mapping_seed', type=int, default=42)

        # projector and consistency losses
        parser.add_argument('--proj_dim', type=int, default=256)
        parser.add_argument('--contrastive_temperature', type=float, default=0.1)
        parser.add_argument('--lambda_con', type=float, default=0.1)
        parser.add_argument('--lambda_align', type=float, default=0.05)

        # branch classification weights
        parser.add_argument('--branch_weight_original', type=float, default=1.0)
        parser.add_argument('--branch_weight_fixed', type=float, default=1.0)
        parser.add_argument('--branch_weight_random', type=float, default=0.5)

        # trainable inference-oriented fusion/evidence heads
        parser.add_argument('--use_multiview_fusion_head', action='store_true',
                            help='train a head over original/fixed/random branch features and logits')
        parser.add_argument('--lambda_multiview_fusion', type=float, default=1.0)
        parser.add_argument('--multiview_fusion_hidden_dim', type=int, default=256)
        parser.add_argument('--multiview_fusion_type', type=str, default='concat',
                            choices=['concat', 'delta'],
                            help='concat keeps old behavior; delta uses explicit cross-view feature/logit deltas')
        parser.add_argument('--use_evidence_head', action='store_true',
                            help='train an auxiliary forensic evidence head from patch/noise/mid statistics')
        parser.add_argument('--lambda_evidence', type=float, default=0.5)
        parser.add_argument('--lambda_evidence_align', type=float, default=0.05)
        parser.add_argument('--evidence_hidden_dim', type=int, default=256)

        # forensic architecture ablations
        parser.add_argument('--lambda_real_consistency', type=float, default=0.0,
                            help='real-only multiview consistency loss; keeps real images stable under mappings')
        parser.add_argument('--lambda_fake_disagreement', type=float, default=0.0,
                            help='fake-only multiview disagreement margin loss')
        parser.add_argument('--fake_disagreement_margin', type=float, default=0.15,
                            help='target minimum mean absolute logit disagreement for fake samples')
        parser.add_argument('--use_patch_mil_head', action='store_true',
                            help='train LOGER-style top-k patch MIL head on original-view tokens')
        parser.add_argument('--lambda_patch_mil', type=float, default=0.5)
        parser.add_argument('--patch_mil_hidden_dim', type=int, default=256)
        parser.add_argument('--patch_mil_topk_ratio', type=float, default=0.10)
        parser.add_argument('--use_query_mil_head', action='store_true',
                            help='train SIDA-inspired learnable-query MIL head on original-view patch tokens')
        parser.add_argument('--lambda_query_mil', type=float, default=0.5)
        parser.add_argument('--lambda_query_diversity', type=float, default=0.001)
        parser.add_argument('--query_mil_hidden_dim', type=int, default=256)
        parser.add_argument('--query_mil_num_queries', type=int, default=4)
        parser.add_argument('--query_mil_num_heads', type=int, default=8)
        parser.add_argument('--query_mil_dropout', type=float, default=0.0)
        parser.add_argument('--query_mil_topk_ratio', type=float, default=0.50)
        parser.add_argument('--use_cross_view_patch_disagreement', action='store_true',
                            help='train top-k patch disagreement head over original/fixed/random tokens')
        parser.add_argument('--lambda_patch_disagreement', type=float, default=0.5)
        parser.add_argument('--patch_disagreement_hidden_dim', type=int, default=256)
        parser.add_argument('--patch_disagreement_topk_ratio', type=float, default=0.10)
        parser.add_argument('--use_fire_error_evidence', action='store_true',
                            help='train lightweight FIRE-style mid-band feature error evidence head')
        parser.add_argument('--lambda_fire_error', type=float, default=0.3)
        parser.add_argument('--lambda_fire_error_align', type=float, default=0.02)
        parser.add_argument('--fire_error_hidden_dim', type=int, default=64)
        parser.add_argument('--fire_error_topk_ratio', type=float, default=0.10)

        # HOS-FIRE / real-centric envelope branch
        parser.add_argument('--use_hos_fire_envelope', action='store_true',
                            help='train HOS-guided FIRE residual evidence with REM-style real-envelope losses')
        parser.add_argument('--train_hos_only', action='store_true',
                            help='freeze all non-HOS modules and optimize only the HOS-FIRE envelope heads')
        parser.add_argument('--lambda_hos_evidence', type=float, default=0.3)
        parser.add_argument('--lambda_hos_mask', type=float, default=0.02)
        parser.add_argument('--lambda_hos_target', type=float, default=0.05)
        parser.add_argument('--lambda_hos_boundary', type=float, default=0.0)
        parser.add_argument('--lambda_hos_rank', type=float, default=0.0)
        parser.add_argument('--lambda_hos_tangent', type=float, default=0.0)
        parser.add_argument('--lambda_hos_cdc', type=float, default=0.0)
        parser.add_argument('--lambda_hos_anchor', type=float, default=0.0)
        parser.add_argument('--lambda_hos_anchor_residual', type=float, default=0.0)
        parser.add_argument('--hos_mask_hidden_dim', type=int, default=64)
        parser.add_argument('--hos_score_hidden_dim', type=int, default=64)
        parser.add_argument('--hos_topk_ratio', type=float, default=0.10)
        parser.add_argument('--hos_num_phase_shifts', type=int, default=6)
        parser.add_argument('--hos_boundary_phase_eps', type=float, default=0.12)
        parser.add_argument('--hos_boundary_margin', type=float, default=0.25)
        parser.add_argument('--hos_rank_margin', type=float, default=0.25)
        parser.add_argument('--hos_evidence_fake_weight', type=float, default=1.0)
        parser.add_argument('--hos_boundary_cls_weight', type=float, default=0.0,
                            help='optional auxiliary weight for also training the main classifier on HOS boundary samples')
        parser.add_argument('--hos_tangent_rank', type=int, default=8)
        parser.add_argument('--hos_detach_aux', action='store_true',
                            help='detach DINO aux maps in HOS extra forwards to reduce peak memory; trains HOS envelope heads without HOS backprop into backbone')
        parser.add_argument('--hos_cdc_degrade_prob', type=float, default=1.0)
        parser.add_argument('--hos_cdc_noise_std', type=float, default=0.015)
        parser.add_argument('--hos_cdc_min_scale', type=float, default=0.50)
        parser.add_argument('--hos_forward_interval', type=int, default=1,
                            help='run HOS-FIRE branch every N optimizer steps; 1 means every step')

        # decision-boundary losses
        parser.add_argument('--use_focal_loss', action='store_true')
        parser.add_argument('--focal_gamma', type=float, default=2.0)
        parser.add_argument('--fake_loss_weight', type=float, default=1.0)
        parser.add_argument('--lambda_fake_margin', type=float, default=0.0)
        parser.add_argument('--fake_margin', type=float, default=1.0)

        # random-branch warmup
        parser.add_argument('--random_branch_warmup_epochs', type=int, default=5)

        # effort regularization weights (kept explicit for reproducibility)
        parser.add_argument('--lambda_orth', type=float, default=0.0)
        parser.add_argument('--lambda_ksv', type=float, default=0.0)

        # training-time mid-frequency auxiliary prior
        parser.add_argument('--use_mid_frequency_prior', action='store_true')
        parser.add_argument('--mid_prior_scope', type=str, default='original',
                            choices=['original', 'original_fixed', 'all'])
        parser.add_argument('--lambda_mid', type=float, default=0.05)
        parser.add_argument('--mid_prior_loss_type', type=str, default='mse',
                            choices=['mse', 'l1', 'smooth_l1'])

        parser.add_argument('--mid_prior_use_grayscale', dest='mid_prior_use_grayscale', action='store_true')
        parser.add_argument('--no_mid_prior_use_grayscale', dest='mid_prior_use_grayscale', action='store_false')
        parser.set_defaults(mid_prior_use_grayscale=True)

        parser.add_argument('--mid_prior_predictor_hidden_dim', type=int, default=128)
        parser.add_argument('--mid_prior_predictor_depth', type=int, default=2)
        parser.add_argument('--mid_prior_out_channels', type=int, default=6,
                            help='multi-band forensic prior channels. 1 keeps the old single-ring target')
        parser.add_argument('--lambda_mid_consistency', type=float, default=0.02,
                            help='consistency between mid-prior predictions for clean and chain-degraded views')
        parser.add_argument('--mid_prior_from_layer', type=str, default='last')
        parser.add_argument('--mid_freq_radius_low_ratio', type=float, default=0.15)
        parser.add_argument('--mid_freq_radius_high_ratio', type=float, default=0.45)

        parser.add_argument('--mid_freq_use_absolute_response', dest='mid_freq_use_absolute_response', action='store_true')
        parser.add_argument('--no_mid_freq_use_absolute_response', dest='mid_freq_use_absolute_response', action='store_false')
        parser.set_defaults(mid_freq_use_absolute_response=True)

        # FIRE-Lite: frequency-guided supervision with pseudo branch
        parser.add_argument('--use_fire_lite', action='store_true')
        parser.add_argument('--lambda_mask', type=float, default=0.02)
        parser.add_argument('--lambda_rec', type=float, default=0.05)
        parser.add_argument('--lambda_rank', type=float, default=0.05)
        parser.add_argument('--fire_rank_margin', type=float, default=0.15)
        parser.add_argument('--fire_mask_hidden_dim', type=int, default=128)
        parser.add_argument('--fire_recon_hidden_dim', type=int, default=64)
        parser.add_argument('--fire_pseudo_interval', type=int, default=1)

        # noise-guided attention prior (guidance-only; no pixel-level supervision required)
        parser.add_argument('--use_noise_guidance', action='store_true')
        parser.add_argument('--noise_guidance_scales', type=str, default='1,2,4,8')
        parser.add_argument('--noise_guidance_embed_dim', type=int, default=128)
        parser.add_argument('--noise_guidance_head_hidden_dim', type=int, default=256)
        parser.add_argument('--noise_guidance_head_dropout', type=float, default=0.10)
        parser.add_argument('--noise_guidance_topk_ratio', type=float, default=0.25)
        parser.add_argument('--noise_guidance_max_tokens', type=int, default=4096)

        parser.add_argument('--noise_guidance_detach_mask', dest='noise_guidance_detach_mask', action='store_true')
        parser.add_argument('--no_noise_guidance_detach_mask', dest='noise_guidance_detach_mask', action='store_false')
        parser.set_defaults(noise_guidance_detach_mask=True)

        parser.add_argument('--noise_guidance_use_grayscale', dest='noise_guidance_use_grayscale', action='store_true')
        parser.add_argument('--no_noise_guidance_use_grayscale', dest='noise_guidance_use_grayscale', action='store_false')
        parser.set_defaults(noise_guidance_use_grayscale=True)

        parser.add_argument('--lambda_noise', type=float, default=0.05)
        parser.add_argument('--lambda_noise_align', type=float, default=0.02)

        # label-preserving fake hardening and detector-aware adversarial hardening
        parser.add_argument('--use_fake_hardening', action='store_true',
                            help='train on extra fake-only hard views with paired consistency regularization')
        parser.add_argument('--fake_hard_start_epoch', type=int, default=2)
        parser.add_argument('--fake_hard_prob', type=float, default=0.35)
        parser.add_argument('--fake_hard_max_batch', type=int, default=32)
        parser.add_argument('--lambda_fake_hard', type=float, default=0.30)
        parser.add_argument('--lambda_fake_hard_consistency', type=float, default=0.05)
        parser.add_argument('--lambda_fake_hard_feat_consistency', type=float, default=0.05)
        parser.add_argument('--lambda_fake_hard_margin', type=float, default=0.10)
        parser.add_argument('--fake_hard_margin', type=float, default=1.0)
        parser.add_argument('--fake_hard_use_hos', dest='fake_hard_use_hos', action='store_true')
        parser.add_argument('--no_fake_hard_use_hos', dest='fake_hard_use_hos', action='store_false')
        parser.set_defaults(fake_hard_use_hos=True)
        parser.add_argument('--fake_hard_hos_weight', type=float, default=1.0)

        parser.add_argument('--use_fake_adv_hardening', action='store_true',
                            help='generate detector-aware adversarial hard-fake views for fake samples during training')
        parser.add_argument('--fake_adv_start_epoch', type=int, default=3)
        parser.add_argument('--fake_adv_prob', type=float, default=0.25)
        parser.add_argument('--fake_adv_max_batch', type=int, default=8)
        parser.add_argument('--fake_adv_eps', type=float, default=4.0,
                            help='L-inf epsilon in 8-bit pixels; values > 1 are divided by 255')
        parser.add_argument('--fake_adv_alpha', type=float, default=2.0,
                            help='PGD step size in 8-bit pixels; values > 1 are divided by 255')
        parser.add_argument('--fake_adv_steps', type=int, default=2)
        parser.add_argument('--fake_adv_eot_views', type=int, default=1)
        parser.add_argument('--fake_adv_random_start', dest='fake_adv_random_start', action='store_true')
        parser.add_argument('--no_fake_adv_random_start', dest='fake_adv_random_start', action='store_false')
        parser.set_defaults(fake_adv_random_start=True)
        parser.add_argument('--fake_adv_on_social_chain', dest='fake_adv_on_social_chain', action='store_true')
        parser.add_argument('--no_fake_adv_on_social_chain', dest='fake_adv_on_social_chain', action='store_false')
        parser.set_defaults(fake_adv_on_social_chain=True)
        parser.add_argument('--fake_adv_attack_use_main', dest='fake_adv_attack_use_main', action='store_true')
        parser.add_argument('--no_fake_adv_attack_use_main', dest='fake_adv_attack_use_main', action='store_false')
        parser.set_defaults(fake_adv_attack_use_main=True)
        parser.add_argument('--fake_adv_attack_use_hos', dest='fake_adv_attack_use_hos', action='store_true')
        parser.add_argument('--no_fake_adv_attack_use_hos', dest='fake_adv_attack_use_hos', action='store_false')
        parser.set_defaults(fake_adv_attack_use_hos=True)
        parser.add_argument('--fake_adv_train_use_hos', dest='fake_adv_train_use_hos', action='store_true')
        parser.add_argument('--no_fake_adv_train_use_hos', dest='fake_adv_train_use_hos', action='store_false')
        parser.set_defaults(fake_adv_train_use_hos=True)
        parser.add_argument('--lambda_fake_adv', type=float, default=0.20)
        parser.add_argument('--lambda_fake_adv_consistency', type=float, default=0.05)
        parser.add_argument('--lambda_fake_adv_feat_consistency', type=float, default=0.05)
        parser.add_argument('--lambda_fake_adv_margin', type=float, default=0.10)
        parser.add_argument('--fake_adv_margin', type=float, default=1.0)

        # training-time multi-resolution augmentation (single-model, shared across DDP ranks)
        parser.add_argument('--use_multi_resolution_training', action='store_true')
        parser.add_argument('--multi_res_sizes', type=str, default='224,256,288')

        # LOGER-inspired dual-resolution consistency (same model, shared weights)
        parser.add_argument('--use_dual_resolution_consistency', action='store_true')
        parser.add_argument('--dual_res_high_size', type=int, default=384)
        parser.add_argument('--lambda_dual_res_cls', type=float, default=0.3)
        parser.add_argument('--lambda_dual_res_align', type=float, default=0.05)

        # mixed precision training
        parser.add_argument('--use_amp', action='store_true')
        parser.add_argument('--amp_dtype', type=str, default='fp16', choices=['fp16', 'bf16'])

        # automated forensic visualizations
        parser.add_argument('--auto_visualize', action='store_true',
                            help='automatically generate paper-style forensic visualizations during training')
        parser.add_argument('--auto_visualize_out_dir', type=str, default='',
                            help='output root. default: <checkpoints_dir>/<name>/visualizations')
        parser.add_argument('--auto_visualize_script', type=str, default='script/visualize_forensic_story.py')
        parser.add_argument('--auto_visualize_max_images', type=int, default=6,
                            help='number of fixed validation samples used by score-stability visualizations')
        parser.add_argument('--auto_visualize_device', type=str, default='cpu', choices=['cpu', 'cuda'],
                            help='device used by visualization subprocess; cpu avoids training GPU OOM')
        parser.add_argument('--auto_visualize_input_size', type=int, default=0,
                            help='visualization input size. 0 means use cropSize')
        parser.add_argument('--auto_visualize_heatmap_grid', type=int, default=10,
                            help='occlusion grid. larger is sharper but slower')
        parser.add_argument('--auto_visualize_timeout_sec', type=int, default=0,
                            help='subprocess timeout. 0 disables timeout')
        parser.add_argument('--auto_visualize_baseline_ckpt', type=str, default='',
                            help='optional baseline checkpoint for stability/heatmap comparison')
        parser.add_argument('--auto_visualize_ablation_csv', type=str, default='',
                            help='optional ablation csv for final bar chart')
        parser.add_argument('--auto_visualize_every_n_epochs', type=int, default=0,
                            help='also generate model-dependent snapshots every N epochs; 0 disables periodic snapshots')
        parser.add_argument('--auto_visualize_progress_max_nodes', type=int, default=8,
                            help='max rows in 10_progressive_summary.png')
        parser.add_argument('--auto_visualize_include_tsne', action='store_true',
                            help='enable t-SNE feature visualization when real/fake paths are available')
        parser.add_argument('--auto_visualize_on_start', dest='auto_visualize_on_start', action='store_true')
        parser.add_argument('--no_auto_visualize_on_start', dest='auto_visualize_on_start', action='store_false')
        parser.set_defaults(auto_visualize_on_start=True)
        parser.add_argument('--auto_visualize_on_best', dest='auto_visualize_on_best', action='store_true')
        parser.add_argument('--no_auto_visualize_on_best', dest='auto_visualize_on_best', action='store_false')
        parser.set_defaults(auto_visualize_on_best=True)
        parser.add_argument('--auto_visualize_on_final', dest='auto_visualize_on_final', action='store_true')
        parser.add_argument('--no_auto_visualize_on_final', dest='auto_visualize_on_final', action='store_false')
        parser.set_defaults(auto_visualize_on_final=True)

        self.isTrain = True
        return parser

