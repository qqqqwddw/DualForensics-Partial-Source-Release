import argparse
import os
import util
import torch


class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument('--mode', default='binary')
        parser.add_argument('--arch', type=str, default='res50', help='see my_models/__init__.py')
        parser.add_argument('--fix_backbone', action='store_true') 
        parser.add_argument('--use_svd', action='store_true')
        parser.add_argument('--svd_low_rank_forward', action='store_true',
                            help='use low-rank forward for SVD residual linear without explicitly constructing full residual weight')
        parser.add_argument('--svd_residual_rank', type=int, default=1,
                            help='number of trainable SVD residual singular directions; 1 matches the original design')
        parser.add_argument('--svd_keep_rank_ratio', type=float, default=0.999,
                            help='fraction of singular components kept in frozen main weight; lower values increase trainable residual capacity')
        parser.add_argument('--dinov3_repo_dir', type=str, default='./dinov3-main/dinov3-main',
                            help='local DINOv3 repo directory that contains hubconf.py and dinov3/')
        parser.add_argument('--dinov3_weights', type=str, default='',
                            help='path/url to DINOv3 pretrained backbone weights (optional)')

        # data augmentation
        parser.add_argument('--rz_interp', default='bilinear')
        parser.add_argument('--blur_prob', type=float, default=0.5)
        parser.add_argument('--blur_sig', default='0.0,3.0')
        parser.add_argument('--jpg_prob', type=float, default=0.5)
        parser.add_argument('--jpg_method', default='cv2,pil')
        parser.add_argument('--jpg_qual', default='30,100')
        
         
        parser.add_argument('--real_list_path', default=None, help='only used if data_mode==ours: path for the list of real images, which should contain train.pickle and val.pickle')
        parser.add_argument('--fake_list_path', default=None, help='only used if data_mode==ours: path for the list of fake images, which should contain train.pickle and val.pickle')
        parser.add_argument('--wang2020_data_path', default=None, help='only used if data_mode==wang2020 it should contain train and test folders')
        parser.add_argument('--data_mode',  default='ours', help='wang2020 or ours')
        parser.add_argument('--data_label', default='train', help='label to decide whether train or validation dataset')
        parser.add_argument('--weight_decay', type=float, default=0.0, help='loss weight for l2 reg')
        
        parser.add_argument('--class_bal', action='store_true') # what is this ?
        parser.add_argument('--batch_size', type=int, default=256, help='input batch size')
        parser.add_argument('--loadSize', type=int, default=256, help='scale images to this size')
        parser.add_argument('--cropSize', type=int, default=224, help='then crop to this size')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--num_threads', default=4, type=int, help='# threads for loading data')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--resize_or_crop', type=str, default='scale_and_crop', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop|none]')
        parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
        parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal|xavier|kaiming|orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{loadSize}')
        parser.add_argument('--infer_mode', type=str, default='original',
                            choices=[
                                'original',
                                'original_fixed_avg',
                                'original_fixed_random_avg',
                                'original_fixed_random_max',
                                'uncertain_multiview',
                                'trained_multiview_fusion',
                                'trained_evidence_avg',
                                'trained_multiview_evidence_avg',
                                'trained_patch_mil',
                                'trained_query_mil',
                                'trained_delta_patch_mil_avg',
                                'trained_forensic_fusion',
                                'trained_hos_fire',
                                'trained_forensic_hos_fusion',
                                'trained_hos_selective_or',
                                'trained_real_manifold',
                                'trained_hos_manifold_logit_fusion',
                                'trained_forensic_hos_logit_fusion',
                            ],
                            help='inference branch selection; default keeps original-only inference')
        parser.add_argument('--infer_random_views', type=int, default=2,
                            help='number of random mapping views used by multiview inference')
        parser.add_argument('--infer_uncertain_delta', type=float, default=0.12,
                            help='use multiview fusion only when |p_original - 0.5| is below this value')
        parser.add_argument('--infer_multiview_alpha', type=float, default=0.5,
                            help='blend strength for uncertain multiview inference')
        parser.add_argument('--hos_logit_scale', type=float, default=1.0,
                            help='affine calibration scale for HOS logit evidence during inference')
        parser.add_argument('--hos_logit_bias', type=float, default=0.0,
                            help='affine calibration bias for HOS logit evidence during inference')
        parser.add_argument('--logit_fusion_weight_original', type=float, default=1.0)
        parser.add_argument('--logit_fusion_weight_hos', type=float, default=1.0)
        parser.add_argument('--logit_fusion_weight_patch', type=float, default=0.0)
        parser.add_argument('--logit_fusion_weight_evidence', type=float, default=0.0)
        parser.add_argument('--logit_fusion_bias', type=float, default=0.0)
        self.initialized = True
        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()
        self.parser = parser

        return parser.parse_args()

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self, print_options=True):

        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test

        # process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)

        # torchrun-based distributed setup (env://)
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        opt.distributed = world_size > 1
        opt.world_size = world_size
        opt.rank = rank
        opt.local_rank = local_rank

        if opt.distributed:
            if len(opt.gpu_ids) == 0:
                raise ValueError("Distributed training requires CUDA GPUs. Please set --gpu_ids.")
            if local_rank < len(opt.gpu_ids):
                selected_gpu = opt.gpu_ids[local_rank]
            else:
                selected_gpu = local_rank
            opt.gpu_ids = [selected_gpu]
            torch.cuda.set_device(selected_gpu)
        elif len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        if print_options and (not opt.distributed or opt.rank == 0):
            self.print_options(opt)

        # additional
        #opt.classes = opt.classes.split(',')
        opt.rz_interp = opt.rz_interp.split(',')
        opt.blur_sig = [float(s) for s in opt.blur_sig.split(',')]
        opt.jpg_method = opt.jpg_method.split(',')
        opt.jpg_qual = [int(s) for s in opt.jpg_qual.split(',')]
        if len(opt.jpg_qual) == 2:
            opt.jpg_qual = list(range(opt.jpg_qual[0], opt.jpg_qual[1] + 1))
        elif len(opt.jpg_qual) > 2:
            raise ValueError("Shouldn't have more than 2 values for --jpg_qual.")

        self.opt = opt
        return self.opt
