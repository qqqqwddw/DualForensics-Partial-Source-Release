import os
import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler


class BaseModel(nn.Module):
    def __init__(self, opt):
        super(BaseModel, self).__init__()
        self.opt = opt
        self.total_steps = 0
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')

    def save_networks(self, save_filename):
        save_path = os.path.join(self.save_dir, save_filename)
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model

        # serialize model and optimizer to dict
        state_dict = {
            'model': model_to_save.state_dict(),
            'optimizer' : self.optimizer.state_dict(),
            'total_steps' : self.total_steps,
            'config': vars(self.opt) if hasattr(self, 'opt') else {},
        }
        if hasattr(self, 'projector'):
            state_dict['projector'] = self.projector.state_dict()
        if hasattr(self, 'mid_prior_predictor') and self.mid_prior_predictor is not None:
            state_dict['mid_prior_predictor'] = self.mid_prior_predictor.state_dict()
        if hasattr(self, 'evidence_head') and self.evidence_head is not None:
            state_dict['evidence_head'] = self.evidence_head.state_dict()
        if hasattr(self, 'multiview_fusion_head') and self.multiview_fusion_head is not None:
            state_dict['multiview_fusion_head'] = self.multiview_fusion_head.state_dict()
        if hasattr(self, 'patch_mil_head') and self.patch_mil_head is not None:
            state_dict['patch_mil_head'] = self.patch_mil_head.state_dict()
        if hasattr(self, 'query_mil_head') and self.query_mil_head is not None:
            state_dict['query_mil_head'] = self.query_mil_head.state_dict()
        if hasattr(self, 'patch_disagreement_head') and self.patch_disagreement_head is not None:
            state_dict['patch_disagreement_head'] = self.patch_disagreement_head.state_dict()
        if hasattr(self, 'fire_error_head') and self.fire_error_head is not None:
            state_dict['fire_error_head'] = self.fire_error_head.state_dict()
        if hasattr(self, 'hos_fire_envelope') and self.hos_fire_envelope is not None:
            state_dict['hos_fire_envelope'] = self.hos_fire_envelope.state_dict()
        if hasattr(self, 'mid_band_mask_head') and self.mid_band_mask_head is not None:
            state_dict['mid_band_mask_head'] = self.mid_band_mask_head.state_dict()
        if hasattr(self, 'recon_score_head') and self.recon_score_head is not None:
            state_dict['recon_score_head'] = self.recon_score_head.state_dict()
        if hasattr(self, 'noise_guidance') and self.noise_guidance is not None:
            state_dict['noise_guidance'] = self.noise_guidance.state_dict()
        if hasattr(self, 'noise_guidance_head') and self.noise_guidance_head is not None:
            state_dict['noise_guidance_head'] = self.noise_guidance_head.state_dict()

        torch.save(state_dict, save_path)


    def eval(self):
        return nn.Module.eval(self)

    def test(self):
        with torch.no_grad():
            self.forward()


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)
