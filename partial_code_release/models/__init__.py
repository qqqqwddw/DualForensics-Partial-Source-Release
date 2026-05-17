import os

from .clip_models import ClipModel
from .dinov3_models import DinoV3Model


# Public fallback model IDs. Set environment variables below to use local checkpoints.
HF_FALLBACKS = {
    'CLIP:ViT-B/16_svd': 'openai/clip-vit-base-patch16',
    'CLIP:ViT-B/32_svd': 'openai/clip-vit-base-patch32',
    'CLIP:ViT-L/14_svd': 'openai/clip-vit-large-patch14',
    'SigLIP:ViT-L/16_256_svd': 'google/siglip-large-patch16-256',
}

LOCAL_MODEL_ENV = {
    'CLIP:ViT-B/16_svd': 'DUALFORENSICS_CLIP_VIT_B16_PATH',
    'CLIP:ViT-B/32_svd': 'DUALFORENSICS_CLIP_VIT_B32_PATH',
    'CLIP:ViT-L/14_svd': 'DUALFORENSICS_CLIP_VIT_L14_PATH',
    'SigLIP:ViT-L/16_256_svd': 'DUALFORENSICS_SIGLIP_L16_256_PATH',
    'BEiTv2:ViT-L/16_svd': 'DUALFORENSICS_BEITV2_L16_PATH',
}

CLIP_VALID_NAMES = {name: None for name in LOCAL_MODEL_ENV.keys()}

DINO_VALID_NAMES = {
    'DINOv3:ViT-S/16_svd': None,
    'DINOv3:ViT-B/16_svd': None,
    'DINOv3:ViT-L/16_svd': None,
    'DINOv3:ViT-L/16plus_svd': None,
    'DINOv3:ViT-H/16plus_svd': None,
    'DINOv3:ViT-7B/16_svd': None,
}

VALID_NAMES = {**CLIP_VALID_NAMES, **DINO_VALID_NAMES}


def _resolve_model_path(name):
    env_name = LOCAL_MODEL_ENV.get(name)
    if env_name:
        local_path = os.environ.get(env_name)
        if local_path:
            return local_path
    return HF_FALLBACKS.get(name)


def get_model(name, opt):
    if name not in VALID_NAMES:
        raise ValueError(f'Unsupported architecture: {name}. Valid names: {list(VALID_NAMES.keys())}')
    if name.startswith('CLIP:') or name.startswith('SigLIP:') or name.startswith('BEiTv2:'):
        return ClipModel(_resolve_model_path(name), opt)
    if name.startswith('DINOv3:'):
        return DinoV3Model(name, opt)
    raise ValueError(f'Unsupported architecture prefix in: {name}')
