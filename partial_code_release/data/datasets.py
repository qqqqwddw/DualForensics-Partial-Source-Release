import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from random import random, choice, shuffle, randint
from io import BytesIO
from PIL import Image
from PIL import ImageEnhance
from PIL import ImageFile
from scipy.ndimage.filters import gaussian_filter
import pickle
import os
import math
from skimage.io import imread
from copy import deepcopy
import torch

ImageFile.LOAD_TRUNCATED_IMAGES = True


MEAN = {
    "imagenet":[0.485, 0.456, 0.406],
    "clip":[0.48145466, 0.4578275, 0.40821073],
    "beitv2": [0.485, 0.456, 0.406],
    "siglip": [0.5, 0.5, 0.5],
    "dinov3": [0.485, 0.456, 0.406],
}

STD = {
    "imagenet":[0.229, 0.224, 0.225],
    "clip":[0.26862954, 0.26130258, 0.27577711],
    "beitv2": [0.229, 0.224, 0.225],
    "siglip": [0.5, 0.5, 0.5],
    "dinov3": [0.229, 0.224, 0.225],
}


def translate_duplicate(img, cropSize):
    if min(img.size) < cropSize:
        width, height = img.size
        
        new_width = width * math.ceil(cropSize/width)
        new_height = height * math.ceil(cropSize/height)
        
        new_img = Image.new('RGB', (new_width, new_height))
        for i in range(0, new_width, width):
            for j in range(0, new_height, height):
                new_img.paste(img, (i, j))
        return new_img
    else:
        return img


def recursively_read(rootdir, must_contain, classes=[], exts=["png", "jpg", "JPEG", "jpeg"]):
    out = [] 
    for r, d, f in os.walk(rootdir):
        for file in f:
            if (file.split('.')[1] in exts)  and  (must_contain in os.path.join(r, file)):
                if len(classes) == 0:
                    out.append(os.path.join(r, file))
                elif os.path.join(r, file).split('/')[-3] in classes:
                    out.append(os.path.join(r, file))
    return out


def get_list(path, must_contain='', classes=[]):
    if ".pickle" in path:
        with open(path, 'rb') as f:
            image_list = pickle.load(f)
        image_list = [ item for item in image_list if must_contain in item   ]
    else:
        image_list = recursively_read(path, must_contain, classes)
    return image_list


def get_explicit_path_list(path):
    if not path:
        return []
    if ".pickle" in path:
        with open(path, 'rb') as f:
            items = pickle.load(f)
        return [str(item) for item in items]
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Supports both plain txt and csv rows where the first column is path.
            out.append(line.split(",", 1)[0].strip())
    return out


def resolve_wang2020_split_root(base_path, data_label):
    if data_label == "train":
        return os.path.join(base_path, "train")
    if data_label == "val":
        val_dir = os.path.join(base_path, "val")
        if os.path.isdir(val_dir):
            return val_dir
        return os.path.join(base_path, "test", "progan")
    if data_label == "test":
        test_dir = os.path.join(base_path, "test")
        if os.path.isdir(test_dir):
            return test_dir
        return os.path.join(base_path, "test", "progan")
    return os.path.join(base_path, "test", "progan")


class RealFakeDataset(Dataset):
    def __init__(self, opt):
        assert opt.data_label in ["train", "val", "test"]
        
        self.opt = opt
        self.data_label  = opt.data_label
        if opt.data_mode == 'ours':
            pickle_name = f"{opt.data_label}.pickle"
            if opt.data_label == "test" and not os.path.exists(os.path.join(opt.real_list_path, pickle_name)):
                pickle_name = "val.pickle"
            real_list = get_list( os.path.join(opt.real_list_path, pickle_name) )
            fake_list = get_list( os.path.join(opt.fake_list_path, pickle_name) )
        elif opt.data_mode == 'wang2020':
            split_root = resolve_wang2020_split_root(opt.wang2020_data_path, opt.data_label)
            if opt.data_label == 'train':
                # 20(all) classes supervision
                real_list = get_list(split_root, must_contain='0_real')
                fake_list = get_list(split_root, must_contain='1_fake')
            else:
                # 20(all) classes supervision
                real_list = get_list(split_root, must_contain='0_real')
                fake_list = get_list(split_root, must_contain='1_fake')
        elif opt.data_mode == 'ours_wang2020':
            pickle_name = f"{opt.data_label}.pickle"
            if opt.data_label == "test" and not os.path.exists(os.path.join(opt.real_list_path, pickle_name)):
                pickle_name = "val.pickle"
            real_list = get_list( os.path.join(opt.real_list_path, pickle_name) )
            fake_list = get_list( os.path.join(opt.fake_list_path, pickle_name) )
            
            split_root = resolve_wang2020_split_root(opt.wang2020_data_path, opt.data_label)
            real_list += get_list(split_root, must_contain='0_real')
            fake_list += get_list(split_root, must_contain='1_fake')

        train_max_per_class = int(getattr(opt, "train_max_per_class", -1))
        if opt.isTrain and train_max_per_class > 0:
            shuffle(real_list)
            shuffle(fake_list)
            real_list = real_list[:train_max_per_class]
            fake_list = fake_list[:train_max_per_class]
            print(
                f"train_max_per_class enabled: real={len(real_list)}, fake={len(fake_list)}"
            )
        if opt.isTrain:
            hard_real = get_explicit_path_list(getattr(opt, "hard_real_list", ""))
            hard_fake = get_explicit_path_list(getattr(opt, "hard_fake_list", ""))
            real_set = set(real_list)
            fake_set = set(fake_list)
            hard_real = [p for p in hard_real if p in real_set or os.path.exists(p)]
            hard_fake = [p for p in hard_fake if p in fake_set or os.path.exists(p)]
            real_repeats = max(0, int(getattr(opt, "hard_real_repeats", 1)))
            fake_repeats = max(0, int(getattr(opt, "hard_fake_repeats", 1)))
            if hard_real and real_repeats > 0:
                real_list += hard_real * real_repeats
            if hard_fake and fake_repeats > 0:
                fake_list += hard_fake * fake_repeats
            if hard_real or hard_fake:
                print(
                    "hard replay enabled: "
                    f"hard_real={len(hard_real)} x{real_repeats}, "
                    f"hard_fake={len(hard_fake)} x{fake_repeats}, "
                    f"train_real_total={len(real_list)}, train_fake_total={len(fake_list)}"
                )
        eval_max_per_class = int(getattr(opt, "eval_max_per_class", -1))
        if (not opt.isTrain) and eval_max_per_class > 0:
            real_list = real_list[:eval_max_per_class]
            fake_list = fake_list[:eval_max_per_class]
            print(
                f"eval_max_per_class enabled: real={len(real_list)}, fake={len(fake_list)}"
            )

        # setting the labels for the dataset
        self.labels_dict = {}
        for i in real_list:
            self.labels_dict[i] = 0
        for i in fake_list:
            self.labels_dict[i] = 1
        self.domain_dict = {p: self._infer_domain_key(p) for p in real_list + fake_list}

        self.total_list = real_list + fake_list
        shuffle(self.total_list)
        self.targets = [self.labels_dict[p] for p in self.total_list]
        self.domain_targets = [f"{self.labels_dict[p]}:{self.domain_dict[p]}" for p in self.total_list]
        if opt.isTrain:
            crop_func = transforms.RandomCrop(opt.cropSize)
        elif opt.no_crop:
            crop_func = transforms.Lambda(lambda img: img)
        else:
            crop_func = transforms.CenterCrop(opt.cropSize)

        if opt.isTrain and not opt.no_flip:
            flip_func = transforms.RandomHorizontalFlip()
        else:
            flip_func = transforms.Lambda(lambda img: img)
        if not opt.isTrain and opt.no_resize:
            rz_func = transforms.Lambda(lambda img: img)
        else:
            rz_func = transforms.Lambda(lambda img: custom_resize(img, opt))
        
        if opt.arch.lower().startswith("imagenet"):
            stat_from = "imagenet"
        elif opt.arch.lower().startswith("clip"):
            stat_from = "clip"
        elif opt.arch.lower().startswith("siglip"):
            stat_from = "siglip"
        elif opt.arch.lower().startswith("beitv2"):
            stat_from = "beitv2"
        elif opt.arch.lower().startswith("dinov3"):
            stat_from = "dinov3"
        else:
            stat_from = "clip"

        print("mean and std stats are from: ", stat_from)
        if '2b' not in opt.arch:
            print ("using selected backbone normalization")
            self.transform = transforms.Compose([
                # rz_func,
                transforms.Lambda(lambda img: translate_duplicate(img, opt.loadSize)),
                crop_func,
                flip_func,
                transforms.ToTensor(),
                transforms.Normalize( mean=MEAN[stat_from], std=STD[stat_from] ),
            ])
        else:
            print ("Using CLIP 2B transform")
            self.transform = None # will be initialized in trainer.py

    def __len__(self):
        return len(self.total_list)

    def __getitem__(self, idx):
        img_path = self.total_list[idx]
        label = self.labels_dict[img_path]
        img = Image.open(img_path).convert("RGB")
        if self.data_label == "train" and getattr(self.opt, "data_aug", False):
            img = data_augment(img, self.opt, label=label)
        img = self.transform(img)
        return img, label

    def _infer_domain_key(self, path):
        parts = os.path.normpath(path).split(os.sep)
        for idx, part in enumerate(parts):
            if part in ("0_real", "1_fake") and idx > 0:
                return parts[idx - 1]
        if len(parts) >= 3:
            return parts[-3]
        return "unknown"


def parse_aug_range(raw, default_min, default_max):
    if raw is None:
        return [default_min, default_max]
    if isinstance(raw, str):
        values = [float(v.strip()) for v in raw.split(",") if v.strip()]
    else:
        values = [float(v) for v in raw]
    if len(values) == 1:
        return [values[0], values[0]]
    return [values[0], values[1]]


def sample_range(raw, default_min, default_max):
    return sample_continuous(parse_aug_range(raw, default_min, default_max))


def resize_down_up(img, scale, interp):
    w, h = img.size
    down_w = max(8, int(round(w * scale)))
    down_h = max(8, int(round(h * scale)))
    resample = rz_dict.get(interp, Image.BILINEAR)
    return img.resize((down_w, down_h), resample=resample).resize((w, h), resample=resample)


def _to_rgb_array(img):
    arr = np.array(img)
    if arr.ndim == 2:
        arr = np.expand_dims(arr, axis=2)
        arr = np.repeat(arr, 3, axis=2)
    return arr


def _apply_mid_freq_suppress_np(img, low_ratio, high_ratio, strength):
    arr = np.asarray(img, dtype=np.float32).clip(0.0, 255.0) / 255.0
    h, w = arr.shape[:2]
    yy, xx = np.meshgrid(
        np.arange(h, dtype=np.float32),
        np.arange(w, dtype=np.float32),
        indexing="ij",
    )
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    max_radius = float(np.sqrt(cy * cy + cx * cx) + 1e-6)
    low = float(low_ratio) * max_radius
    high = float(high_ratio) * max_radius
    ring = ((dist >= low) & (dist <= high)).astype(np.float32)
    atten = 1.0 - np.clip(float(strength), 0.0, 1.0) * ring
    freq = np.fft.fftshift(np.fft.fft2(arr, axes=(0, 1)), axes=(0, 1))
    freq = freq * atten[:, :, None]
    rec = np.fft.ifft2(np.fft.ifftshift(freq, axes=(0, 1)), axes=(0, 1)).real
    rec = np.clip(rec, 0.0, 1.0)
    return Image.fromarray((rec * 255.0).astype(np.uint8))


def _random_crop_resize(img, min_scale, max_scale, interp):
    w, h = img.size
    scale = sample_continuous([float(min_scale), float(max_scale)])
    scale = max(0.1, min(1.0, float(scale)))
    crop_w = max(8, int(round(w * scale)))
    crop_h = max(8, int(round(h * scale)))
    if crop_w >= w and crop_h >= h:
        return img
    left = randint(0, max(0, w - crop_w))
    top = randint(0, max(0, h - crop_h))
    resample = rz_dict.get(interp, Image.BILINEAR)
    return img.crop((left, top, left + crop_w, top + crop_h)).resize((w, h), resample=resample)


def _apply_social_chain_pil(img, opt, hard=False):
    out = img
    min_ops = int(getattr(opt, "social_chain_min_ops", 2))
    max_ops = int(getattr(opt, "social_chain_max_ops", 4))
    if hard:
        min_ops = int(getattr(opt, "fake_social_chain_min_ops", max(min_ops, 3)))
        max_ops = int(getattr(opt, "fake_social_chain_max_ops", max(max_ops, 5)))
    min_ops = max(1, min_ops)
    max_ops = max(min_ops, max_ops)

    ops = []
    if getattr(opt, "social_chain_use_crop", True):
        ops.append("crop")
    if getattr(opt, "social_chain_use_resize", True):
        ops.append("resize")
    if getattr(opt, "social_chain_use_jpeg", True):
        ops.append("jpeg")
    if getattr(opt, "social_chain_use_blur", True):
        ops.append("blur")
    if getattr(opt, "social_chain_use_color", True):
        ops.append("color")
    if getattr(opt, "social_chain_use_noise", True):
        ops.append("noise")
    if hard and getattr(opt, "fake_dataset_use_mid_suppress", True):
        ops.append("mid_suppress")

    if not ops:
        return out

    shuffle(ops)
    num_ops = randint(min_ops, min(max_ops, len(ops)))
    selected = ops[:num_ops]

    for op in selected:
        if op == "crop":
            min_scale, max_scale = parse_aug_range(
                getattr(opt, "social_chain_crop_scale", "0.78,1.0"),
                0.78,
                1.0,
            )
            interp = sample_discrete(getattr(opt, "rz_interp", ["bilinear"]))
            out = _random_crop_resize(out, min_scale, max_scale, interp)
        elif op == "resize":
            min_scale, max_scale = parse_aug_range(
                getattr(opt, "social_chain_resize_scale", "0.45,0.95"),
                0.45,
                0.95,
            )
            scale = sample_continuous([min_scale, max_scale])
            interp = sample_discrete(getattr(opt, "rz_interp", ["bilinear"]))
            out = resize_down_up(out, scale, interp)
        elif op == "jpeg":
            quality_raw = getattr(opt, "social_chain_jpg_qual", "30,95")
            if hard:
                quality_raw = getattr(opt, "fake_dataset_hard_jpg_qual", "18,80")
            qual = sample_discrete(range(int(parse_aug_range(quality_raw, 30, 95)[0]), int(parse_aug_range(quality_raw, 30, 95)[1]) + 1))
            method = sample_discrete(getattr(opt, "jpg_method", ["pil"]))
            arr = _to_rgb_array(out)
            out = Image.fromarray(np.clip(jpeg_from_key(arr, int(qual), method), 0, 255).astype(np.uint8))
        elif op == "blur":
            blur_sig = getattr(opt, "social_chain_blur_sig", "0.0,1.2")
            if hard:
                blur_sig = getattr(opt, "fake_dataset_hard_blur_sig", "0.6,2.0")
            sig = sample_continuous(parse_aug_range(blur_sig, 0.0, 1.2))
            arr = _to_rgb_array(out)
            gaussian_blur(arr, sig)
            out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        elif op == "color":
            brightness = sample_range(getattr(opt, "social_chain_brightness_range", "0.90,1.12"), 0.90, 1.12)
            contrast = sample_range(getattr(opt, "social_chain_contrast_range", "0.90,1.12"), 0.90, 1.12)
            saturation = sample_range(getattr(opt, "social_chain_saturation_range", "0.90,1.12"), 0.90, 1.12)
            if hard:
                brightness = sample_range(getattr(opt, "fake_dataset_hard_brightness_range", "0.82,1.18"), 0.82, 1.18)
                contrast = sample_range(getattr(opt, "fake_dataset_hard_contrast_range", "0.82,1.18"), 0.82, 1.18)
                saturation = sample_range(getattr(opt, "fake_dataset_hard_saturation_range", "0.82,1.18"), 0.82, 1.18)
            out = ImageEnhance.Brightness(out).enhance(brightness)
            out = ImageEnhance.Contrast(out).enhance(contrast)
            out = ImageEnhance.Color(out).enhance(saturation)
        elif op == "noise":
            arr = _to_rgb_array(out).astype(np.float32)
            if hard:
                noise_std = sample_continuous(parse_aug_range(getattr(opt, "fake_dataset_hard_noise_std", "0.0,8.0"), 0.0, 8.0))
            else:
                noise_std = sample_continuous(parse_aug_range(getattr(opt, "social_chain_noise_std", "0.0,4.0"), 0.0, 4.0))
            if noise_std > 0:
                arr = arr + np.random.normal(0.0, float(noise_std), size=arr.shape).astype(np.float32)
            out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        elif op == "mid_suppress":
            low_ratio, high_ratio = parse_aug_range(
                getattr(opt, "fake_dataset_mid_suppress_band", "0.15,0.45"),
                0.15,
                0.45,
            )
            strength = sample_continuous(parse_aug_range(getattr(opt, "fake_dataset_mid_suppress_strength", "0.25,0.70"), 0.25, 0.70))
            out = _apply_mid_freq_suppress_np(out, low_ratio, high_ratio, strength)
    return out


def data_augment(img, opt, label=None):
    if random() < getattr(opt, "color_aug_prob", 0.5):
        brightness = sample_range(getattr(opt, "brightness_range", "0.85,1.15"), 0.85, 1.15)
        contrast = sample_range(getattr(opt, "contrast_range", "0.85,1.15"), 0.85, 1.15)
        saturation = sample_range(getattr(opt, "saturation_range", "0.85,1.15"), 0.85, 1.15)
        img = ImageEnhance.Brightness(img).enhance(brightness)
        img = ImageEnhance.Contrast(img).enhance(contrast)
        img = ImageEnhance.Color(img).enhance(saturation)

    if random() < getattr(opt, "resize_aug_prob", 0.25):
        scale = sample_range(getattr(opt, "resize_aug_scale", "0.5,1.0"), 0.5, 1.0)
        interp = sample_discrete(getattr(opt, "rz_interp", ["bilinear"]))
        img = resize_down_up(img, scale, interp)

    if getattr(opt, "use_social_chain_aug", False) and random() < float(getattr(opt, "social_chain_prob", 0.0)):
        img = _apply_social_chain_pil(img, opt, hard=False)

    if label == 1 and getattr(opt, "use_fake_dataset_hardening", False) and random() < float(getattr(opt, "fake_dataset_hard_prob", 0.0)):
        img = _apply_social_chain_pil(img, opt, hard=True)

    img = np.array(img)
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
        img = np.repeat(img, 3, axis=2)

    if random() < opt.blur_prob:
        sig = sample_continuous(opt.blur_sig)
        gaussian_blur(img, sig)

    if random() < opt.jpg_prob:
        method = sample_discrete(opt.jpg_method)
        qual = sample_discrete(opt.jpg_qual)
        img = jpeg_from_key(img, qual, method)

    img = np.clip(img, 0, 255).astype(np.uint8)
    return Image.fromarray(img)


def sample_continuous(s):
    if len(s) == 1:
        return s[0]
    if len(s) == 2:
        rg = s[1] - s[0]
        return random() * rg + s[0]
    raise ValueError("Length of iterable s should be 1 or 2.")


def sample_discrete(s):
    if len(s) == 1:
        return s[0]
    return choice(s)


def gaussian_blur(img, sigma):
    gaussian_filter(img[:,:,0], output=img[:,:,0], sigma=sigma)
    gaussian_filter(img[:,:,1], output=img[:,:,1], sigma=sigma)
    gaussian_filter(img[:,:,2], output=img[:,:,2], sigma=sigma)


def cv2_jpg(img, compress_val):
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("cv2 JPEG augmentation requires OpenCV runtime dependencies.") from exc
    img_cv2 = img[:,:,::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode('.jpg', img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:,:,::-1]


def pil_jpg(img, compress_val):
    out = BytesIO()
    img = Image.fromarray(img)
    img.save(out, format='jpeg', quality=compress_val)
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return img


jpeg_dict = {'cv2': cv2_jpg, 'pil': pil_jpg}
def jpeg_from_key(img, compress_val, key):
    method = jpeg_dict[key]
    return method(img, compress_val)


rz_dict = {'bilinear': Image.BILINEAR,
           'bicubic': Image.BICUBIC,
           'lanczos': Image.LANCZOS,
           'nearest': Image.NEAREST}
def custom_resize(img, opt):
    interp = sample_discrete(opt.rz_interp)
    return TF.resize(img, opt.loadSize, interpolation=rz_dict[interp])
