import torch
import numpy as np
import math
from torch.utils.data import Sampler
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler

from .datasets import RealFakeDataset


class DistributedWeightedSampler(Sampler):
    def __init__(
        self,
        weights,
        num_replicas=None,
        rank=None,
        replacement=True,
        seed=0,
        drop_last=False,
    ):
        if num_replicas is None:
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            rank = torch.distributed.get_rank()
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.replacement = bool(replacement)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0

        if self.drop_last and len(self.weights) % self.num_replicas != 0:
            self.num_samples = math.ceil((len(self.weights) - self.num_replicas) / self.num_replicas)
        else:
            self.num_samples = math.ceil(len(self.weights) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(
            self.weights,
            self.total_size,
            self.replacement,
            generator=generator,
        ).tolist()
        indices = indices[self.rank:self.total_size:self.num_replicas]
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = int(epoch)


def get_bal_sampler(dataset):
    targets = dataset.targets if hasattr(dataset, "targets") else []
    if len(targets) == 0:
        raise ValueError("Dataset does not expose targets.")

    ratio = np.bincount(targets)
    w = 1. / torch.tensor(ratio, dtype=torch.float)
    sample_weights = w[targets]
    sampler = WeightedRandomSampler(weights=sample_weights,
                                    num_samples=len(sample_weights))
    return sampler


def get_domain_bal_weights(dataset):
    domain_targets = getattr(dataset, "domain_targets", None)
    if domain_targets is None:
        raise ValueError("Dataset does not expose domain_targets.")
    counts = {}
    for target in domain_targets:
        counts[target] = counts.get(target, 0) + 1
    return torch.tensor(
        [1.0 / float(counts[target]) for target in domain_targets],
        dtype=torch.float,
    )


def get_domain_bal_sampler(dataset):
    sample_weights = get_domain_bal_weights(dataset)
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights))


def create_dataloader(opt, preprocess=None):
    shuffle = not opt.serial_batches if (opt.isTrain and not opt.class_bal) else False
    dataset = RealFakeDataset(opt)
    
    if '2b' in opt.arch:
        dataset.transform = preprocess
    sampler = None
    if getattr(opt, "use_domain_balanced_sampler", False) and opt.isTrain:
        if getattr(opt, "distributed", False):
            sampler = DistributedWeightedSampler(
                get_domain_bal_weights(dataset),
                num_replicas=opt.world_size,
                rank=opt.rank,
                seed=0,
            )
        else:
            sampler = get_domain_bal_sampler(dataset)
    elif opt.class_bal:
        if getattr(opt, "distributed", False) and opt.isTrain:
            raise ValueError("class_bal is not supported together with distributed training.")
        sampler = get_bal_sampler(dataset)
    if getattr(opt, "distributed", False) and opt.isTrain:
        if sampler is not None:
            shuffle = False
        else:
            sampler = DistributedSampler(
                dataset,
                num_replicas=opt.world_size,
                rank=opt.rank,
                shuffle=shuffle,
            )
            shuffle = False
    elif sampler is not None:
        shuffle = False

    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=opt.batch_size,
                                              shuffle=shuffle,
                                              sampler=sampler,
                                              num_workers=int(opt.num_threads),
                                              pin_memory=torch.cuda.is_available(),
                                              persistent_workers=int(opt.num_threads) > 0)
    return data_loader
