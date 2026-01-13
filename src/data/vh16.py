import jax.numpy as jnp
import numpy as np
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, TensorDataset
from torchvision import datasets, transforms
from typing import Callable, Optional, Tuple

from . import datamodule

def split_data(n_samples: int, n_blocks: int = 1, vld_portion: float = 0.2):
    assert 0 < vld_portion < 1
    indices = range(n_samples)
    block_size = len(indices) // n_blocks

    trn_inds, vld_inds = [], []
    for b in range(n_blocks):
        start = b * block_size
        if b == n_blocks - 1:
            end = len(indices)
        else:
            end = start + block_size

        block_inds = indices[start:end]
        vld_size = np.round(len(block_inds) * vld_portion)
        trn_size = len(block_inds) - int(vld_size)

        trn_inds.extend(block_inds[:trn_size])
        vld_inds.extend(block_inds[trn_size:])

    assert not set(trn_inds).intersection(vld_inds), "must be non-overlapping"
    trn_inds, vld_inds = map(lambda x: np.array(x), [trn_inds, vld_inds])
    return trn_inds, vld_inds

def setup_kwargs(defaults, kwargs):
    if not kwargs:
        return defaults
    for k, v in defaults.items():
        if k not in kwargs:
            kwargs[k] = v
    return kwargs

def make_dataset(dataset: str, load_dir: str='data', **kwargs):
    vld = None
    tst = None

    if any(s in dataset for s in ['vH', 'Kyoto']):
        defaults = dict(
            file_name='processed.npy',
            shift_rescale=False,
            # trn/vld split
            vld_portion=0.2,
            n_blocks=100,
        )
        kwargs = setup_kwargs(defaults, kwargs)

        data = os.path.join(
            load_dir,
            'DOVES' if 'vH' in dataset else 'Kyoto',
            dataset,
            kwargs['file_name'],
        )
        data = np.load(data)

        if kwargs['shift_rescale']:
            mu = np.nanmean(data)
            sd = np.nanstd(data)
            data = (data - mu) / sd

        trn_inds, vld_inds = split_data(
            n_samples=len(data),
            n_blocks=kwargs['n_blocks'],
            vld_portion=kwargs['vld_portion'],
        )
        trn, tst = data[trn_inds], data[vld_inds]
        trn, tst = map(lambda a: torch.tensor(data=a, dtype=torch.float),
                       [trn, tst])
        trn = torch.utils.data.TensorDataset(trn)
        tst = torch.utils.data.TensorDataset(tst)
    elif dataset == 'CIFAR16':
        path = os.path.join(load_dir, 'CIFAR10', 'xtract16')
        trn = np.load(os.path.join(path, 'trn', 'processed.npy'))
        vld = np.load(os.path.join(path, 'vld', 'processed.npy'))

        trn, vld = map(_to_device_fun(device), [trn, vld])
        trn = torch.utils.data.TensorDataset(trn)
        vld = torch.utils.data.TensorDataset(vld)
    elif dataset in ['MNIST', 'FashionMNIST']:
        path = os.path.join(load_dir, dataset, 'processed')
        if os.path.isfile(os.path.join(path, 'x_trn.npy')):
            trn, vld, tst = _load(path, device)
        else:
            # get transforms
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            # make dataset
            kws = dict(root=load_dir, transform=transform)
            dataset = getattr(torchvision.datasets, dataset)
            trn = dataset(train=True, **kws)
            vld = dataset(train=False, **kws)
            # process and save
            trn, vld, _ = _process_and_save(
                trn=trn, vld=vld, save_dir=path)
    elif dataset == 'EMNIST':
        path = os.path.join(load_dir, 'EMNIST', 'processed')
        if os.path.isfile(os.path.join(path, 'x_trn.npy')):
            trn, vld, tst = _load(path, device)
        else:
            # get transforms
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            # make dataset
            kws = dict(root=load_dir, transform=transform, split='letters')
            trn = torchvision.datasets.EMNIST(train=True, **kws)
            vld = torchvision.datasets.EMNIST(train=False, **kws)
            # process and save
            trn, vld, _ = _process_and_save(
                trn=trn,
                vld=vld,
                save_dir=path,
                transpose=True,
            )
    elif dataset == 'Omniglot':
        path = os.path.join(load_dir, 'omniglot-py', 'processed')
        if os.path.isfile(os.path.join(path, 'x_trn.npy')):
            trn, vld, tst = _load(path, device)
        else:
            # get transforms
            kws_resize = dict(
                size=28,
                antialias=True,
                interpolation=F_vis.InterpolationMode.NEAREST,
            )
            transform = transforms.Compose([
                transforms.Resize(**kws_resize),
                InvertBinaryPILImage(),
                transforms.ToTensor(),
            ])
            # make dataset
            kws = dict(root=load_dir, transform=transform)
            trn = torchvision.datasets.Omniglot(background=True, **kws)
            vld = torchvision.datasets.Omniglot(background=False, **kws)
            # process and save
            trn, vld, _ = _process_and_save(
                trn=trn, vld=vld, save_dir=path)
    elif dataset == 'SVHN':
        defaults = dict(grey=True)
        kwargs = setup_kwargs(defaults, kwargs)
        path = 'processed' + ('_grey' if kwargs['grey'] else '')
        path = os.path.join(load_dir, 'SVHN', path)
        if os.path.isfile(os.path.join(path, 'x_trn.npy')):
            trn, vld, tst = _load(path, device)
        else:
            # get transforms
            transform = [transforms.ToTensor(), transforms.Grayscale()
                         if kwargs['grey'] else None, transforms.Normalize(
                            mean=(0.5,) * 3, std=(0.5,) * 3
                         ) if not kwargs['grey'] else None]
            transform = transforms.Compose([t for t in transform if t\
                                            is not None])
            # make dataset
            kws = dict(root=os.path.join(load_dir, 'SVHN'), transform=transform)
            trn = torchvision.datasets.SVHN(split='train', **kws)
            vld = torchvision.datasets.SVHN(split='test', **kws)
            # process and save
            trn, vld, _ = _process_and_save(trn=trn, vld=vld, save_dir=path)
    elif dataset == 'CIFAR10':
        defaults = dict(grey=False, augment=True)
        kwargs = setup_kwargs(defaults, kwargs)
        path = 'processed' + ('_grey' if kwargs['grey'] else '')
        path = os.path.join(load_dir, 'CIFAR10', path)
        if os.path.isfile(os.path.join(path, 'x_trn.npy')):
            trn, vld, tst = _load(path, device)
        else:
            # get transforms
            transform_list = [transforms.ToTensor(), transforms.Grayscale()
                              if kwargs['grey'] else None,
                              transforms.Normalize(
                                  mean=(0.5,) if kwargs['grey'] else (0.5,) * 3,
                                  std=(0.5,) if kwargs['grey'] else (0.5,) * 3
                              )]
            transform_list = [
                t for t in transform_list
                if t is not None
            ]
            transform = transforms.Compose(transform_list)
            # make dataset
            kws = dict(
                root=os.path.join(load_dir, 'CIFAR10'),
                transform=transform,
            )
            trn = torchvision.datasets.CIFAR10(train=True, **kws)
            vld = torchvision.datasets.CIFAR10(train=False, **kws)
            if kwargs['augment']:
                flip = transforms.RandomHorizontalFlip(p=1.0)
                transform = transforms.Compose([flip] + transform_list)
                kws['transform'] = transform
                trn_aug = torchvision.datasets.CIFAR10(train=True, **kws)
            else:
                trn_aug = None
                # process and save
                trn, vld, _ = _process_and_save(
                    trn=trn,
                    vld=vld,
                    trn_aug=trn_aug,
                    save_dir=path,
                )
    elif dataset == 'ImageNet32':
        defaults = dict(skip_trn=False)
        kwargs = setup_kwargs(defaults, kwargs)
        path = os.path.join(load_dir, dataset, 'processed')
        if os.path.isfile(os.path.join(path, 'x_trn.npy')):
            trn, vld, tst = _load(path, device, kwargs['skip_trn'])
        else:
            raise RuntimeError(dataset)
    elif dataset == 'CelebA':
        defaults = dict(grey=False)
        kwargs = setup_kwargs(defaults, kwargs)
        path = 'processed' + ('_grey' if kwargs['grey'] else '')
        path = os.path.join(load_dir, dataset, path)
        if os.path.isfile(os.path.join(path, 'x_trn.npy')):
            trn, vld, tst = _load(path, device)
        else:
            # get transforms
            transform_list = [transforms.ToTensor(),
                              transforms.Grayscale() if kwargs['grey']\
                              else None,
                              transforms.Normalize(mean=(0.5,)\
                              if kwargs['grey'] else (0.5,) * 3,
                              std=(0.5,) if kwargs['grey'] else (0.5,) * 3)]
            transform_list = [t for t in transform_list if t is not None]
            transform = transforms.Compose(transform_list)
            # make dataset
            kws = dict(
                root=os.path.join(load_dir, dataset),
                transform=transform,
            )
            trn = torchvision.datasets.CelebA(split='train', **kws)
            vld = torchvision.datasets.CelebA(split='valid', **kws)
            tst = torchvision.datasets.CelebA(split='test', **kws)
            # process and save
            trn, vld, tst = _process_and_save(
                trn=trn,
                vld=vld,
                tst=tst,
                save_dir=path,
            )
    elif dataset.startswith('BALLS'):
        defaults = dict(vld_split=0.25)
        kwargs = setup_kwargs(defaults, kwargs)

        # create load path
        path = os.path.join(
            load_dir,
            'BALLS',
            f"npix-{int_from_str(dataset)}",
        )
        # load
        x_tst = np.load(os.path.join(path, 'x_tst.npy'))
        z_tst = np.load(os.path.join(path, 'z_tst.npy'))
        x = np.load(os.path.join(path, 'x.npy'))
        z = np.load(os.path.join(path, 'z.npy'))
        # split into trn / vld
        frac = 1 - kwargs['vld_split']
        idx = int(len(x) * frac)
        x_trn, x_vld = x[:idx], x[idx:]
        z_trn, z_vld = z[:idx], z[idx:]

        # to tensor
        x_trn, x_vld, x_tst, z_trn, z_vld, z_tst = map(
            _to_device_fun(device),
            [x_trn, x_vld, x_tst, z_trn, z_vld, z_tst],
        )
        # to dataset
        trn = torch.utils.data.TensorDataset(x_trn, z_trn)
        vld = torch.utils.data.TensorDataset(x_vld, z_vld)
        tst = torch.utils.data.TensorDataset(x_tst, z_tst)
    else:
        raise ValueError(dataset)

    return trn, vld, tst

class VanHaterenDataModule(datamodule.DataModule):
    def __init__(self, root="data", patch_side=16, **kwargs):
        self.num_data = 0
        self.patch_side = patch_side
        self.root_path = root
        super().__init__(**kwargs)

    def prepare_data(self) -> Tuple[Dataset, Dataset]:
        dim = self.patch_side
        train, valid, test = make_dataset("vH%d" % dim, load_dir=self.root_path)
        self.num_data += len(train) + len(test)
        return train, test

    @property
    def shape(self):
        return (self.num_data, 1, self.patch_side, self.patch_side)
