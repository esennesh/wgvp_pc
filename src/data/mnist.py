import jax.numpy as jnp
import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from typing import Tuple

from . import datamodule

class MnistDataModule(datamodule.DataModule):
    def __init__(self, *args, binarize=True, **kwargs):
        trnsfrms = [
            transforms.Lambda(
                lambda x: np.array(x, dtype=jnp.float32) / 255.
            ),
            transforms.Lambda(lambda x: x.reshape(1, 28, 28)),
        ]
        if binarize:
            trnsfrms.append(
                transforms.Lambda(lambda x: np.round(x, decimals=0))
            )
        self.transforms = transforms.Compose(trnsfrms)
        self.num_data = 0

        super().__init__(*args, **kwargs)

    def prepare_data(self) -> Tuple[Dataset, Dataset]:
        data_train = datasets.MNIST(self.data_dir, train=True, download=True,
                                    transform=self.transforms)
        data_test = datasets.MNIST(self.data_dir, train=False, download=True,
                                   transform=self.transforms)
        self.num_data += len(data_train) + len(data_test)
        return data_train, data_test

    @property
    def shape(self) -> Tuple:
        return (self.num_data, 1, 28, 28)
