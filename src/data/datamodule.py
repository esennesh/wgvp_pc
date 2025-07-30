from abc import abstractmethod, abstractproperty
from jax import Array
import jax.numpy as jnp
import numpy as np
from jax.tree_util import tree_map
from pytrie import SortedStringTrie as Trie
from torch.utils.data import Dataset, DataLoader, default_collate, random_split
from typing import Any, Dict, Tuple

def numpy_collate(batch):
  return tree_map(np.asarray, default_collate(batch))

class IndexedDataset(Dataset):
    def __init__(self, dataset):
        self._dataset = dataset

    def __getitem__(self, idx):
        item = self._dataset[idx]
        return (*item, np.array(idx))

    def __len__(self):
        return len(self._dataset)

class DataMutables:
    def __init__(self, length, idx_dim=1):
        self._idx_dim = idx_dim
        self._length = length
        self._mutables = Trie()

    def get_parameter(self, idx: np.ndarray, key: str) -> Array:
        val = np.take(self.mutables[key], idx, axis=self._idx_dim)
        assert val.shape[self._idx_dim] == len(idx)
        return jnp.array(val)

    def get_parameters(self, idx: np.ndarray) -> Dict[str, Array]:
        return {name: self.get_parameter(idx, name) for name in self.mutables}

    def __len__(self):
        return self._length

    @property
    def mutables(self):
        return self._mutables

    def pickle(self):
        return {
            "idx_dim": self._idx_dim,
            "length": len(self),
            "parameters": {k: v for k, v in self.mutables.items()}
        }

    def _require(self, key: str, shape: Tuple):
        assert len(shape) >= 2
        shape = list(shape)
        shape[self._idx_dim] = len(self)
        if key not in self.mutables:
            self.mutables[key] = np.empty(tuple(shape))

    def set_parameter(self, idx: np.ndarray, key: str, val: Array):
        self._require(key, val.shape)
        indices = idx.reshape((1, len(idx)) +\
                              (1,) * len(val.shape[self._idx_dim+1:]))
        np.put_along_axis(self.mutables[key],
                          np.broadcast_to(indices, val.shape),
                          np.array(val), self._idx_dim)

    def set_parameters(self, idx: np.ndarray, parameters: Dict[str, Array]):
        for key, val in parameters.items():
            self.set_parameter(idx, key, val)

    @classmethod
    def unpickle(cls, saved):
        self = cls(saved["length"], idx_dim=saved["idx_dim"])
        for k, v in saved["parameters"].items():
            self.mutables[k] = v
        return self

class DataModule:
    """
    Base class for all data modules
    """
    def __init__(self, batch_size: int=64, data_dir: str="data/", drop_last=False,
                 collate_fn=numpy_collate, indexed: bool=False, idx_dim=1,
                 mutables=False, num_workers: int=1, pin_memory: bool=False,
                 shuffle: bool=True, validation_split: float=0.1):
        self.data_dir = data_dir
        self.validation_split = validation_split
        data_train, data_test = self.prepare_data()
        if indexed:
            data_train = IndexedDataset(data_train)
            data_test = IndexedDataset(data_test)
        self.data_train, self.data_val, self.data_test =\
            self.setup(data_train, data_test, validation_split)
        if mutables:
            assert indexed
            self.train_mutables = DataMutables(len(data_train), idx_dim)
            self.test_mutables = DataMutables(len(data_test), idx_dim)
        else:
            self.train_mutables, self.valid_mutables, self.test_mutables =\
                (None, None, None)

        self.dataloader_kwargs = {
            'batch_size': batch_size,
            'collate_fn': collate_fn,
            'drop_last': drop_last,
            'num_workers': num_workers,
            'pin_memory': pin_memory,
            'shuffle': shuffle,
        }

    @abstractmethod
    def prepare_data(self) -> Tuple[Dataset, Dataset]:
        raise NotImplementedError

    def resume(self, checkpoint):
        if self.train_mutables and self.test_mutables:
            self.test_mutables =\
                DataMutables.unpickle(checkpoint['test_mutables'])
            self.train_mutables =\
                DataMutables.unpickle(checkpoint['train_mutables'])

    def save(self) -> Dict[str, Any]:
        return {
            "test_mutables": self.test_mutables.pickle(),
            "train_mutables": self.train_mutables.pickle()
        }

    @staticmethod
    def setup(train_data, test_data, validation_split) -> Tuple[Dataset, Dataset, Dataset]:
        val_length = int(len(train_data) * validation_split)
        train_val_split = (len(train_data) - val_length, val_length)
        train_data, val_data = random_split(dataset=train_data,
                                            lengths=train_val_split)
        return train_data, val_data, test_data

    @abstractproperty
    def shape(self) -> Tuple:
        raise NotImplementedError

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_test,
            **self.dataloader_kwargs,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_train,
            **self.dataloader_kwargs,
        )

    def valid_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_val,
            **self.dataloader_kwargs,
        )
