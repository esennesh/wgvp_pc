from abc import abstractmethod, abstractproperty
from jax import Array
import jax.numpy as jnp
import numpy as np
from pytrie import SortedStringTrie as Trie
from torch.utils.data import DataLoader
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.data import DataModule

class ParaMonad:
    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def load(self, checkpoint: Dict[str, Any]):
        raise NotImplementedError

    @abstractproperty
    def parameters(self) -> Dict[str, Dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def save(self) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def setup_step(self, datamodule: DataModule, stage: str=""):
        raise NotImplementedError

    @abstractmethod
    def test_step(self, *args, **kwargs) -> Dict[str, float]:
        raise NotImplementedError

    @abstractmethod
    def train_step(self, *args, **kwargs) -> Dict[str, float]:
        raise NotImplementedError

    @abstractmethod
    def valid_step(self, *args, **kwargs) -> Dict[str, float]:
        raise NotImplementedError

class BatchParameters:
    def __init__(self, length, axis=1):
        self._axis = axis
        self._length = length
        self._tensors = Trie()

    def __contains__(self, key: str) -> bool:
        return key in self.tensors

    def get_parameter(self, idx: np.ndarray, key: str) -> Array:
        val = np.take(self.tensors[key], idx, axis=self._axis)
        assert val.shape[self._axis] == len(idx)
        return jnp.array(val)

    def get_parameters(self, idx: np.ndarray) -> Dict[str, Array]:
        return {name: self.get_parameter(idx, name) for name in self.tensors}

    def __len__(self):
        return self._length

    def pickle(self):
        return {
            "axis": self._axis,
            "length": len(self),
            "tensors": {k: v for k, v in self.tensors.items()}
        }

    def _require(self, key: str, shape: Tuple):
        assert len(shape) >= 2
        shape = list(shape)
        shape[self._axis] = len(self)
        if key not in self.tensors:
            self.tensors[key] = np.empty(tuple(shape))

    def set_parameter(self, idx: np.ndarray, key: str, val: Array):
        self._require(key, val.shape)
        indices = idx.reshape((1, len(idx)) +\
                              (1,) * len(val.shape[self._axis+1:]))
        np.put_along_axis(self.tensors[key],
                          np.broadcast_to(indices, val.shape), np.array(val),
                          self._axis)

    def set_parameters(self, idx: np.ndarray,
                       parameters: Dict[Tuple[str, bool], Array]):
        for key, val in parameters.items():
            self.set_parameter(idx, key, val)

    @property
    def tensors(self):
        return self._tensors

    @classmethod
    def unpickle(cls, saved):
        self = cls(saved["length"], axis=saved["axis"])
        for k, v in saved["tensors"].items():
            self.tensors[k] = v
        return self
