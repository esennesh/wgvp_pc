from contextlib import ExitStack
import functools
import jax
from jax import Array
import jax.numpy as jnp
import math
import numpy as np
import numpyro
from numpyro.distributions import constraints
import numpyro.distributions as dist
from numpyro.infer.autoguide import AutoGuide
from numpyro.infer.initialization import init_to_sample
from numpyro.infer import Predictive
from numpyro.infer.util import log_density
from pytrie import SortedStringTrie as Trie
from typing import Any, Dict, Tuple

from src.inference import AutoLangevin
from src.utils import uncondition
from .svi import SviPara

class ParameterParticles:
    def __init__(self, num_data, num_particles, batch_dim=1, particle_dim=0):
        self._batch_dim = batch_dim
        self._num_data = num_data
        self._num_particles = num_particles
        self._parameters = Trie()
        self._particle_dim = particle_dim

    def get_parameters(self, idx: np.ndarray, key: str) -> Array:
        val = np.take(self.parameters[key], idx, axis=self._batch_dim)
        assert val.shape[self._batch_dim] == len(idx)
        assert val.shape[self._particle_dim] == self.num_particles
        return jnp.array(val)

    @property
    def num_data(self):
        return self._num_data

    @property
    def num_particles(self):
        return self._num_particles

    @property
    def parameters(self):
        return self._parameters

    def pickle(self):
        return {
            "batch_dim": self._batch_dim,
            "num_data": self.num_data,
            "num_particles": self.num_particles,
            "parameters": {k: v for k, v in self.parameters.items()},
            "particle_dim": self._particle_dim
        }

    def _require(self, key: str, shape: Tuple):
        assert len(shape) > 2
        shape = list(shape)
        shape[self._batch_dim] = self._num_data
        shape[self._particle_dim] = self.num_particles
        if key not in self.parameters:
            self.parameters[key] = np.empty(tuple(shape))

    def set_parameters(self, idx: np.ndarray, key: str, val: Array):
        assert val.shape[self._particle_dim] == self.num_particles
        self._require(key, val.shape)

        indices = idx.reshape((1, len(idx)) +\
                              (1,) * len(val.shape[self._batch_dim+1:]))
        np.put_along_axis(self.parameters[key],
                          np.broadcast_to(indices, val.shape),
                          np.array(val), self._batch_dim)

    @classmethod
    def unpickle(cls, saved):
        self = ParameterParticles(saved["num_data"], saved["num_particles"],
                                  batch_dim=saved["batch_dim"],
                                  particle_dim=saved["particle_dim"])
        for k, v in saved["parameters"].items():
            self.parameters[k] = v
        return self

class PgdPara(SviPara):
    def __init__(self, data_shape, lr, model, num_particles, rng, guide=None,
                 lrq=1e-4):
        super().__init__(data_shape, AutoLangevin(model, lr=lrq), lr, model,
                         num_particles, rng)

    def __call__(self, data, targets, indices, mutables=None):
        if mutables is None:
            mutables = {}

        for site in self.svi.guide.prototype_trace:
            if site not in mutables:
                continue
            mutable = "{}_{}_loc".format(site, self.svi.guide.prefix)
            self.svi_state.mutable_state[mutable]["value"] = mutables[site]

        params = {**self.svi.get_params(self.svi_state),
                  **self.svi_state.mutable_state}
        predictive = Predictive(
            uncondition(self.svi.model), guide=self.svi.guide, num_samples=1,
            parallel=False, params=params
        )
        return predictive(self.svi_state.rng_key, data)

    def train_step(self, data, target, indices, mutables=None):
        if mutables is None:
            mutables = {}

        for site in self.svi.guide.prototype_trace:
            if site not in mutables:
                continue
            mutable = "{}_{}_loc".format(site, self.svi.guide.prefix)
            self.svi_state.mutable_state[mutable]["value"] = mutables[site]

        self.svi_state, loss = self.svi_update(self.svi, self.svi_state, data)

        for name, site in self.svi.guide.prototype_trace.items():
            if site["type"] != "sample" or site["is_observed"]:
                continue
            mutable = "{}_{}_loc".format(name, self.svi.guide.prefix)
            mutables[name] = self.svi_state.mutable_state[mutable]["value"]

        return {"loss": loss}, mutables

    def test_step(self, data, target, indices, mutables=None):
        if mutables is None:
            mutables = {}

        for site in self.svi.guide.prototype_trace:
            if site not in mutables:
                continue
            mutable = "{}_{}_loc".format(site, self.svi.guide.prefix)
            self.svi_state.mutable_state[mutable]["value"] = mutables[site]

        self.svi_state, loss = self.svi_evaluate(self.svi, self.svi_state, data)

        for name, site in self.svi.guide.prototype_trace.items():
            if site["type"] != "sample" or site["is_observed"]:
                continue
            mutable = "{}_{}_loc".format(name, self.svi.guide.prefix)
            mutables[name] = self.svi_state.mutable_state[mutable]["value"]

        return {"loss": loss}, mutables

    def valid_step(self, data, target, indices, mutables=None):
        if mutables is None:
            mutables = {}

        for site in self.svi.guide.prototype_trace:
            if site not in mutables:
                continue
            mutable = "{}_{}_loc".format(site, self.svi.guide.prefix)
            self.svi_state.mutable_state[mutable]["value"] = mutables[site]

        self.svi_state, loss = self.svi_evaluate(self.svi, self.svi_state, data)

        for name, site in self.svi.guide.prototype_trace.items():
            if site["type"] != "sample" or site["is_observed"]:
                continue
            mutable = "{}_{}_loc".format(name, self.svi.guide.prefix)
            mutables[name] = self.svi_state.mutable_state[mutable]["value"]

        return {"loss": loss}, mutables
