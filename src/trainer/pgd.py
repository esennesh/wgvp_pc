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
