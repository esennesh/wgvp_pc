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
from numpyro.infer import ELBO, Predictive
from numpyro.infer.util import log_density
from pytrie import SortedStringTrie as Trie
from typing import Any, Dict, Tuple

from .para import BatchParameters
from src.data import DataModule
from src.inference import AutoLangevin
from src.utils import uncondition
from .svi import SviPara

class PgdPara(SviPara):
    def __init__(self, data_shape, lr, model, elbo: ELBO, rng, lrq=1e-4):
        self.test_particles, self.train_particles = None, None
        super().__init__(data_shape, AutoLangevin(model, lr=lrq), lr, model,
                         elbo, rng)

    def __call__(self, data, targets, indices, stage="train"):
        mutables = getattr(self, stage + "_particles").get_parameters(indices)

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

    def load(self, checkpoint: Dict[str, Any]):
        super().load(checkpoint)
        self.test_particles = BatchParameters.unpickle(
            checkpoint["test_particles"]
        )
        self.train_particles = BatchParameters.unpickle(
            checkpoint["train_particles"]
        )

    def save(self):
        state = super().save()
        return {**state, "test_particles": self.test_particles.pickle(),
                "train_particles": self.train_particles.pickle()}

    def setup_step(self, datamodule: DataModule):
        if self.test_particles is None:
            self.test_particles = BatchParameters(len(datamodule.data_test))
        if self.train_particles is None:
            self.train_particles = BatchParameters(len(datamodule.data_train) +\
                                                   len(datamodule.data_val),
                                                   axis=1)
        return super().setup_step(datamodule)

    def train_step(self, data, target, indices):
        for site in self.svi.guide.prototype_trace:
            if site not in self.train_particles:
                continue
            mutable = "{}_{}_loc".format(site, self.svi.guide.prefix)
            self.svi_state.mutable_state[mutable]["value"] =\
                self.train_particles.get_parameter(indices, site)

        self.svi_state, loss = self.svi_update(self.svi, self.svi_state, data)

        for name, site in self.svi.guide.prototype_trace.items():
            if site["type"] != "sample" or site["is_observed"]:
                continue
            mutable = "{}_{}_loc".format(name, self.svi.guide.prefix)
            self.train_particles.set_parameter(
                indices, name, self.svi_state.mutable_state[mutable]["value"]
            )

        return {"loss": loss}

    def test_step(self, data, target, indices):
        for site in self.svi.guide.prototype_trace:
            if site not in self.test_particles:
                continue
            mutable = "{}_{}_loc".format(site, self.svi.guide.prefix)
            self.svi_state.mutable_state[mutable]["value"] =\
                self.test_particles.get_parameter(indices, site)

        self.svi_state, loss = self.svi_evaluate(self.svi, self.svi_state, data)

        for name, site in self.svi.guide.prototype_trace.items():
            if site["type"] != "sample" or site["is_observed"]:
                continue
            mutable = "{}_{}_loc".format(name, self.svi.guide.prefix)
            self.test_particles.set_parameter(
                indices, name, self.svi_state.mutable_state[mutable]["value"]
            )

        return {"loss": loss}

    def valid_step(self, data, target, indices):
        for site in self.svi.guide.prototype_trace:
            if site not in self.train_particles:
                continue
            mutable = "{}_{}_loc".format(site, self.svi.guide.prefix)
            self.svi_state.mutable_state[mutable]["value"] =\
                self.train_particles.get_parameter(indices, site)

        self.svi_state, loss = self.svi_evaluate(self.svi, self.svi_state, data)

        for name, site in self.svi.guide.prototype_trace.items():
            if site["type"] != "sample" or site["is_observed"]:
                continue
            mutable = "{}_{}_loc".format(name, self.svi.guide.prefix)
            self.train_particles.set_parameter(
                indices, name, self.svi_state.mutable_state[mutable]["value"]
            )

        return {"loss": loss}
