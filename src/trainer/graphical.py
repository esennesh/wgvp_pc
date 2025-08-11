from functools import cached_property, partial
import itertools
import jax
import jax.numpy as jnp
import jax.random as random
import numpyro
from numpyro.distributions import constraints
from numpyro.distributions.transforms import biject_to
from numpyro.infer.elbo import get_nonreparam_deps
from numpyro.infer.util import (get_importance_trace, helpful_support_errors,
                                transform_fn)
from typing import Any, Dict

from .para import ParaMonad
from src.inference.graphical import ParticleTracer
from src.utils import uncondition

class GraphicalImportancePara(ParaMonad):
    def __init__(self, data_shape, guide, log_weights: ParticleTracer, lr,
                 model, rng):
        if not isinstance(rng, jax.Array):
            rng = random.key(rng)
        self._constrain_fn = None
        self._guide = guide
        self._log_weights = log_weights
        self._lr = lr
        self._model = model
        self.mutable_state = None
        self.optim_state = None
        self.optimizer = numpyro.optim.Adam(step_size=lr)
        self._rng = rng
        self.trace = None

    def __call__(self, *args, **kwargs):
        self._rng, rng = random.split(self.rng)
        predictive = Predictive(
            uncondition(self.model), guide=self.guide,
            num_samples=self.log_weights.num_particles, batch_ndims=None,
            parallel=False, params=self.parameters
        )
        return predictive(rng, *args, **kwargs)

    @cached_property
    def _evaluate(self):
        @jax.jit
        def fn(data, mutables, params, rng):
            next_rng, rng = random.split(rng)
            params.update(jax.lax.stop_gradient(mutables))
            loss, state = self.log_weights.loss(rng, params, self.model,
                                                self.guide, data)
            return loss, next_rng, state
        return fn

    @property
    def guide(self):
        return self._guide

    def load(self, checkpoint: Dict[str, Any]):
        self.mutable_state = checkpoint["mutable_state"]
        self.optim_state = checkpoint["optim_state"]

    @property
    def log_weights(self):
        return self._log_weights

    @property
    def model(self):
        return self._model

    @property
    def parameters(self):
        return self._constrain_fn(self.optimizer.get_params(self.optim_state))

    @property
    def rng(self):
        return self._rng

    def save(self):
        return {"mutable_state": self.mutable_state,
                "optim_state": self.optim_state}

    def setup_step(self, data, *args, **kwargs):
        from numpyro.handlers import replay, seed, substitute, trace
        self._rng, model_seed, guide_seed = random.split(self._rng, 3)
        init_model = seed(self.model, model_seed)
        init_guide = seed(self.guide, guide_seed)
        model_trace, guide_trace = get_importance_trace(init_model, init_guide,
                                                        (data,), kwargs, {})

        params, inv_transforms, self._mutable_state = {}, {}, {}
        for site in itertools.chain(guide_trace.values(), model_trace.values()):
            if site["type"] == "param":
                constraint = site["kwargs"].pop("constraint", constraints.real)
                with helpful_support_errors(site):
                    transform = biject_to(constraint)
                inv_transforms[site["name"]] = transform
                params[site["name"]] = transform.inv(site["value"])
            elif site["type"] == "mutable":
                self._mutable_state[site["name"]] = site["value"]

        if not self.mutable_state:
            self._mutable_state = None
        self._constrain_fn = partial(transform_fn, inv_transforms)
        # we convert weak types like float to float32/float64
        # to avoid recompiling body_fn later
        params, self._mutable_state = jax.tree.map(
            lambda x: jax.lax.convert_element_type(x, jnp.result_type(x)),
            (params, self._mutable_state),
        )
        if not self.optim_state:
            self.optim_state = self.optimizer.init(params)

        latents = {}
        for name, site in guide_trace.items():
            if site["type"] == "sample" and\
               not site.get("is_observed", False):
                latents[name] = site["value"]
        model_deps, guide_deps = get_nonreparam_deps(
            init_model, init_guide, (data,), kwargs, params, latents=latents
        )
        self.log_weights.setup(guide_deps, model_deps, guide_trace, model_trace)

    @cached_property
    def _update(self):
        @jax.jit
        def fn(data, mutables, optim_state, rng):
            next_rng, rng = random.split(rng)
            def loss_fn(params):
                params.update(jax.lax.stop_gradient(mutables))
                return self.log_weights.loss(rng, params, self.model,
                                             self.guide, data)
            (loss, state), optim_state = self.optimizer.eval_and_update(
                loss_fn, optim_state
            )
            return loss, optim_state, next_rng, state
        return fn

    def test_step(self, data, *args, mutables=None):
        if mutables is None:
            mutables = {}
        loss, self._rng, state = self._evaluate(data, mutables, self.parameters,
                                                self.rng)
        return {"loss": loss, "log_w": state["log_w"]}, state["mutable_state"]

    def train_step(self, data, *args, mutables=None):
        if mutables is None:
            mutables = {}
        loss, self.optim_state, self._rng, state = self._update(
            data, mutables, self.optim_state, self.rng
        )
        self._mutable_state = state["mutable_state"]
        self.trace = state["trace"]
        return {"loss": loss, "log_w": state["log_w"]}, state["mutable_state"]

    def valid_step(self, data, *args, mutables=None):
        if mutables is None:
            mutables = {}
        loss, self._rng, state = self._evaluate(data, mutables, self.parameters,
                                                self.rng)
        return {"loss": loss, "log_w": state["log_w"]}, state["mutable_state"]
