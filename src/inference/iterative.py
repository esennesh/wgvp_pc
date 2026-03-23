from functools import cached_property
import jax
import numpyro
from numpyro.infer.autoguide import AutoGuide
from numpyro.infer.initialization import init_to_sample
import optax
from omegaconf.dictconfig import DictConfig

from src.inference.graphical import ParticleTracer
from src.utils import initialize_traces, is_autoguide

class IterativeGuide(AutoGuide):
    def __init__(self, model, guide, optim, tracer, num_iterations: int=1, *,
                 create_plates=None, prefix="auto"):
        self.guide = guide
        self.model = model
        self.num_iterations = num_iterations
        if isinstance(optim, numpyro.optim._NumPyroOptim):
            self.optimizer = optim
        else:
            if isinstance(optim, dict) or isinstance(optim, DictConfig):
                optim = optax.chain(*optim.values())
            self.optimizer = numpyro.optim.optax_to_numpyro(optim)
        self.tracer = tracer

        super().__init__(model, init_loc_fn=init_to_sample, prefix=prefix,
                         create_plates=create_plates)

    def adapt(self, *args, **kwargs):
        if self.prototype_trace is None:
            self._setup_prototype(*args, **kwargs)

        from numpyro.handlers import block, trace

        def hide_guide_params(msg):
            return self.guide.prefix in msg.get("name", "") or\
                   msg["type"] != "param"
        with block(hide_fn=hide_guide_params):
            guide_trace = trace(self.guide).get_trace(*args, **kwargs)

        buffers, params = {}, {}
        model_params = {k: numpyro.param(k) for k in self.model_params}
        for name, site in guide_trace.items():
            if site["type"] not in ["mutable", "param"] or\
               name in self.model_params:
                continue

            if site["type"] == "param":
                params[name] = site["value"]
            else:
                buffers[name] = site["value"]
        buffers.update(**{k: v for k, v in model_params.items()
                          if v is not None})
        optim_state = self.optimizer.init(params)

        iteration_rngs = jax.random.split(numpyro.prng_key(),
                                          self.num_iterations)
        for i, rng in enumerate(iteration_rngs):
            params = self.optimizer.get_params(optim_state)
            with block(expose=self.model_params):
                (ll, aux), grads = self.elbo_grad(buffers, params, rng, *args,
                                                  **kwargs)
            optim_state = self.optimizer.update(grads, optim_state, value=ll)

        return jax.lax.stop_gradient(self.optimizer.get_params(optim_state))

    def __call__(self, *args, adaptation=None, **kwargs):
        if self.prototype_trace is None:
            self._setup_prototype(*args, **kwargs)

        guide = numpyro.handlers.substitute(self.guide, data=adaptation)\
                if adaptation else self.guide
        with numpyro.handlers.block(expose_types=["sample"]):
            return guide(*args, **kwargs)

    @cached_property
    def elbo_grad(self):
        def fn(buffers, params, rng, *args, **kwargs):
            params.update(**jax.lax.stop_gradient(buffers))
            return self.tracer.loss(rng, params, {}, self.model, self.guide,
                                    *args, **kwargs)
        return jax.value_and_grad(fn, argnums=1, has_aux=True)

    @property
    def model_params(self):
        return {name for name, site in self.prototype_trace.items()
                if site["type"] == "param"}

    def sample_posterior(self, rng_key, params, *args, sample_shape=(),
                         **kwargs):
        raise NotImplementedError()

    def _setup_prototype(self, *args, **kwargs):
        self.guide = self.guide(self.model) if is_autoguide(self.guide)\
                     else self.guide
        self.guide._setup_prototype(*args, **kwargs)
        self.prototype_trace = self.guide.prototype_trace
