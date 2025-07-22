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
from numpyro.infer.util import log_density
from pytrie import SortedStringTrie as Trie
from typing import Tuple

class AutoLangevin(AutoGuide):
    def __init__(self, model, *, create_plates=None, lr=1e-4, prefix="auto"):
        self._event_dims = {}
        self._grad_log_densities = {}
        self._lr = lr
        super().__init__(model, init_loc_fn=init_to_sample, prefix=prefix,
                         create_plates=create_plates)

    def _setup_prototype(self, *args, **kwargs):
        super()._setup_prototype(*args, **kwargs)

        for name, site in self.prototype_trace.items():
            if site["type"] != "sample" or site["is_observed"]:
                continue

            event_dim = (
                site["fn"].event_dim
                + jnp.ndim(self._init_locs[name])
                - jnp.ndim(site["value"])
            )
            self._event_dims[name] = event_dim

            # If subsampling, repeat init_value to full size.
            for frame in site["cond_indep_stack"]:
                full_size = self._prototype_frame_full_sizes[frame.name]
                if full_size != frame.size:
                    dim = frame.dim - event_dim
                    self._init_locs[name] = periodic_repeat(
                        self._init_locs[name], full_size, dim
                    )

            def var_log_density(site):
                def fn(val, *args, **kwargs):
                    params = {site: val}
                    with numpyro.handlers.block(expose_types=["param"]):
                        return log_density(self.model, args, kwargs, params)[0]
                return fn

            self._grad_log_densities[name] = jax.grad(var_log_density(name),
                                                      argnums=0)

    def __call__(self, *args, **kwargs):
        if self.prototype_trace is None:
            # run model to inspect the model structure
            self._setup_prototype(*args, **kwargs)

        plates = self._create_plates(*args, **kwargs)
        result = {}
        for name, site in self.prototype_trace.items():
            if site["type"] != "sample" or site["is_observed"]:
                continue

            event_dim = self._event_dims[name]
            init_loc = self._init_locs[name]
            with ExitStack() as stack:
                for frame in site["cond_indep_stack"]:
                    stack.enter_context(plates[frame.name])

                site_loc = numpyro.primitives.mutable(
                    "{}_{}_loc".format(name, self.prefix), {"value": init_loc}
                )
                update = jax.lax.stop_gradient(
                    self._grad_log_densities[name](site_loc["value"],
                                                   *args, **kwargs)
                )

                site_fn = dist.Normal(site_loc["value"] + self._lr * update,
                                      math.sqrt(2 * self._lr)).to_event(event_dim)
                if site["fn"].support is constraints.real or (
                    isinstance(site["fn"].support, constraints.independent)
                    and site["fn"].support.base_constraint is constraints.real
                ):
                    result[name] = numpyro.sample(name, site_fn)
                else:
                    with helpful_support_errors(site):
                        transform = biject_to(site["fn"].support)
                    guide_dist = dist.TransformedDistribution(site_fn, transform)
                    result[name] = numpyro.sample(name, guide_dist)

                site_loc["value"] = result[name]

        return result

    def sample_posterior(self, rng_key, params, *args, sample_shape=(),
                         **kwargs):
        samples = {}
        with numpyro.infer.handlers.seed(rng_seed=rng_key):
            for site in self.prototype_trace:
                if site["type"] != "sample" or site["is_observed"]:
                    continue

                init_loc = self._init_locs[site]
                site_loc = numpyro.primitives.mutable(
                    "{}_{}_loc".format(name, self.prefix), {"value": init_loc}
                )
                update = self._grad_log_densities[name](site_loc["value"],
                                                        *args, **kwargs)
                loc = site_loc + self._lr * update
                density = dist.Normal(loc, (2 * self._lr).sqrt())
                samples[site] = numpyro.sample(site,
                                               density.expand_by(sample_shape))
                site_loc["value"] = samples[site]

        return self._constrain(samples)
