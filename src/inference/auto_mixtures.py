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

class AutoMixtureProposal(AutoGuide):
    def __init__(self, model, num_particles=1, *, create_plates=None,
                 prefix="auto"):
        self._event_dims = {}
        self._init_locs = {}
        self._num_particles = num_particles
        super().__init__(model, init_loc_fn=init_to_sample, prefix=prefix,
                         create_plates=create_plates)

    def _setup_prototype(self, *args, **kwargs):
        from numpyro.handlers import block, seed, trace
        seeded_model = seed(self.model, numpyro.prng_key())
        with block(expose_types=["param"]):
            self.prototype_trace = trace(seeded_model).get_trace(*args,
                                                                 **kwargs)

        for name, site in self.prototype_trace.items():
            if site["type"] == "sample":
                if site["is_observed"]:
                    continue

                self._event_dims[name] = site["fn"].event_dim
                self._init_locs[name] = site["value"]

                # If subsampling, repeat init_value to full size.
                for frame in site["cond_indep_stack"]:
                    if frame.name in self._prototype_frames:
                        assert frame == self._prototype_frames[frame.name], (
                            f"The plate {frame.name} has inconsistent dim or size. Please check your model again."
                        )
                    else:
                        self._prototype_frames[frame.name] = frame
                    full_size = self._prototype_frame_full_sizes[frame.name]
                    if full_size != frame.size:
                        dim = frame.dim - event_dim
                        self._init_locs[name] = periodic_repeat(
                            self._init_locs[name], full_size, dim
                        )
            elif site["type"] == "plate":
                self._prototype_frame_full_sizes[name] = site["args"][0]

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
            with ExitStack() as stack:
                for frame in site["cond_indep_stack"]:
                    stack.enter_context(plates[frame.name])

                q = self._mixture(name, site["fn"])
                result[name] = numpyro.sample(name, q.to_event(event_dim))

        return result

    def _mixture(self, name, site_fn):
        event_dim = self._event_dims[name]
        init_loc = self._init_locs[name]
        site_shape = list(init_loc.shape)
        site_shape.insert(event_dim, self.num_particles)

        weights_shape = init_loc.shape[:event_dim] + (self.num_particles,)
        weights = numpyro.primitives.param(
            "{}_{}_weights".format(name, self.prefix),
            init_value=jnp.zeros(weights_shape)
        )
        indices = numpyro.sample("{}_{}_components".format(name, self.prefix),
                                 dist.CategoricalLogits(weights),
                                 infer={"is_auxiliary": True})
        indexer = functools.partial(
            jax.vmap(lambda index, component: component[index]),
            indices
        )

        if site_fn.support is constraints.real or (
            isinstance(site_fn.support, constraints.independent)
            and site_fn.support.base_constraint is constraints.real
        ):
            loc = numpyro.primitives.param(
                "{}_{}_loc".format(name, self.prefix),
                init_value=jnp.broadcast_to(jnp.expand_dims(init_loc,
                                                            event_dim),
                                            site_shape)
            )
            log_scale = numpyro.primitives.param(
                "{}_{}_log_scale".format(name, self.prefix),
                init_value=jnp.zeros(site_shape)
            )
            return dist.Normal(indexer(loc), indexer(jnp.exp(log_scale)))
        elif site_fn.support is constraints.nonnegative_integer or (
            isinstance(site_fn.support, constraints.independent)
            and site_fn.support.base_constraint is\
                constraints.nonnegative_integer
        ):
            init_loc = jnp.stack((init_loc,) * self.num_particles,
                                 axis=event_dim)
            log_rate = numpyro.primitives.param(
                "{}_{}_log_rate".format(name, self.prefix),
                init_value=jnp.log(init_loc + 1e-5) # rate >= 1e-5
            )
            return dist.Poisson(indexer(jnp.exp(log_rate)))
        else:
            transform = biject_to(site_fn.support)
            loc = numpyro.primitives.param(
                "{}_{}_loc".format(name, self.prefix),
                init_value=jnp.broadcast_to(jnp.expand_dims(init_loc,
                                                            event_dim),
                                            site_shape),
            )
            log_scale = numpyro.primitives.param(
                "{}_{}_log_scale".format(name, self.prefix),
                init_value=jnp.zeros(site_shape)
            )
            return dist.TransformedDistribution(
                dist.Normal(indexer(loc), indexer(jnp.exp(log_scale))),
                transform
            )

    @property
    def num_particles(self):
        return self._num_particles

    def sample_posterior(self, rng_key, params, *args, sample_shape=(),
                         **kwargs):
        samples = {}
        with numpyro.infer.handlers.seed(rng_seed=rng_key):
            for site in self.prototype_trace:
                if site["type"] != "sample" or site["is_observed"]:
                    continue

                q = self._mixture(name, site["fn"])
                samples[site] = numpyro.sample(site, q.expand_by(sample_shape))

        return self._constrain(samples)
