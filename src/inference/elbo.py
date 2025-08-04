from collections import defaultdict
from jax import Array
import jax
import jax.numpy as jnp
import numpyro
from numpyro.infer.elbo import ELBO, get_nonreparam_deps, guess_max_plate_nesting, MultiFrameTensor
from numpyro.infer.util import get_importance_trace
from numpyro.util import _validate_model, check_model_guide_match
from typing import Dict, Optional

class TraceVectorized_ELBO(ELBO):
    def __init__(self, num_particles: int = 1, particles_dim: Optional[int] = None,
                 sum_sites: bool = True):
        self.particles_dim = particles_dim
        self.sum_sites = sum_sites
        super().__init__(num_particles=num_particles, vectorize_particles=True)

    def loss_with_mutable_state(self, rng_key, param_map, model, guide, *args,
                                **kwargs):
        param_map = param_map.copy()
        mutable_map = {}
        for name, param in list(param_map.items()):
            if isinstance(param, dict) and "value" in param:
                mutable_map[name] = param
                if param["value"].shape[0] != self.num_particles:
                    param["value"] = jnp.broadcast_to(
                        param["value"],
                        (self.num_particles, *param["value"].shape)
                    )
                del param_map[name]

        def single_particle_elbo(rng_key, mutable_map):
            param_map.update(mutable_map)

            model_seed, guide_seed = jax.random.split(rng_key)
            seeded_model = numpyro.handlers.seed(model, model_seed)
            seeded_guide = numpyro.handlers.seed(guide, guide_seed)
            model_trace, guide_trace = get_importance_trace(seeded_model,
                                                            seeded_guide, args,
                                                            kwargs, param_map)

            check_model_guide_match(model_trace, guide_trace)
            _validate_model(model_trace, plate_warning="loose")
            model_mutables = {name: site["value"] for name, site
                              in model_trace.items()
                              if site["type"] == "mutable"}
            guide_mutables = {name: site["value"] for name, site
                              in guide_trace.items()
                              if site["type"] == "mutable"}
            mutable_params = model_mutables | guide_mutables
            model_log_probs = {
                name: site["log_prob"] for name, site in model_trace.items()
                      if site["type"] == "sample"
            }
            guide_log_probs = {
                name: site["log_prob"] for name, site in guide_trace.items()
                      if site["type"] == "sample"
            }
            log_probs = set(model_log_probs).union(guide_log_probs)

            elbos = {name: model_log_probs.get(name, 0.0) -\
                     guide_log_probs.get(name, 0.0) for name in log_probs}
            if self.sum_sites:
                elbos = sum(elbos.values(), start=0.0)
            reparameterized = [site["fn"].has_rsample for name, site in
                               guide_trace.items() if site["type"] == "sample"]
            reparameterized = reparameterized + [site["fn"].has_rsample
                                                 for name, site in
                                                 model_trace.items()
                                                 if site["type"] == "sample"]
            return elbos, mutable_params, all(reparameterized)

        rng_keys = jax.random.split(rng_key, self.num_particles)
        particle_elbos = jax.vmap(single_particle_elbo)
        log_ws, mutable_states = particle_elbos(rng_keys, mutable_map)
        if not mutable_states:
            mutable_states = None
        return {"loss": jnp.mean(-log_ws), "mutable_state": mutable_states}

    def loss(self, rng_key, param_map, model, guide, *args, **kwargs):
        def single_particle_elbo(rng_key):
            model_seed, guide_seed = jax.random.split(rng_key)
            seeded_model = numpyro.handlers.seed(model, model_seed)
            seeded_guide = numpyro.handlers.seed(guide, guide_seed)
            model_trace, guide_trace = get_importance_trace(seeded_model,
                                                            seeded_guide, args,
                                                            kwargs, param_map)

            check_model_guide_match(model_trace, guide_trace)
            _validate_model(model_trace, plate_warning="loose")
            latents = {}
            for name, site in guide_trace.items():
                if site["type"] == "sample" and\
                   not site.get("is_observed", False):
                    latents[name] = site["value"]
            model_deps, guide_deps = get_nonreparam_deps(
                model, guide, args, kwargs, param_map, latents=latents
            )

            log_w = jnp.array(0.0)
            # mapping from non-reparameterizable sample sites to cost terms influenced by each of them
            downstream_costs: Dict[str, MultiFrameTensor] =\
                defaultdict(lambda: MultiFrameTensor())
            for name, site in model_trace.items():
                if site["type"] == "sample":
                    log_w = log_w + site["log_prob"]
                    # add the log_prob to each non-reparam sample site upstream
                    for key in model_deps[name]:
                        downstream_costs[key].add((site["cond_indep_stack"],
                                                   site["log_prob"]))
            for name, site in guide_trace.items():
                if site["type"] == "sample":
                    log_q = site["log_prob"] if site["fn"].has_rsample else\
                            jax.lax.stop_gradient(site["log_prob"])
                    log_w = log_w - log_q
                    # add the -log_prob to each non-reparam sample site upstream
                    for key in guide_deps[name]:
                        downstream_costs[key].add(
                            (site["cond_indep_stack"], -site["log_prob"])
                        )

            for node, cost in downstream_costs.items():
                guide_site = guide_trace[node]
                downstream_cost = cost.sum_to(guide_site["cond_indep_stack"])
                surrogate = guide_site["log_prob"] *\
                            jax.lax.stop_gradient(downstream_cost)
                log_w = log_w + surrogate - jax.lax.stop_gradient(surrogate)

            return log_w

        rng_keys = jax.random.split(rng_key, self.num_particles)
        log_ws = jax.vmap(single_particle_elbo)(rng_keys)
        return jnp.mean(-log_ws)
