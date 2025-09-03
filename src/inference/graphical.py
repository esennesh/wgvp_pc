from collections import defaultdict
import jax
import jax.numpy as jnp
import jax.random as random
import numpyro
from numpyro.infer.elbo import MultiFrameTensor
from numpyro.infer.util import compute_log_probs, get_importance_trace
from numpyro._typing import Message
from numpyro.util import _validate_model, check_model_guide_match

from typing import Dict

def configure_sample(msg: Message, /, **kwargs) -> Dict:
    return kwargs

class ParticleTracer:
    def __init__(self, num_particles: int=1):
        self.num_particles = num_particles

    def __call__(self, rng_key, param_map, model, guide, *args, **kwargs):
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

        def single_trace(rng_key, mutable_map, particle=None):
            import functools

            param_map.update(mutable_map)
            particle_guide, particle_model = guide, model

            model_seed, guide_seed = random.split(rng_key)
            if particle is not None:
                particle_guide = numpyro.handlers.infer_config(
                    particle_guide,
                    functools.partial(configure_sample, k=particle)
                )
                particle_model = numpyro.handlers.infer_config(
                    particle_model,
                    functools.partial(configure_sample, k=particle)
                )
            seeded_model = numpyro.handlers.seed(particle_model, model_seed)
            seeded_guide = numpyro.handlers.seed(particle_guide, guide_seed)
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

            graph_state = {
                name: (model_trace[name]["value"],
                       model_log_probs.get(name, 0.0),
                       guide_log_probs.get(name, 0.0),
                       not model_trace[name].get("is_observed", False))
                for name in log_probs
            }
            graph_state.update({
                name: (site["value"], 0., 0., False)
                for name, site in model_trace.items()
                if site["type"] == "deterministic"
            })

            return graph_state, model_mutables

        rng_keys = random.split(rng_key, self.num_particles)
        particles = jnp.arange(self.num_particles)
        particle_traces = jax.vmap(single_trace)
        trace, mutables  = particle_traces(rng_keys, mutable_map,
                                           particle=particles)
        return {"mutable_state": mutables, "trace": trace}

    def log_probs(self, model, params, traces, *args, **kwargs):
        params = params.copy()
        mutable_map = {}
        for name, param in list(params.items()):
            if isinstance(param, dict) and "value" in param:
                mutable_map[name] = param
                if param["value"].shape[0] != self.num_particles:
                    param["value"] = jnp.broadcast_to(
                        param["value"],
                        (self.num_particles, *param["value"].shape)
                    )
                del params[name]

        def single_log_prob(mutable_map, trace, particle=None):
            import functools

            params.update(mutable_map)
            params.update(trace)
            particle_model = model
            if particle is not None:
                particle_model = numpyro.handlers.infer_config(
                    particle_model,
                    functools.partial(configure_sample, k=particle)
                )
            log_ps, _ = compute_log_probs(model, args, kwargs, params)
            return log_ps

        particles = jnp.arange(self.num_particles)
        particle_log_probs = jax.vmap(single_log_prob)
        return particle_log_probs(mutable_map, traces, particle=particles)

    def loss(self, *args, **kwargs):
        objective = super().loss(*args, **kwargs)
        log_ws = sum(site[1] - site[2] for name, site in
                     objective["trace"].items())
        return {"loss": jnp.mean(-log_ws), "log_w": log_ws, **objective}

    def setup(self, guide_deps, model_deps, guide_trace, model_trace):
        pass

class ELBOTracer(ParticleTracer):
    def __init__(self, num_particles: int=1):
        super().__init__(num_particles=num_particles)
        self._guide_deps, self._model_deps = None, None
        self._guide_properties, self._model_properties = {}, {}

    def loss(self, *args, **kwargs):
        objective = self(*args, **kwargs)
        if objective["mutable_state"]:
            log_ws = sum(jnp.sum(site[1] - site[2], axis=-1) for name, site in
                         objective["trace"].items())
        else:
            log_ws = jnp.array(0.0)
            # mapping from non-reparameterizable sample sites to cost terms influenced by each of them
            downstream_costs: Dict[str, MultiFrameTensor] =\
                defaultdict(lambda: MultiFrameTensor())
            for name, site in objective["trace"].items():
                log_ws = log_ws + jnp.sum(site[1], axis=-1)
                for key in self._model_deps[name]:
                    downstream_costs[key].add((
                        self._model_properties[name]["cond_indep_stack"],
                        site[1]
                    ))
                if name in self._guide_properties:
                    log_q = jnp.sum(site[2], axis=-1)
                    if self._guide_properties[name]["reparameterized"]:
                        log_q = jax.lax.stop_gradient(loq_q)
                    log_ws = log_ws - log_q
                    for key in self._guide_deps[name]:
                        downstream_costs[key].add((
                            self._guide_properties[name]["cond_indep_stack"],
                            -site[2]
                        ))

            for node, cost in downstream_costs.items():
                downstream_cost = cost.sum_to(
                    self._guide_properties[node]["cond_indep_stack"]
                )
                log_q = objective["trace"][node][2]
                surrogate = jnp.sum(
                    log_q * jax.lax.stop_gradient(downstream_cost), axis=-1
                )
                log_ws = log_ws + surrogate - jax.lax.stop_gradient(surrogate)

        objective["log_w"] = log_ws
        return jnp.mean(-log_ws), objective

    def setup(self, guide_deps, model_deps, guide_trace, model_trace):
        self._guide_deps, self._model_deps = guide_deps, model_deps
        for name, site in guide_trace.items():
            if site["type"] != "sample":
                continue

            self._guide_properties[name] = {
                "cond_indep_stack": site["cond_indep_stack"],
                "reparameterized": site["fn"].has_rsample
            }

        for name, site in model_trace.items():
            if site["type"] != "sample":
                continue

            self._model_properties[name] = {
                "cond_indep_stack": site["cond_indep_stack"],
            }
