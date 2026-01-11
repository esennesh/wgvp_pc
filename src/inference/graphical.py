from abc import ABC
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

class VariationalMixin(ABC):
    def log_weights(self, traces, mutables):
        raise NotImplementedError

    def loss_fn(self, log_ws):
        raise NotImplementedError

class ELBOMixin(VariationalMixin):
    def log_weights(self, traces, mutables):
        return sum(site[1] - site[2] for name, site in traces.items())

    def loss_fn(self, log_ws):
        return -jnp.mean(log_ws, axis=0).sum()

class IwaeMixin(ELBOMixin):
    def loss_fn(self, log_ws):
        return -jax.nn.logmeanexp(log_ws)

class ParticleTracer(ELBOMixin):
    def __init__(self, num_particles: int=1):
        self.num_particles = num_particles

    def __call__(self, rng_key, param_map, particle_params, model, guide,
                 *args, **kwargs):
        param_map = param_map.copy()
        particle_params = jax.tree.map(
            lambda leaf: jnp.broadcast_to(leaf, (self.num_particles,
                                                 *leaf.shape))
                         if leaf.shape[0] != self.num_particles else leaf,
            particle_params
        )

        def single_trace(rng_key, pwise_params, particle=None):
            import functools

            param_map.update(pwise_params)
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

            graph_state = {
                name: (site["value"], site["log_prob"],
                       guide_trace[name]["log_prob"] if name in guide_trace\
                       else jnp.zeros_like(site["log_prob"]),
                       site["is_observed"])
                for name, site in model_trace.items()
                if site["type"] == "sample"
            }
            graph_state.update({
                name: (site["value"], jnp.zeros_like(site["log_prob"]),
                       site["log_prob"], False)
                      for name, site in guide_trace.items()
                      if site["type"] == "sample" and name not in graph_state
            })
            graph_state.update({
                name: (site["value"], 0., 0., False)
                for name, site in model_trace.items()
                if site["type"] == "deterministic"
            })
            graph_state.update({
                name: (site["value"], 0., 0., False)
                for name, site in guide_trace.items()
                if site["type"] == "deterministic"
            })
            mutables = {name: site["value"] for name, site in
                        model_trace.items() if site["type"] == "mutable"}

            return graph_state, mutables

        rng_keys = random.split(rng_key, self.num_particles)
        particles = jnp.arange(self.num_particles)
        particle_traces = jax.vmap(single_trace)
        return particle_traces(rng_keys, particle_params, particle=particles)

    def guided_log_weights(self, rng_key, param_map, particle_params, model,
                           guide, *args, **kwargs):
        traces = self(rng_key, param_map, particle_params, model, guide, *args,
                      **kwargs)
        return {k: (log_p, log_q) for k, (_, log_p, log_q, _) in traces.items()
                if log_p is not 0.}

    def log_probs(self, model, params, particle_params, traces, *args,
                  **kwargs):
        params = params.copy()
        particle_params = jax.tree.map(
            lambda leaf: jnp.broadcast_to(leaf, (self.num_particles,
                                                 *leaf.shape))
                         if leaf.shape[0] != self.num_particles else leaf,
            particle_params
        )

        def single_log_prob(pwise_params, trace, particle=None):
            import functools

            params.update(pwise_params)
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
        return particle_log_probs(particle_params, traces, particle=particles)

    def loss(self, *args, **kwargs):
        traces, mutables = self(*args, **kwargs)
        for k, v in traces.items():
            is_observed = jnp.broadcast_to(jnp.expand_dims(v[-1], axis=-1),
                                           v[0].shape[:2])
            traces[k] = v[:-1] + (is_observed,)
        log_ws = self.log_weights(traces, mutables)
        return self.loss_fn(log_ws), {"log_w": log_ws.sum(axis=-1),
                                      "mutables": mutables, "trace": traces}

    def setup(self, guide_deps, model_deps, guide_trace, model_trace):
        pass

class ELBOTracer(ParticleTracer):
    def __init__(self, num_particles: int=1):
        super().__init__(num_particles=num_particles)
        self._guide_deps, self._model_deps = None, None
        self._guide_properties, self._model_properties = {}, {}

    def log_weights(self, traces, mutables):
        if jax.tree.leaves(mutables):
            return super().log_weights(traces, mutables)
        log_ws = jnp.array(0.0)
        # mapping from non-reparameterizable sample sites to cost terms
        # influenced by each of them
        downstream_costs: Dict[str, MultiFrameTensor] =\
            defaultdict(lambda: MultiFrameTensor())
        for name, site in traces.items():
            log_ws = log_ws + site[1]
            for key in self._model_deps.get(name, []):
                downstream_costs[key].add((
                    self._model_properties[name]["cond_indep_stack"],
                    site[1]
                ))
            if name in self._guide_properties:
                log_q = site[2]
                if not self._guide_properties[name]["reparameterized"]:
                    log_q = jax.lax.stop_gradient(log_q)
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
            advantage = downstream_cost - downstream_cost.mean(axis=0)
            surrogate = traces[node][2] * jax.lax.stop_gradient(advantage)
            log_ws = log_ws + surrogate - jax.lax.stop_gradient(surrogate)
        return log_ws

    def loss_fn(self, log_ws):
        reparameterized = all(site["reparameterized"] for site
                              in self._guide_properties.values())
        if reparameterized:
            return super().loss_fn(log_ws)
        return -(jnp.sum(log_ws, axis=0) / (log_ws.shape[0] - 1)).sum()

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
