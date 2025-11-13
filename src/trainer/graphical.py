from functools import cached_property, partial
import itertools
import jax
import jax.random as random
import networkx as nx
import numpyro
from numpyro.infer.autoguide import AutoGuide
from numpyro.infer.elbo import get_nonreparam_deps
from numpyro.infer import Predictive

from typing import Any, Dict

from .para import ParaMonad
from src.data import DataModule
from src.inference.graphical import ParticleTracer
from src.utils import initialize_traces, uncondition

def _is_autoguide(g):
    import abc
    if isinstance(g, abc.ABCMeta) and issubclass(g, AutoGuide):
        return True
    if isinstance(g, partial):
        if isinstance(g.func, abc.ABCMeta) and issubclass(g.func, AutoGuide):
            return True
    return False

class GraphicalImportancePara(ParaMonad):
    def __init__(self, data_shape, guide, tracer: ParticleTracer, lr,
                 model, rng):
        if _is_autoguide(guide):
            guide = guide(model)
        if not isinstance(rng, jax.Array):
            rng = random.key(rng)
        self._constrain_fn = None
        self._graph = nx.DiGraph()
        self._guide = guide
        self._lr = lr
        self._model = model
        self._mutable_state = {}
        self.optim_state = None
        self.optimizer = numpyro.optim.Adam(step_size=lr)
        self._relations = {}
        self._rng = rng
        self._tracer = tracer

    def __call__(self, *args, stage="train", **kwargs):
        self._rng, rng = random.split(self.rng)
        trace, mutables = self.tracer(rng, self.parameters,
                                      uncondition(self.model),
                                      self.guide, *args, **kwargs)
        return {k: v[0] for k, v in trace.items()}

    @cached_property
    def _evaluate(self):
        @jax.jit
        def fn(data, params, rng):
            next_rng, rng = random.split(rng)
            mutable_map = jax.lax.stop_gradient(self.mutable_state)
            loss, state = self.tracer.loss(rng, params, mutable_map, self.model,
                                           self.guide, data)
            return loss, next_rng, state
        return fn

    @property
    def guide(self):
        return self._guide

    def load(self, checkpoint: Dict[str, Any]):
        self._mutable_state = checkpoint["mutable_state"]
        self.optim_state = checkpoint["optim_state"]

    @property
    def mutable_state(self):
        return self._mutable_state

    @property
    def tracer(self):
        return self._tracer

    @property
    def model(self):
        return self._model

    @property
    def parameters(self):
        return self._constrain_fn(self.optimizer.get_params(self.optim_state))

    def render_model(self, filename=None, render_distributions=False,
                     render_params=False):
        from numpyro.infer.inspect import (generate_graph_specification,
                                           render_graph)
        graph_spec = generate_graph_specification(self._relations,
                                                  render_params=render_params)
        graph = render_graph(graph_spec,
                             render_distributions=render_distributions)

        if filename is not None:
            filename = Path(filename)
            # remove leading period from suffix
            filename_without_suffix = filename.with_suffix("")
            graph.render(
                filename_without_suffix,
                view=False,
                cleanup=True,
                format=filename.suffix[1:],
            )

        return graph

    @property
    def rng(self):
        return self._rng

    def save(self):
        return {"mutable_state": self.mutable_state,
                "optim_state": self.optim_state}

    def _setup_graph(self, *args, **kwargs):
        state = initialize_traces(self.model, self.guide, self._rng, {}, *args,
                                  **kwargs)
        self._constrain_fn, self._mutable_state, self._rng =\
            state.constrain_fn, state.mutables, state.rng
        guide_trace, model_trace = state.guide_trace, state.model_trace

        latents = {}
        for name, site in guide_trace.items():
            if site["type"] == "sample" and not site.get("is_observed", False):
                latents[name] = site["value"]

        from numpyro.handlers import replay, seed
        self._rng, model_seed, guide_seed = random.split(self._rng, 3)
        init_guide = replay(seed(self.guide, guide_seed), guide_trace)
        init_model = replay(seed(self.model, model_seed), model_trace)
        model_deps, guide_deps = get_nonreparam_deps(init_model, init_guide,
                                                     args, kwargs, state.params,
                                                     latents=latents)
        self.tracer.setup(guide_deps, model_deps, guide_trace, model_trace)

        from src.utils import get_model_relations
        self._relations = get_model_relations(init_model, args, kwargs)
        for var, parents in self._relations["sample_sample"].items():
            self._graph.add_node(var)
            for par in parents:
                self._graph.add_edge(par, var)
        return state

    def setup_step(self, datamodule: DataModule):
        for batch in getattr(datamodule, stage + "_dataloader")():
            data = batch[0]
            break

        state = self._setup_graph(data)
        if not self.optim_state:
            self.optim_state = self.optimizer.init(state.params)

        return state.guide_trace, state.model_trace

    @cached_property
    def _update(self):
        @jax.jit
        def fn(data, optim_state, rng):
            next_rng, rng = random.split(rng)
            def loss_fn(params):
                mutable_map = jax.lax.stop_gradient(self.mutable_state)
                return self.tracer.loss(rng, params, mutable_map, self.model,
                                        self.guide, data)
            (loss, state), optim_state = self.optimizer.eval_and_update(
                loss_fn, optim_state
            )
            return loss, optim_state, next_rng, state
        return fn

    def test_step(self, data, *args, **kwargs):
        loss, self._rng, state = self._evaluate(data, self.parameters, self.rng)
        self._mutable_state = state["mutables"]
        return {"loss": loss, "log_w": state["log_w"]}

    def train_step(self, data, *args, **kwargs):
        loss, self.optim_state, self._rng, state = self._update(
            data, self.optim_state, self.rng
        )
        self._mutable_state = state["mutables"]
        return {"loss": loss, "log_w": state["log_w"]}

    def valid_step(self, data, *args, **kwargs):
        loss, self._rng, state = self._evaluate(data, self.parameters, self.rng)
        self._mutable_state = state["mutables"]
        return {"loss": loss, "log_w": state["log_w"]}
