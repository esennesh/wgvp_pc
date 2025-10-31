from functools import cached_property, partial
import itertools
import jax
from jax.example_libraries.optimizers import (JoinPoint, pack_optimizer_state,
                                              unpack_optimizer_state)
import jax.numpy as jnp
import jax.random as random
import networkx as nx
import numpyro
from numpyro.distributions import constraints
from numpyro.distributions.transforms import biject_to
from numpyro.infer.elbo import get_nonreparam_deps
from numpyro.infer import Predictive
from numpyro.infer.util import (get_importance_trace, helpful_support_errors,
                                transform_fn)
from typing import Any, Dict

from .graphical import GraphicalImportancePara
from .para import BatchParameters, ParaMonad
from src.data import DataModule
from src.inference.graphical import ParticleTracer
from src.utils import initialize_traces

class BatchVariationalPara(GraphicalImportancePara):
    def __init__(self, data_shape, guide, local_lr, lr, model, rng,
                 tracer: ParticleTracer, batch_axis=0, plate="batch"):
        self._batch_axis = batch_axis
        self.local_optimizer = numpyro.optim.Adam(step_size=local_lr)
        self.local_optim_state = None
        self.local_parameters = None
        self._plate = plate
        self.stage_parameters, self.stage_buffers = {}, {}
        super().__init__(data_shape, guide, tracer, lr, model, rng)

    def __call__(self, data, indices, *args, stage="train", **kwargs):
        from src.utils import uncondition

        global_optim, local_optim, buffers = self.load_batch(indices, stage)
        params, particle_params = {}, {}
        for param, value in self._parameters(global_optim).items():
            if param in self._particle_params:
                particle_params[param] = value
            else:
                params[param] = value
        for param, value in self._parameters(local_optim, local=True).items():
            if param in self._particle_params:
                particle_params[param] = value
            else:
                params[param] = value
        for buffer, value in buffers.items():
            value = jax.lax.stop_gradient(value)
            if buffer in self._particle_params:
                particle_params[buffer] = value
            else:
                params[buffer] = value

        self._rng, rng = random.split(self.rng)
        trace, mutables = self.tracer(rng, params, particle_params,
                                      uncondition(self.model), self.guide, data,
                                      **kwargs)
        return {k: v[0] for k, v in trace.items()}

    def load(self, checkpoint: Dict[str, Any]):
        super().load(checkpoint)
        self._batch_axis = checkpoint["batch_axis"]
        self.local_optim_state = checkpoint["local_optim_state"]
        self.local_parameters = checkpoint["local_parameters"]
        self._plate = checkpoint["plate"]
        self.stage_parameters = checkpoint["stage_parameters"]
        self.stage_buffers = checkpoint["stage_buffers"]

    def load_batch(self, indices, stage: str="train"):
        if stage == "valid":
            stage = "train"

        buffers = self._buffer_state.copy()
        local_buffers = self.stage_buffers[stage].get_parameters(indices)
        for key, buffer in local_buffers.items():
            val = buffer
            for addr in reversed(key.split('/')):
                val = {addr: val}
            buffers.update(val)
        local_params = self.stage_parameters[stage].get_parameters(indices)
        local_state = unpack_optimizer_state(self.local_optim_state[1])
        local_state = pack_optimizer_state({
            param: JoinPoint(jax.tree.map_with_path(
                lambda path, _: local_params[
                    jax.tree_util.keystr((param, *path), separator='/')
                ], join.subtree
            )) for param, join in local_state.items()
        })
        local_optim_state = (self.local_optim_state[0], local_state)
        return (self.optim_state, local_optim_state, buffers)

    def _parameters(self, optim_state, local=False):
        optimizer = self.local_optimizer if local else self.optimizer
        return self._constrain_fn(optimizer.get_params(optim_state))

    def save(self):
        state = super().save()
        return {**state, "batch_axis": self._batch_axis,
                "local_optim_state": self.local_optim_state,
                "local_parameters": self.local_parameters, "plate": self._plate,
                "stage_parameters": self.stage_parameters,
                "stage_buffers": self.stage_buffers}

    def save_batch(self, indices, local_optim_state, local_buffers,
                   global_optim_state=None, stage: str="train"):
        saver = "train" if stage == "valid" else stage

        self.local_optim_state = local_optim_state
        local_params = unpack_optimizer_state(local_optim_state[1])
        for param, join in local_params.items():
            for k, v in jax.tree.leaves_with_path(join.subtree):
                key = param + "/" + jax.tree_util.keystr(k, separator='/')
                if param in self._particle_params and\
                   v.shape[0] != self.tracer.num_particles:
                    v = jnp.broadcast_to(v, (self.tracer.num_particles,
                                             *v.shape))
                self.stage_parameters[saver].set_parameter(indices, key, v)

        for buffer, subtree in local_buffers.items():
            for k, v in jax.tree.leaves_with_path(subtree):
                if buffer in self._particle_params and\
                   v.shape[0] != self.tracer.num_particles:
                    v = jnp.broadcast_to(v, (self.tracer.num_particles,
                                             *v.shape))

                key = buffer + '/' + jax.tree_util.keystr(k, separator='/',
                                                          simple=True)
                self.stage_buffers[saver].set_parameter(indices, key, v)

        if global_optim_state is not None:
            self.optim_state = global_optim_state

    def _setup_stage(self, datamodule: DataModule, stage: str=""):
        dataloader = getattr(datamodule, stage + "_dataloader")()
        global_buffers, global_params = {}, {}
        for data, _, indices in dataloader:
            if not self._relations:
                state = self._setup_graph(data)
            else:
                state = initialize_traces(self.model, self.guide, self._rng,
                                          self.parameters, data)
            self._rng = state.rng

            local_buffers = {}
            for name, site in itertools.chain(state.guide_trace.items(),
                                              state.model_trace.items()):
                if site["type"] == "mutable":
                    self._particle_params.add(name)
                    if self._site_in_scope(site, "mutable"):
                        local_buffers[name] = site["value"]
                    else:
                        global_buffers[name] = site["value"]

            local_params = {}
            for param, value in state.params.items():
                site = state.guide_trace.get(param, None)
                site = state.model_trace.get(param, site)
                if site["kwargs"].get("particle", False):
                    self._particle_params.add(param)
                requires_grad = site["kwargs"].get("requires_grad", True)
                if requires_grad:
                    if self._site_in_scope(site, "param"):
                        local_params[param] = value
                    else:
                        global_params[param] = value
                else:
                    if self._site_in_scope(site, "param"):
                        local_buffers[param] = value
                    else:
                        global_buffers[param] = value

            if self.optim_state:
                optim_state = None
            else:
                self._buffer_state = global_buffers
                optim_state = self.optimizer.init(global_params)

            local_optim_state = self.optimizer.init(local_params)
            if self.local_parameters is None:
                self.local_parameters = set(local_params.keys())
            self.save_batch(indices, local_optim_state, local_buffers,
                            optim_state, stage=stage)

        return state.guide_trace, state.model_trace

    def setup_step(self, datamodule: DataModule):
        self.stage_parameters["test"] = BatchParameters(
            len(datamodule.data_test), axis=self._batch_axis
        )
        self.stage_parameters["train"] = BatchParameters(
            len(datamodule.data_train) + len(datamodule.data_val),
            axis=self._batch_axis
        )
        self.stage_buffers["test"] = BatchParameters(
            len(datamodule.data_test), axis=self._batch_axis + 1
        )
        self.stage_buffers["train"] = BatchParameters(
            len(datamodule.data_train) + len(datamodule.data_val),
            axis=self._batch_axis + 1
        )

        self._setup_stage(datamodule, stage="test")
        return self._setup_stage(datamodule, stage="train")

    def _site_in_scope(self, site, site_type):
        if site_type == "mutable":
            return site["type"] == site_type and\
                   (self._plate + "_") in site["name"]
        return site["type"] == site_type and any([
            frame.name == self._plate for frame
            in site["cond_indep_stack"]
        ])

    def _step(self, data, indices, stage: str="train"):
        loss, (global_optim, local_optim), self._rng, state = self._update(
            data, *self.load_batch(indices, stage=stage), self.rng
        )
        loader = "train" if stage == "valid" else stage
        local_buffers = {k: v[0] for k, v in state["trace"].items()
                         if k in self.stage_buffers[loader].tensors.keys()}
        self.save_batch(
            indices, local_optim, local_buffers,
            global_optim_state=global_optim if stage == "train" else None,
            stage=stage
        )
        self._global_buffers = {k: v for k, v in state["mutables"].items()
                                if k not in local_buffers}
        return {"loss": loss, "log_w": state["log_w"]}

    def train_step(self, data, _, indices, *args, **kwargs):
        return self._step(data, indices, stage="train")

    def test_step(self, data, _, indices, *args, **kwargs):
        return self._step(data, indices, stage="test")

    @cached_property
    def _update(self):
        @jax.jit
        def fn(data, global_optim_state, local_optim_state, buffers, rng):
            next_rng, rng = random.split(rng)
            def loss_fn(params, particle_params):
                for buffer, value in buffers.items():
                    value = jax.lax.stop_gradient(value)
                    if buffer in self._particle_params:
                        particle_params[buffer] = value
                    else:
                        params[buffer] = value
                return self.tracer.loss(rng, params, particle_params,
                                        self.model, self.guide, data)
            loss_grads = jax.value_and_grad(loss_fn, argnums=[0, 1],
                                            has_aux=True)

            params, particle_params = {}, {}
            for param, value in self._parameters(global_optim_state).items():
                if param in self._particle_params:
                    particle_params[param] = value
                else:
                    params[param] = value
            for param, value in self._parameters(local_optim_state,
                                                 local=True).items():
                if param in self._particle_params:
                    particle_params[param] = value
                else:
                    params[param] = value

            (loss, state), grads = loss_grads(params, particle_params)
            grads = grads[0] | grads[1]
            global_grads, local_grads = {}, {}
            for k, v in grads.items():
                if k in self.local_parameters:
                    local_grads[k] = v
                else:
                    global_grads[k] = v
            global_optim_state = self.optimizer.update(global_grads,
                                                       global_optim_state,
                                                       value=loss)
            local_optim_state = self.local_optimizer.update(local_grads,
                                                            local_optim_state,
                                                            value=loss)
            optim_states = (global_optim_state, local_optim_state)
            return loss, optim_states, next_rng, state
        return fn

    def valid_step(self, data, _, indices, *args, **kwargs):
        return self._step(data, indices, stage="valid")
