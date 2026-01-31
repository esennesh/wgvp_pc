from functools import partial
from jax import Array, jit
import jax.numpy as jnp
import jax.random as random
from numpyro.infer import ELBO, Predictive, SVI
import numpyro
from typing import Any, Dict

from .para import ParaMonad
from src.data import DataModule
from src.inference.elbo import TraceVectorized_ELBO
from src.utils import reconstruct

class SviPara(ParaMonad):
    def __init__(self, data_shape, guide, lr, model, elbo: ELBO, rng):
        if not isinstance(rng, Array):
            rng = random.key(rng)
        self.optimizer = numpyro.optim.Adam(step_size=lr)
        self._rng = rng
        self.svi = SVI(model, guide, self.optimizer, elbo)
        self.svi_state = None

    def __call__(self, *args, **kwargs):
        predictive = Predictive(
            reconstruct(self.svi.model), guide=self.svi.guide,
            num_samples=self.svi.loss.num_particles, batch_ndims=None,
            parallel=False, params=self.svi.get_params(self.svi_state)
        )
        return predictive(self.svi_state.rng_key, *args, **kwargs)

    def load(self, checkpoint: Dict[str, Any]):
        self.svi_state = checkpoint["svi_state"]

    @property
    def parameters(self):
        return self.optimizer.get_params(self.svi_state.optim_state)

    def save(self):
        return {"svi_state": self.svi_state}

    def setup_step(self, datamodule: DataModule):
        for batch in datamodule.test_dataloader():
            data = batch[0]
            break

        if self.svi_state is None:
            self.svi_state = self.svi.init(self._rng, data)
        else:
            self.svi.init(self.svi_state.rng_key, data)
        return self.svi_state

    @staticmethod
    @partial(jit, static_argnums=0)
    def svi_evaluate(svi, state, data):
        rng_key, rng_key_eval = random.split(state.rng_key)
        loss_fn = numpyro.infer.svi._make_loss_fn(
            svi.loss, rng_key_eval, svi.constrain_fn, svi.model, svi.guide,
            (data,), {}, svi.static_kwargs, mutable_state=state.mutable_state
        )
        loss, mutable_state = loss_fn(svi.get_params(state))
        return (numpyro.infer.svi.SVIState(state.optim_state, mutable_state,
                                           rng_key), loss)

    @staticmethod
    @partial(jit, static_argnums=0)
    def svi_update(svi, state, data):
        return svi.update(state, data)

    def test_step(self, data, *args, **kwargs):
        self.svi_state, loss = self.svi_evaluate(self.svi, self.svi_state, data)
        return {"loss": loss}

    def train_step(self, data, *args, **kwargs):
        self.svi_state, loss = self.svi_update(self.svi, self.svi_state, data)
        return {"loss": loss}

    def valid_step(self, data, *args, **kwargs) -> Dict[str, float]:
        self.svi_state, loss = self.svi_evaluate(self.svi, self.svi_state, data)
        return {"loss": loss}
