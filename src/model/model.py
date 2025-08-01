from flax import nnx
from jax import jit, lax
from jax.example_libraries import stax
import jax.numpy as jnp

import numpyro
from numpyro.contrib.module import nnx_module
import numpyro.distributions as dist

class PVaeEncoder(nnx.Module):
    def __init__(self, z_dim, *, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(1, 16, kernel_size=(3, 3), strides=2, padding=1,
                              rngs=rngs)
        self.conv2 = nnx.Conv(16, 32, kernel_size=(3, 3), strides=2, padding=1,
                              rngs=rngs)
        self.linear = nnx.Linear(32 * 7 * 7, z_dim, rngs=rngs)

    def __call__(self, xs, rngs=None):
        hs = nnx.swish(self.conv1(xs.swapaxes(-3, -1)))
        hs = nnx.swish(self.conv2(hs))
        return self.linear(hs.reshape(hs.shape[0], -1))

def pvae_guide(xs, encoder: PVaeEncoder):
    encoder = nnx_module("encoder", encoder)
    log_u = encoder(xs)
    with numpyro.plate("batch", xs.shape[0]):
        return numpyro.sample("z", dist.Poisson(jnp.exp(log_u)).to_event(1))

def pvae_model(xs, decoder: nnx.Linear, z_dim=1024, x_side=28):
    decoder = nnx_module("decoder", decoder)
    scale = jnp.exp(numpyro.param("log_scale", jnp.zeros(())))
    with numpyro.plate("batch", xs.shape[0]):
        z = numpyro.sample("z", dist.Poisson(1).expand([z_dim]).to_event(1))
        loc = decoder(z).reshape(xs.shape)
        return numpyro.sample("x", dist.Normal(loc, scale).to_event(3), obs=xs)

def encoder(hidden_dim, z_dim):
    return stax.serial(
        stax.Dense(hidden_dim, W_init=stax.randn()),
        stax.Softplus,
        stax.FanOut(2),
        stax.parallel(
            stax.Dense(z_dim, W_init=stax.randn()),
            stax.serial(stax.Dense(z_dim, W_init=stax.randn()), stax.Exp),
        ),
    )

def decoder(hidden_dim, out_dim):
    return stax.serial(
        stax.Dense(hidden_dim, W_init=stax.randn()),
        stax.Softplus,
        stax.Dense(out_dim, W_init=stax.randn()),
        stax.Sigmoid,
    )

def mnist_model(batch, hidden_dim=400, z_dim=100):
    batch = jnp.reshape(batch, (batch.shape[0], -1))
    batch_dim, out_dim = jnp.shape(batch)
    decode = numpyro.module("decoder", decoder(hidden_dim, out_dim),
                            (batch_dim, z_dim))
    with numpyro.plate("batch", batch_dim):
        z = numpyro.sample("z", dist.Normal(0, 1).expand([z_dim]).to_event(1))
        img_loc = decode(z)
        return numpyro.sample("obs", dist.Bernoulli(img_loc).to_event(1),
                              obs=batch)

def mnist_guide(batch, hidden_dim=400, z_dim=100):
    batch = jnp.reshape(batch, (batch.shape[0], -1))
    batch_dim, out_dim = jnp.shape(batch)
    encode = numpyro.module("encoder", encoder(hidden_dim, z_dim),
                            (batch_dim, out_dim))
    z_loc, z_std = encode(batch)
    with numpyro.plate("batch", batch_dim):
        return numpyro.sample("z", dist.Normal(z_loc, z_std).to_event(1))
