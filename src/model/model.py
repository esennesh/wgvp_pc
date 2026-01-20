from flax import nnx
import functools
import jax
from jax import jit, lax
from jax.example_libraries import stax
import jax.numpy as jnp
import math
import numpyro
from numpyro.contrib.module import nnx_module
import numpyro.distributions as dist
import reversible_deq as rdeq

# Takes pixel intensities of the attention window to parameters (mean,
# standard deviation) of the distribution over the latent code, z_what.
class WhatEncoder(nnx.Module):
    def __init__(self, hidden_dim=400, in_side=20, z_what_dim=50, *,
                 rngs: nnx.Rngs):
        self._att_side = in_side
        self.convs = nnx.Sequential(
            # Toil and trouble.
            nnx.ConvTranspose(in_features=1, out_features=8, kernel_size=(4, 4),
                              rngs=rngs),
            nnx.silu,
            nnx.Conv(in_features=8, out_features=16, kernel_size=(5, 5),
                     strides=(2, 2), rngs=rngs),
            nnx.silu,
            nnx.Conv(in_features=16, out_features=32, kernel_size=(5, 5),
                     strides=(2, 2), rngs=rngs),
            nnx.silu,
        )

        self.mlp = nnx.Sequential(
            nnx.Linear(7 * 7 * 32, hidden_dim, rngs=rngs), nnx.silu,
            nnx.Linear(hidden_dim, z_what_dim * 2, rngs=rngs)
        )

    @property
    def att_side(self):
        return self._att_side

    def __call__(self, att, rngs=None):
        h = self.convs(att.transpose(0, 2, 3, 1)).reshape((-1, 7 * 7 * 32))
        a = self.mlp(h)
        return a[:, 0:50], nnx.softplus(a[:, 50:])

class WhereEncoder(nnx.Module):
    def __init__(self, img_side=50, hidden_dim=256, *, rngs: nnx.Rngs):
        self._img_side = img_side
        self.layers = nnx.Sequential(
            nnx.Linear(img_side ** 2, hidden_dim, rngs=rngs), nnx.tanh,
            nnx.Linear(hidden_dim, 6, rngs=rngs),
        )

    def __call__(self, img, rngs=None):
        img = img.reshape((img.shape[0], math.prod(img.shape[1:]),))
        a = self.layers(img)
        z_where_loc = jnp.concatenate((nnx.softplus(a[:, 0:1]), a[:, 1:3]),
                                      axis=1)
        z_where_scale = nnx.softplus(a[:, 3:]) # Squish to >0
        return z_where_loc, z_where_scale

    @property
    def img_side(self):
        return self._img_side

def z_where_inv(z_where):
    # Take a batch of z_where vectors, and compute their "inverse".
    # That is, for each row compute:
    # [s,x,y] -> [1/s,-x/s,-y/s]
    # These are the parameters required to perform the inverse of the
    # spatial transform performed in the generative model.
    s = z_where[:, 0]
    return jnp.array([1 / s, -z_where[:, 1] / s, -z_where[:, 2] / s]).T

def air_guide(xs, what_enc: WhatEncoder, where_enc: WhereEncoder):
    enc_what = nnx_module("what_enc", what_enc)
    enc_where = nnx_module("where_enc", where_enc)

    with numpyro.plate("batch", xs.shape[0]):
        z_where_loc, z_where_scale = enc_where(xs)
        z_where = numpyro.sample(
            'z_where', dist.Normal(z_where_loc, z_where_scale).to_event(1)
        )
        blitter = jax.vmap(functools.partial(scale_and_translate,
                                             out_side=what_enc.att_side))
        x_att = blitter(xs, z_where_inv(z_where))
        z_what_loc, z_what_scale = enc_what(x_att)
        z_what = numpyro.sample(
            'z_what', dist.Normal(z_what_loc, z_what_scale).to_event(1)
        )

class AirDecoder(nnx.Module):
    def __init__(self, hidden_dim=400, out_side=20, z_what_dim=50, *,
                 rngs: nnx.Rngs):
        super().__init__()
        self._out_side = out_side
        self._z_what_dim = z_what_dim
        self.mlp = nnx.Sequential(
            nnx.Linear(z_what_dim, hidden_dim, rngs=rngs), nnx.silu,
            nnx.Linear(hidden_dim, 7 * 7 * 32, rngs=rngs), nnx.silu
        )
        self.convs = nnx.Sequential(
            # Double,
            nnx.ConvTranspose(in_features=32, out_features=16,
                              kernel_size=(5, 5), strides=(2, 2), rngs=rngs),
            nnx.silu,
            # Double,
            nnx.ConvTranspose(in_features=16, out_features=8,
                              kernel_size=(5, 5), strides=(2, 2), rngs=rngs),
            nnx.silu,
            # Toil and trouble.
            nnx.Conv(in_features=8, out_features=1, kernel_size=(4, 4),
                     rngs=rngs),
            nnx.sigmoid
        )
        self.precision = nnx.Linear(7 * 7 * 32, 1, rngs=rngs)

    def __call__(self, z_what, rngs=None):
        h = self.mlp(z_what)
        x = self.convs(h.reshape(-1, 7, 7, 32))
        x = x.reshape(-1, 1, self._out_side, self._out_side)
        return x, jnp.exp(2 * self.precision(h)).squeeze()

    @property
    def z_what_dim(self):
        return self._z_what_dim

def scale_and_translate(image, where, out_side=50):
    scalar = where[0]
    where = where[1:]
    translate = abs(image.shape[-1] - out_side) * (where[..., ::-1] + 1) / 2
    return jax.image.scale_and_translate(image, (1, out_side, out_side), (1, 2),
                                         jnp.ones(2) * scalar, translate,
                                         method="cubic", antialias=False)

def air_model(xs, decoder: AirDecoder, out_side=50):
    decode = nnx_module("decoder", decoder)
    log_scale = numpyro.param("log_scale", jnp.array(0.3))
    canvas_variance = jnp.exp(2 * log_scale) * jnp.ones(xs.shape)
    canvas = jnp.zeros(xs.shape)

    with numpyro.plate("batch", xs.shape[0]):
        # Sample object pose. This is a 3-dimensional vector representing (x, y)
        # position and size.

        z_where = numpyro.sample(
            'z_where',
            dist.Normal(jnp.array([3., 0., 0.]),
                        jnp.array([0.1, 1., 1.])).to_event(1)
        )

        # Sample object code. This is a 50-dimensional vector.
        z_what = numpyro.sample('z_what', dist.Normal(
            jnp.zeros(decoder.z_what_dim),
            jnp.ones(decoder.z_what_dim)
        ).to_event(1))

        # Map code to pixel space using the neural network.
        x_att, variance_att = decode(z_what)
        # Position/scale object within larger image.
        blitter = jax.vmap(functools.partial(scale_and_translate,
                                             out_side=out_side))
        xhat = canvas + blitter(x_att, z_where)
        variance_att = jnp.expand_dims(variance_att, [1, 2, 3]) *\
                       jnp.where(xhat > 0, jnp.ones(xhat.shape),
                                 jnp.zeros(xhat.shape))
        scale = jnp.sqrt(canvas_variance + variance_att)
        numpyro.sample('x', dist.Normal(xhat, scale).to_event(3), obs=xs)

class PVaeEncoder(nnx.Module):
    def __init__(self, z_dim, *, rngs: nnx.Rngs, x_dim=28):
        self.conv1 = nnx.Conv(1, 16, kernel_size=(3, 3), strides=2, padding=1,
                              rngs=rngs)
        self.conv2 = nnx.Conv(16, 32, kernel_size=(3, 3), strides=2, padding=1,
                              rngs=rngs)
        feature_area = (x_dim // 4) ** 2
        self.linear = nnx.Linear(32 * feature_area, z_dim, rngs=rngs)

    def __call__(self, xs, rngs=None):
        hs = nnx.swish(self.conv1(xs.swapaxes(-3, -1)))
        hs = nnx.swish(self.conv2(hs))
        return self.linear(hs.reshape(hs.shape[0], -1))

class DEQ(nnx.Module):
    def __init__(self, adjoint: rdeq.AbstractAdjoint, function,
                 solver: rdeq.AbstractSolver, *, rngs: nnx.Rngs, max_steps=2,
                 tol=1e-6):
        self.adjoint = adjoint
        self.function = function
        self.max_steps = max_steps
        self.solver = solver
        self.tol = tol

    def __call__(self, xs, z0, rngs: nnx.Rngs=None):
        solution = rdeq.solve(self.function, jax.lax.stop_gradient(z0),
                              jax.lax.stop_gradient(xs),
                              self.solver, self.adjoint, self.tol,
                              self.max_steps)
        return solution.z1

class DeqEncoder(nnx.Module):
    def __init__(self, adjoint, solver, x_dim, z_dim, *, rngs: nnx.Rngs,
                 max_steps=2, tol=1e-6):
        self.input = nnx.Linear(x_dim, z_dim, rngs=rngs, use_bias=False)
        self.readout = nnx.Linear(z_dim, z_dim, rngs=rngs, use_bias=True)
        self.recurrence = nnx.Linear(z_dim, z_dim, rngs=rngs, use_bias=True)

        def fn(zs, xs):
            return nnx.tanh(self.input(xs) + self.recurrence(zs))
        self.deq = DEQ(adjoint, fn, solver, max_steps=max_steps, rngs=rngs,
                       tol=tol)

        self._x_dim = x_dim
        self._z_dim = z_dim

    def __call__(self, u, xs):
        return self.readout(self.deq(xs, u))

    @property
    def z_dim(self):
        return self._z_dim

def pvae_fpi_guide(xs, dynamics: DeqEncoder):
    z_dim = dynamics.z_dim
    dynamics = nnx_module("dynamics", dynamics)
    u_0 = numpyro.param("prior$params")
    if u_0 is not None:
        u_0 = jnp.expand_dims(u_0["log_rate"], (0,))
        u_0 = jnp.broadcast_to(u_0, (xs.shape[0], z_dim))
    else:
        u_0 = jnp.zeros((xs.shape[0], z_dim))
    u = dynamics(u_0, xs.reshape((xs.shape[0], -1)))
    with numpyro.plate("batch", xs.shape[0]):
        return numpyro.sample("z", dist.Poisson(jnp.exp(u)).to_event(1))

def pvae_guide(xs, encoder: PVaeEncoder):
    encoder = nnx_module("encoder", encoder)
    u = encoder(xs)
    with numpyro.plate("batch", xs.shape[0]):
        return numpyro.sample("z", dist.Poisson(jnp.exp(u)).to_event(1))

def pvae_linear_guide(xs, encoder: nnx.Linear):
    encoder = nnx_module("encoder", encoder)
    u = encoder(xs.reshape((xs.shape[0], -1)))
    with numpyro.plate("batch", xs.shape[0]):
        return numpyro.sample("z", dist.Poisson(jnp.exp(u)).to_event(1))

class PVaePrior(nnx.Module):
    def __init__(self, z_dim, *, rngs: nnx.Rngs):
        self.log_rate = nnx.Param(rngs.uniform(shape=(z_dim,), minval=-6.,
                                               maxval=-4.))

    def __call__(self, rngs=None):
        return jnp.exp(self.log_rate)

def pvae_model(xs, decoder: nnx.Linear, prior: PVaePrior, scale=None):
    decoder = nnx_module("decoder", decoder)
    prior = nnx_module("prior", prior)
    if scale is None:
        scale = jnp.exp(numpyro.param("log_scale", jnp.zeros(())))
    with numpyro.plate("batch", xs.shape[0]):
        z = numpyro.sample("z", dist.Poisson(prior()).to_event(1))
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

def mnist_normal_model(batch, hidden_dim=400, z_dim=100):
    batch = jnp.reshape(batch, (batch.shape[0], -1))
    batch_dim, out_dim = jnp.shape(batch)
    decode = numpyro.module("decoder", decoder(hidden_dim, out_dim),
                            (batch_dim, z_dim))
    scale = jnp.exp(numpyro.param("log_scale", jnp.zeros(())))
    with numpyro.plate("batch", batch_dim):
        z = numpyro.sample("z", dist.Normal(jnp.zeros(z_dim),
                                            jnp.ones(z_dim)).to_event(1))
        img_loc = decode(z)
        return numpyro.sample("obs", dist.Normal(img_loc, scale).to_event(1),
                              obs=batch)

def mnist_model(batch, hidden_dim=400, z_dim=100):
    batch = jnp.reshape(batch, (batch.shape[0], -1))
    batch_dim, out_dim = jnp.shape(batch)
    decode = numpyro.module("decoder", decoder(hidden_dim, out_dim),
                            (batch_dim, z_dim))
    with numpyro.plate("batch", batch_dim):
        z = numpyro.sample("z", dist.Normal(jnp.zeros((z_dim,)),
                                            jnp.ones((z_dim,))).to_event(1))
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
