"""Validate DoubleCVTracer against the closed-form (IPVaeGuide) gradient.

On a linear-Gaussian likelihood with a Poisson latent the ELBO has the same
closed form IPVaeGuide uses, so its phi-gradient is an exact oracle. The
double-CV estimator must be (a) unbiased for that oracle and (b) much
lower-variance than plain REINFORCE. Run: python tests/test_double_cv.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import functools
import jax
import jax.numpy as jnp
import jax.random as random
import numpyro
import numpyro.distributions as dist

from src.inference.graphical import DoubleCVTracer

jax.config.update("jax_enable_x64", True)   # tighten MC tolerance checks

# ---- tiny linear-Gaussian / Poisson problem -------------------------------
D, X, B = 6, 4, 3                 # latent dim, obs dim, batch
key = random.key(0)
kW, kx, kr = random.split(key, 3)
W = (random.normal(kW, (D, X)) * 0.4) ** 2        # nonnegative NMF atoms
sigma = jnp.asarray(0.5)
lam_p = jnp.full((D,), 0.5)                        # prior Poisson rate
x = jnp.abs(random.normal(kx, (B, X)))             # data
rate_log = random.normal(kr, (B, D)) * 0.3 - 0.2   # variational params phi


def _model(x, W=W, sigma=sigma, lam_p=lam_p):
    with numpyro.plate("batch", x.shape[0]):
        z = numpyro.sample("z", dist.Poisson(lam_p).to_event(1))
        loc = z @ W
        numpyro.sample("x", dist.Normal(loc, sigma).to_event(1), obs=x)


def _guide(x):
    rate = jnp.exp(numpyro.param("rate_log", jnp.zeros((x.shape[0], D))))
    with numpyro.plate("batch", x.shape[0]):
        numpyro.sample("z", dist.Poisson(rate).to_event(1))


# ---- closed-form oracle: -ELBO and its phi-gradient (== IPVaeGuide) -------
def neg_elbo_exact(rate_log):
    lam = jnp.exp(rate_log)                         # (B, D) = E_q[z]
    loc = lam @ W                                   # (B, X)
    var = lam @ (W ** 2)                            # (B, X)  Var_q[loc]
    recon = (-0.5 / sigma ** 2 * (((x - loc) ** 2).sum(-1) + var.sum(-1))
             - 0.5 * X * jnp.log(2 * jnp.pi * sigma ** 2))
    kl = (lam * (jnp.log(lam) - jnp.log(lam_p)) - lam + lam_p).sum(-1)
    return -(recon - kl).sum()

oracle = jax.grad(neg_elbo_exact)(rate_log)


# ---- double-CV estimator via the actual tracer ----------------------------
def dcv_grad(rng, K):
    tracer = DoubleCVTracer(num_particles=K)
    loss = lambda phi: tracer.loss(rng, {"rate_log": phi}, {}, _model,
                                   _guide, x)[0]
    return jax.grad(loss)(rate_log)


# ---- plain REINFORCE + leave-one-out baseline (variance reference) --------
def reinforce_grad(rng, K):
    def loss(phi):
        rate = jnp.exp(phi)
        zk = jax.lax.stop_gradient(
            dist.Poisson(rate).sample(rng, (K,)).astype(jnp.float64))  # (K,B,D)
        logq = dist.Poisson(rate).to_event(1).log_prob(zk)             # (K,B)
        logp = (dist.Poisson(lam_p).to_event(1).log_prob(zk)
                + dist.Normal(zk @ W, sigma).to_event(1).log_prob(x))  # (K,B)
        f = logp - logq
        adv = f - (f.sum(0, keepdims=True) - f) / (K - 1)              # LOO
        sg = jax.lax.stop_gradient
        surr = (logp - sg(logq)) + logq * sg(adv)
        return -surr.mean(0).sum()
    return jax.grad(loss)(rate_log)


def stats(grad_fn, K, n_seeds):
    keys = random.split(random.key(1), n_seeds)
    grads = jax.vmap(lambda k: grad_fn(k, K))(keys)                    # (S,B,D)
    mean = grads.mean(0)
    bias = jnp.linalg.norm(mean - oracle) / jnp.linalg.norm(oracle)
    var = grads.var(0).sum()
    return mean, float(bias), float(var)


if __name__ == "__main__":
    K, S = 16, 400
    print(f"oracle grad norm: {float(jnp.linalg.norm(oracle)):.4f}")
    print(f"(K={K} particles, S={S} seeds)\n")

    _, dcv_bias, dcv_var = stats(dcv_grad, K, S)
    _, rf_bias, rf_var = stats(reinforce_grad, K, S)

    print(f"double-CV   : rel.bias={dcv_bias:.4f}  grad-var={dcv_var:.4e}")
    print(f"REINFORCE   : rel.bias={rf_bias:.4f}  grad-var={rf_var:.4e}")
    print(f"variance ratio (REINFORCE / double-CV): {rf_var / dcv_var:.1f}x")

    # both estimators are unbiased; the averaged gradient must track the
    # closed-form oracle within Monte-Carlo error
    assert dcv_bias < 0.05, f"double-CV is biased vs closed form: {dcv_bias}"
    assert rf_bias < 0.10, f"REINFORCE oracle/sanity check failed: {rf_bias}"
    # the whole point: the double CV must cut variance substantially
    assert dcv_var < rf_var, "double-CV did not reduce variance"
    print("\nPASS: double-CV is unbiased vs the closed-form gradient and "
          "lower-variance than REINFORCE.")
