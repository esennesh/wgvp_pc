from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from numpyro.contrib.module import nnx_module
import tqdm
from typing import Optional

from .para import ParaMonad
from src.data import DataModule
import src.utils as utils
from .trainer import Trainer

def r2(xs, xs_hat):
    squared_errors = ((xs - xs_hat) ** 2).sum(axis=(-3, -2, -1))
    total_squares = ((xs - xs.mean(axis=0, keepdims=True)) ** 2).sum(axis=(-3,
                                                                           -2,
                                                                           -1))
    return 1.0 - squared_errors / (total_squares + jnp.finfo("float64").eps)

def sparse_score(z, cutoff: float = None):
    def _compute_score(axis: int, fix: bool = True):
        m = z.shape[axis]
        numen = jnp.sum(z, axis=axis) ** 2
        denum = jnp.sum(z ** 2, axis=axis)
        mask = (numen == 0) & (denum == 0)  # no spikes
        denum[denum == 0] = jnp.finfo("float32").eps
        score = 1 - (numen / denum) / m
        score /= (1 - 1 / m)
        if fix:  # no spikes
            score[mask] = 1.0
        return score

    if z.ndim == 1:
        z = z.reshape(-1, 1)

    lifetime = _compute_score(0)
    population = _compute_score(-1)

    # percentages
    if cutoff is None:
    	percents = None
    else:
        z = z.ravel()
        counts = collections.Counter(jnp.round(z).astype(int))
        portions = {k: v / jnp.prod(z.shape) for k, v in counts.most_common()}
        try:
            cutoff = next(k + 1 for k, v in portions.items() if v < cutoff)
        except StopIteration:
            cutoff = np.inf
        percents = {str(k): v for k, v in portions.items() if k < cutoff}
        percents[f'{cutoff}+'] = sum(v for k, v in portions.items()
                                     if k >= cutoff)
        percents = {k: np.round(v * 100, 1) for k, v in percents.items()}
        percents = dict(sorted(percents.items(),
                               key=lambda t: utils.sort_key(t[0])))
    return lifetime, population, percents

class PVaeTrainer(Trainer):
    def analysis(self, datamodule: DataModule, monad: ParaMonad, active=None,
                 average_samples=True, compute_sparsity=False,
                 return_recons=False, stage="valid", t_total=None,
                 verbose=True):
        dataloader = getattr(datamodule, stage + "_dataloader")()

        extra_items = ['samples', 'du', 'r2', 'mse']
        if return_recons:
            extra_items.append("recon")

        if hasattr(monad.guide, "keywords"):
            dynamics = monad.guide.keywords["dynamics"]
            nnx.update(dynamics, monad.parameters["dynamics$params"])
        else:
            dynamics = monad.guide.dynamics
        prior = monad.model.keywords["prior"]
        nnx.update(prior, monad.parameters["prior$params"])
        u_0 = prior.log_rate.value
        x_dim = monad.model.keywords["decoder"].kernel.shape[1]
        z_dim = monad.model.keywords["decoder"].kernel.shape[0]
        if active is None:
            active = jnp.ones(z_dim) > 0

        shape = (t_total,)
        if not average_samples:
            shape = (len(dataloader.dataset),) + shape
        du_norm = np.empty(shape)
        elbo = np.empty(shape)
        kl = np.empty(shape)
        sse = np.empty(shape)
        total_r2 = np.empty(shape)

        samples_shape = (len(dataloader.dataset), t_total)
        state_final = np.empty(samples_shape)

        zeroes_count, zeroes_total = np.zeros(t_total), np.zeros(t_total)
        if compute_sparsity:
            lifetime_acc = jnp.zeros(t_total)
            population_acc = jnp.zeros(t_total)
            sparsity_num_samples = 0

        if return_recons:
            recons = np.empty((len(dataloader.dataset), t_total, x_dim))
        else:
            recons = None

        for b, (xs, *_) in tqdm.tqdm(enumerate(dataloader), disable=not verbose,
                                     ncols=70):
            start = dataloader.batch_size * b
            batch_interval = range(start, start + len(xs))
            batch_recons, norms = [], []
            kls, elbos, r2s, sses = [], [], [], []
            if compute_sparsity:
                z_sparsities = []

            for t in tqdm.tqdm(range(0, t_total), ncols=70, leave=False):
                traces = monad(xs, stage=stage, return_trace=True,
                               max_steps=t+1)
                xs_hat = traces["x"][0].mean(axis=0)
                zs = traces["z"][0]

                if u_0.shape[0] != xs.shape[0]:
                    u_0 = jnp.broadcast_to(u_0[jnp.newaxis, ...], (xs.shape[0],
                                                                   *u_0.shape))
                du = dynamics(u_0, jax.lax.collapse(xs, 1), max_steps=t+1)
                batch_elbo = np.array(
                    sum(v[1] - v[2] for v in traces.values()).mean(0)
                )
                batch_kl = np.array(
                    (traces["z"][2] - traces["z"][1]).mean(axis=0)
                )
                batch_r2 = np.array(r2(xs, xs_hat))
                batch_sse = np.array(
                    ((xs - xs_hat) ** 2).sum(axis=(-3, -2, -1))
                )
                norms_batch = np.linalg.norm(du[:, active], axis=-1)

                z_active = traces["z"][0][:, :, active]
                zeroes_count[t] += (z_active == 0).sum()
                zeroes_total[t] += z_active.size

                del traces

                if compute_sparsity:
                    z_sparsities.append(z_active)

                if average_samples:
                    kls.append(batch_kl.sum(axis=0))
                    elbos.append(batch_elbo.sum(axis=0))
                    norms.append(norms_batch.sum(axis=0))
                    r2s.append(batch_r2.sum(axis=0))
                    sses.append(batch_sse.sum(axis=0))
                else:
                    elbos.append(batch_elbo)
                    kls.append(batch_kl)
                    norms.append(norms_batch)
                    r2s.append(batch_r2)
                    sses.append(batch_sse)

                if return_recons:
                    batch_recons.append(xs_hat)

            elbos = np.stack(elbos, axis=0 if average_samples else 1)
            kls = np.stack(kls, axis=0 if average_samples else 1)
            r2s = np.stack(r2s, axis=0 if average_samples else 1)
            norms = np.stack(norms, axis=0 if average_samples else 1)
            sses = np.stack(sses, axis=0 if average_samples else 1)

            if average_samples:
                du_norm = du_norm + norms
                elbo = elbo + elbos
                kl = kl + kls
                total_r2 = total_r2 + r2s
                sse = sse + sses
            else:
                du_norm = du_norm.at[batch_interval].set(norms)
                elbo = elbo.at[batch_interval].set(elbos)
                kl = kl.at[batch_interval].set(kls)
                total_r2 = total_r2.at[batch_interval].set(r2s)
                sse = sse.at[batch_interval].set(sses)

            if return_recons:
                recons[batch_interval] = np.stack(batch_recons, axis=1)

            if compute_sparsity:
                z_sparsities = np.stack(z_sparsities, axis=1)
                lt, pop, _ = sparse_score(z_sparsities, cutoff=None)
                lifetime_acc += np.mean(lt, axis=1) * len(xs)
                population_acc += pop.sum(axis=0)
                sparsity_num_samples += len(xs)

        if average_samples:
            elbo /= len(dataloader.dataset)
            kl /= len(dataloader.dataset)
            total_r2 /= len(dataloader.dataset)
            sse /= len(dataloader.dataset)
            du_norm /= len(dataloader.dataset)

        portion_zeroes = zeroes_count / zeroes_total
        sparse_coding_performance = np.sqrt((1 - total_r2) ** 2 +\
                                            (1 - portion_zeroes) ** 2)
        sparse_coding_performance = sparse_coding_performance / np.sqrt(2)
        metrics = {"du_norm": du_norm, "elbo": elbo, "kl": kl,
                   "nelbo": kl + sse, "%-zeros": portion_zeroes, "r2": total_r2,
                   "sse": sse,
                   "sparse_coding_performance": sparse_coding_performance}
        if return_recons:
            metrics["recons"] = recons

        if compute_sparsity:
            metrics["lifetime"] = lifetime_acc / sparsity_num_samples
            metrics["population"] = population_acc / sparsity_num_samples

        return metrics

    def proposal_us(self, datamodule: DataModule, monad: ParaMonad,
                    max_steps=100, stage="test", verbose=True):
        from numpyro.handlers import seed, substitute, trace

        dataloader = getattr(datamodule, stage + "_dataloader")()
        guide = seed(substitute(monad.guide, monad.parameters), monad.rng)
        us = jnp.zeros_like(monad.model.keywords["prior"].log_rate)
        n = 0
        for b, (xs, *_) in tqdm.tqdm(enumerate(dataloader), disable=not verbose,
                                     ncols=70):
            guide_trace = trace(guide).get_trace(xs)
            if hasattr(guide_trace["z"]["fn"].base_dist, "rate"):
                u = jnp.log(guide_trace["z"]["fn"].base_dist.rate)
            else:
                u = guide_trace["z"]["fn"].base_dist.log_mean
            us = us + u.sum(axis=0)
            n += len(xs)

        return us / n
