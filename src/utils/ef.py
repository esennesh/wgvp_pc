import efax
import jax
from jax.typing import ArrayLike
import numpyro.distributions as dist
from numpyro.distributions import constraints
from numpyro.distributions.util import promote_shapes
from typing import Optional, Tuple

class ExponentialFamily(dist.Distribution):
    def __init__(self, np: efax.NaturalParametrization, batch_shape=(),
                 event_shape=()):
        self.np = np
        super(ExponentialFamily, self).__init__(batch_shape=batch_shape,
                                                event_shape=event_shape)

    def cdf(self, value: ArrayLike) -> ArrayLike:
        return self._to_pyro().cdf(value)

    def entropy(self) -> ArrayLike:
        return self.np.entropy()

    def enumerate_support(self, expand: bool = True) -> ArrayLike:
        return self._to_pyro().enumerate_support(expand)

    def icdf(self, value: ArrayLike) -> ArrayLike:
        return self._to_pyro().icdf(value)

    def log_prob(self, value: ArrayLike) -> ArrayLike:
        @jax.custom_jvp
        def log_prob(np: efax.NaturalParametrization, value: ArrayLike) -> ArrayLike:
            return np.log_pdf(value)

        @log_prob.defjvp
        def _natural_jvp(primals, tangents) -> Tuple[ArrayLike, ArrayLike]:
            np, value = primals
            np_dot, value_dot = tangents

            ep = np.to_exp()
            ep_dot = ep.__class__(**{
                eparam: getattr(np_dot, nparam) for nparam, eparam
                        in zip(np_dot.dynamic_fields, ep.dynamic_fields)
            })
            return jax.jvp(lambda ep, value: ep.to_nat().log_pdf(value),
                           (ep, value), (ep_dot, value_dot))

        return log_prob(self.np, value)

    @property
    def mean(self) -> ArrayLike:
        return self._to_pyro.mean

    def natural_score(self, value: ArrayLike, score_value=False) -> ArrayLike:
        ep = self.np.to_exp()

        def fn(params, value):
            return ep.__class__(**params).to_nat().log_pdf(value).sum()
        argnums = (0, 1) if score_value else 0
        grads = jax.grad(fn, argnums=argnums)(self.np.to_exp().__dict__, value)
        return {nk: grads[ek] for nk, ek in
                zip(self.np.__dict__.keys(), grads.keys())}

    def sample(self, key: Optional[jax.dtypes.prng_key],
               sample_shape: tuple[int, ...] = ()) -> ArrayLike:
        return self.np.sample(key, sample_shape)

    def score(self, value: ArrayLike, score_value=False) -> ArrayLike:
        def fn(params, value):
            return self.np.__class__(**params).log_pdf(value).sum()
        argnums = (0, 1) if score_value else 0
        return jax.grad(fn, argnums=argnums)(self.np.__dict__, value)

    def _to_pyro(self) -> dist.Distribution:
        raise NotImplementedError

    @property
    def variance(self) -> ArrayLike:
        return self._to_pyro().variance
