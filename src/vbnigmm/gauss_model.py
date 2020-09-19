__author__ = 'TAKEKAWA Takashi <takekawa@tk2lab.org>'
__credits__ = 'Copyright 2020, TAKEKAWA Takashi'


import numpy as np

from .mixturebase import MixtureBase
from .gauss_posterior import BayesianGaussianMixturePosterior


class BayesianGaussianMixture(MixtureBase):

    posterior_type = BayesianGaussianMixturePosterior

    def _get_prior(self, x, mean=None, cov=None,
                   concentration_prior_type='dpm', l0=None, r0=None,
                   mean_scale=1.0, cov_scale=0.3, cov_reliability=2.0):
        x, mean, cov = self._check_data(x, mean, cov)
        l0, r0 = self._check_concentration(concentration_prior_type, l0, r0)
        s0, t0 = self._check_covariance(cov_reliability, cov_scale, cov)
        u0 = (cov_scale / mean_scale) ** 2
        m0 = mean
        return l0, r0, s0, t0, u0, m0

    def _init_expect(self, z):
        return z,

    def _update(self, x, yz, l0, r0, s0, t0, u0, m0):
        Yz = yz.sum(axis=0)
        Xz = x.T @ yz
        X2 = np.tensordot(x, yz[:, None, :] * x[:, :, None], (0, 0))

        l1 = l0 + Yz
        if r0 is None:
            r1 = None
        else:
            r1 = r0 + np.hstack((np.cumsum(Yz[::-1])[-2::-1], 0))
        u1 = u0 + Yz
        m1 = (u0 * m0[:, None] + Xz) / u1
        s1 = s0 + Yz
        t1 = t0.copy()
        t1 += u0 * m0 * m0[:, None]
        t1 = t1[:, :, None] + X2
        t1 -= u1 * m1 * m1[:, None, :]
        return l1, r1, s1, t1.T, u1, m1.T
