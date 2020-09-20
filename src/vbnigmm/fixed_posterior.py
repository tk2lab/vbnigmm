__author__ = 'TAKEKAWA Takashi <takekawa@tk2lab.org>'
__credits__ = 'Copyright 2020, TAKEKAWA Takashi'


import numpy as np
import scipy.special as sp

from .nig_posterior import BayesianNIGMixturePosterior, _MeanBias
from .distributions import BaseDist
from .distributions import Dirichlet
from .distributions import TruncatedGaussian
from .distributions import Wishart
from .distributions import InverseGaussian


class BayesianFixedNIGMixturePosterior(BayesianNIGMixturePosterior):

    def __init__(self, l, r, f, g, s, t, u, v, w, m, n):
        self.params = l, r, f, g, s, t, u, v, w, m, n
        self.alpha = Dirichlet(l, r)
        self.lmd = TruncatedGaussian(f, np.sqrt(g))
        self.tau = Wishart(s, t, inv=True)
        self.mu_beta = _MeanBias(self.tau, u, v, w, m, n)
        self._save_x = None

    def cross_entropy(self, l0, r0, f0, g0, s0, t0, u0, v0, w0, m0, n0):
        return np.array([
            self.alpha.cross_entropy(l0, r0),
            self.lmd.cross_entropy(f0, np.sqrt(g0)),
            self.tau.cross_entropy(s0, t0, inv=True),
            self.mu_beta.cross_entropy(u0, v0, w0, m0, n0),
        ])

    def _expect(self, x):
        if x is not self._save_x:
            zmean, zcorr, zbias = self.mu_beta.mahalanobis_factors(x)
            d = x.shape[-1]

            a = self.lmd.moment + zbias
            b = 1 + zmean
            c = - (d + 1) / 2
            ydist = InverseGaussian(a, b, c, halfint=True)

            yz = (
                + sp.xlogy(c, 2 * np.pi)
                + self.alpha.log_mean
                + self.lmd.mean - zcorr
                + self.tau.log_det_mean / 2
                - ydist.log_const
            )

            self._save_x = x
            self._save_yz = yz
            self._save_ydist = ydist
        return self._save_yz, self._save_ydist
