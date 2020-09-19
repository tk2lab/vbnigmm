__author__ = 'TAKEKAWA Takashi <takekawa@tk2lab.org>'
__credits__ = 'Copyright 2020, TAKEKAWA Takashi'


import numpy as np
import scipy.special as sp

from .distributions import BaseDist
from .distributions import Dirichlet
from .distributions import InverseGaussian
from .distributions import Wishart


class BayesianNIGMixturePosterior(BaseDist):

    def __init__(self, l, r, f, g, h, s, t, u, v, w, m, n):
        self.params = l, r, f, g, h, s, t, u, v, w, m, n
        self.alpha = Dirichlet(l, r)
        self.lmd = InverseGaussian(f, g, h)
        self.tau = Wishart(s, t, inv=True)
        self.mu_beta = _MeanBias(self.tau, u, v, w, m, n)
        self._save_x = None

    @property
    def weight(self):
        return self.alpha.mean

    @property
    def normality(self):
        return self.lmd.mean

    @property
    def mean(self):
        return self.mu_beta.mean + self.mu_beta.bias

    @property
    def bias_mode(self):
        lmd_mean = self.lmd.mean
        y = InverseGaussian(lmd_mean, lmd_mean)
        return y.mode[:, None] * self.mu_beta.bias

    @property
    def covariance_mode(self):
        lmd_mean = self.lmd.mean
        y = InverseGaussian(lmd_mean, lmd_mean)
        return y.mode[:, None, None] * np.linalg.inv(self.tau.mean)

    def cross_entropy(self, l0, r0, f0, g0, h0, s0, t0, u0, v0, w0, m0, n0):
        return np.array([
            self.alpha.cross_entropy(l0, r0),
            self.lmd.cross_entropy(f0, g0, h0),
            self.tau.cross_entropy(s0, t0, inv=True),
            self.mu_beta.cross_entropy(u0, v0, w0, m0, n0),
        ])

    def log_pdf(self, x):
        yz, ydist = self._expect(x)
        return sp.logsumexp(yz, axis=-1)

    def expect(self, x):
        yz, ydist = self._expect(x)
        return sp.softmax(yz, axis=-1)

    def expect_all(self, x):
        yz, ydist = self._expect(x)
        return sp.softmax(yz, axis=-1), ydist.mean, ydist.inv_mean

    def _expect(self, x):
        if x is not self._save_x:
            zmean, zcorr, zbias = self.mu_beta.mahalanobis_factors(x)
            d = x.shape[-1]

            a = self.lmd.mean + zbias
            b = self.lmd.mean + zmean
            c = - (d + 1) / 2
            ydist = InverseGaussian(a, b, c, halfint=True)

            yz = (
                + sp.xlogy(c, 2 * np.pi)
                + self.alpha.log_mean
                + 0.5 * self.lmd.log_mean
                + self.lmd.mean - zcorr
                + 0.5 * self.tau.log_det_mean
                - ydist.log_const
            )

            self._save_x = x
            self._save_yz = yz
            self._save_ydist = ydist
        return self._save_yz, self._save_ydist


class _MeanBias(BaseDist):

    def __init__(self, tau, *args):
        self.tau = tau
        self.params = list(map(np.asarray, args))

    @property
    def mean(self):
        u, v, w, m, n = self.params
        return m

    @property
    def bias(self):
        u, v, w, m, n = self.params
        return n

    def cross_entropy(self, *args):
        u, v, w, m, n = map(np.asarray, args)
        u1, v1, w1, m1, n1 = self.params
        d = m.shape[-1]
        tau = self.tau.mean
        dm = m1 - m
        dn = n1 - n
        dmtau = np.sum(dm[..., None, :] * tau, axis=-1)
        dntau = np.sum(dn[..., None, :] * tau, axis=-1)
        return -(
            - sp.xlogy(d, 2 * np.pi)
            + sp.xlogy(d / 2, u * v - w * w)
            + self.tau.log_det_mean
            - u * (d / u1 + np.sum(dmtau * dm, axis=-1)) / 2
            - w * (d / w1 + np.sum(dmtau * dn, axis=-1))
            - v * (d / v1 + np.sum(dntau * dn, axis=-1)) / 2
        )

    def mahalanobis_factors(self, x):
        u, v, w, m, n = self.params
        d = m.shape[-1]
        tau = self.tau.mean
        x = x[..., None, :] - m
        n = 0 - n
        xt = np.sum(x[..., None, :] * tau, axis=-1)
        nt = np.sum(n[..., None, :] * tau, axis=-1)
        return (
            d / u + np.sum(xt * x, axis=-1),
            d / w + np.sum(xt * n, axis=-1),
            d / v + np.sum(nt * n, axis=-1),
        )
