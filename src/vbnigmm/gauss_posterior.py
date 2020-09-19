__author__ = 'TAKEKAWA Takashi <takekawa@tk2lab.org>'
__credits__ = 'Copyright 2020, TAKEKAWA Takashi'


import numpy as np
import scipy.special as sp

from .distributions import BaseDist, Dirichlet, Wishart


class BayesianGaussianMixturePosterior(BaseDist):

    def __init__(self, l, r, s, t, u, m):
        alpha = Dirichlet(l, r)
        tau = Wishart(s, t, inv=True)
        mu = _Mean(tau, u, m)
        self.params = l, r, s, t, u, m
        self._dists = alpha, tau, mu

    @property
    def weight(self):
        alpha, tau, mu = self._dists
        return alpha.mean

    @property
    def mean(self):
        l, r, s, t, u, m = self.params
        return m

    @property
    def covariance(self):
        alpha, tau, mu = self._dists
        return np.linalg.inv(tau.mean)

    def log_pdf(self, x):
        return sp.logsumexp(self._expect(x), axis=-1)

    def expect(self, x):
        return sp.softmax(self._expect(x), axis=-1)

    def expect_all(self, x):
        return sp.softmax(self._expect(x), axis=-1),

    def cross_entropy(self, a0, b0, s0, t0, u0, m0):
        alpha1, tau1, mu1 = self._dists
        return np.array([
            alpha1.cross_entropy(a0, b0),
            tau1.cross_entropy(s0, t0, inv=True),
            mu1.cross_entropy(u0, m0),
        ])

    def _expect(self, x):
        alpha, tau, mu = self._dists
        d = x.shape[-1]
        return (
            + alpha.log_mean
            - d * np.log(2 * np.pi) / 2
            + tau.log_det_mean / 2
            - mu.mahalanobis(x) / 2
        )


class _Mean(BaseDist):

    def __init__(self, tau, *args):
        u, m = map(np.asarray, args)
        self.tau = tau
        self.params = u, m

    def cross_entropy(self, *args):
        u, m = map(np.asarray, args)
        u1, m1 = self.params
        d = m.shape[-1]
        x = m1 - m
        xtau = np.sum(x[..., None, :] * self.tau.mean, axis=-1)
        return - (
            - sp.xlogy(d / 2, 2 * np.pi)
            + sp.xlogy(d / 2, u)
            + self.tau.log_det_mean / 2
            - u * (d / u1 + np.sum(xtau * x, axis=-1)) / 2
        )

    def mahalanobis(self, x):
        u, m = self.params
        d = m.shape[-1]
        x = x[..., None, :] - m
        xtau = np.sum(x[..., None, :] * self.tau.mean, axis=-1)
        return d / u + np.sum(xtau * x, axis=-1)