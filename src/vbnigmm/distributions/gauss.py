__author__ = 'TAKEKAWA Takashi <takekawa@tk2lab.org>'
__credits__ = 'Copyright 2020, TAKEKAWA Takashi'


import numpy as np
import scipy.special as sp

from .basedist import BaseDist


class Gauss(BaseDist):

    def __init__(self, mu, tau):
        self.mu = mu
        self.tau = tau

    @property
    def dim(self):
        return self.shape[-1]

    @property
    def mean(self):
        return self.mu.mean if hasattr(self.mu, 'mean') else self.mu

    @property
    def presicion(self):
        return self.tau.mean if hasattr(self.tau, 'mean') else self.tau

    @property
    def log_det_precision(self):
        if hasattr(self.tau, 'mean_log_det'):
            return self.tau.mean_log_det
        return np.linalg.logdet(self.tau)

    def log_pdf(self, x):
        x = x.mean if hasattr(x, 'mean') else x
        mu = self.mean
        log_det_tau = self.log_det_precision
        dx = x - self.mu
        return (
            - math.log(2 * math.pi) / 2 * self.dim
            + (1 / 2) * self.tau.mean_log_det()
            - (1 / 2) * 

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
