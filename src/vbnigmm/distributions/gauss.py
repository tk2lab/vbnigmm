__author__ = 'TAKEKAWA Takashi <takekawa@tk2lab.org>'
__credits__ = 'Copyright 2020, TAKEKAWA Takashi'

import math

import numpy as np
import scipy.special as sp

from .basedist import BaseDist


class Gauss(BaseDist):

    def __init__(self, mu, tau, u=1):
        self.mu = mu
        self.u = u
        self.tau = tau

    @property
    def dim(self):
        return self.mu.shape[-1]

    @property
    def mean(self):
        return self.mu

    @property
    def presicion(self):
        return self.tau

    def log_pdf(self, x):
        dim = self.dim
        mu = self.mean
        tau = self.presicion
        log_det_tau = self.log_det_precision
        x = x.mean if hasattr(x, 'mean') else x
        dx = x - mu
        outer = dx[..., :, None] * dx[..., None, :]
        outer += 
        return (
            - (d / 2) * math.log(2 * math.pi)
            + (d / 2) * log_u
            + (1 / 2) * log_det_precision
            - (1 / 2) * np.sum(precision * outer, axis=(-2, -1))
        )
        if hasattr(x, 'precision'):
            if np.all(x.precision == tau):
                r += dim / 
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
