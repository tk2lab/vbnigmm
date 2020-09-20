__author__ = 'TAKEKAWA Takashi <takekawa@tk2lab.org>'
__credits__ = 'Copyright 2020, TAKEKAWA Takashi'


import numpy as np
import scipy.special as sp

from .basedist import BaseDist


class MultivariateNormal(BaseDist):

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
