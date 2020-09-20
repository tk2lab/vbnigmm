__author__ = 'TAKEKAWA Takashi <takekawa@tk2lab.org>'
__credits__ = 'Copyright 2020, TAKEKAWA Takashi'


import numpy as np
import scipy.special as sp

from .basedist import BaseDist


class BlockNormal(BaseDist):

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
