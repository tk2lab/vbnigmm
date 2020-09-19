"""Dirichlet Process."""

__author__ = 'TAKEKAWA Takashi <takekawa@tk2lab.org>'
__credits__ = 'Copyright 2020, TAKEKAWA Takashi'


import numpy as np
import scipy.special as sp

from .basedist import BaseDist


class DirichletProcess(BaseDist):

    def __init__(self, *args):
        a, b = map(np.asarray, args)
        g = a / (a + b)
        digamma_ab = sp.digamma(a + b)
        log_g = sp.digamma(a) - digamma_ab
        log_gi = sp.digamma(b) - digamma_ab
        self.params = a, b
        self.mean = g * np.hstack((1, np.cumprod(1 - g[:-1])))
        self.log_mean = log_g + np.hstack((0, np.cumsum(log_gi)[:-1]))

    def cross_entropy(self, *args):
        a, b = map(np.asarray, args)
        a1, b1 = self.params
        return -(
            + sp.gammaln(a + b)
            - sp.gammaln(a) - sp.gammaln(b)
            - (a + b - 2) * sp.digamma(a1 + b1)
            + (a - 1) * sp.digamma(a1)
            + (b - 1) * sp.digamma(b1)
        )
