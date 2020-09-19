"""Gamma distributions."""

__author__ = 'TAKEKAWA Takashi <takekawa@tk2lab.org>'
__credits__ = 'Copyright 2018, TAKEKAWA Takashi'


import numpy as np
import scipy.special as sp

from .basedist import BaseDist


class Gamma(BaseDist):

    def __init__(self, *args):
        g, h = map(np.asarray, args)
        self.params = g, h
        self.dof = g
        self.mean = g / h
        self.log_mean = sp.digamma(g) - np.log(h)
        self.log_const = sp.xlogy(g, h) - sp.gammaln(g)

    def log_pdf(self, x):
        g, h = self.params
        return self.log_const + sp.xlogy(g - 1, x) - h * x

    def cross_entropy(self, *params):
        g, h = map(np.asarray, params)
        return -(
            + sp.xlogy(g, h)
            - sp.gammaln(g)
            + (g - 1) * self.log_mean
            - h * self.mean
        )
