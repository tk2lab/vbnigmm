"""Dirichlet distributions."""

__author__ = 'TAKEKAWA Takashi <takekawa@tk2lab.org>'
__credits__ = 'Copyright 2018, TAKEKAWA Takashi'


import numpy as np
import scipy.special as sp

from .basedist import BaseDist
from .dpm import DirichletProcess


def Dirichlet(l, r=None):
    return DirichletDist(l) if r is None else DirichletProcess(l, r)


class DirichletDist(BaseDist):

    def __init__(self, *args):
        alpha, = (np.asarray(a) for a in args if a is not None)
        self.params = alpha,
        self.dof = alpha.sum()
        self.mean = alpha / alpha.sum()
        self.log_mean = sp.digamma(alpha) - sp.digamma(alpha.sum())

    def log_pdf(self, x):
        alpha, = self.params
        return (
            + sp.gammaln(alpha.sum())
            - sp.gammaln(alpha).sum()
            + sp.xlogy(alpha - 1, x).sum()
        )

    def cross_entropy(self, *args):
        alpha, = (np.asarray(a) for a in args if a is not None)
        return -(
            + sp.gammaln(alpha.sum()) / alpha.size
            - sp.gammaln(alpha)
            + (alpha - 1) * self.log_mean
        )
