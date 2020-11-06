"""Dirichlet distributions."""

__author__ = 'TAKEKAWA Takashi <takekawa@tk2lab.org>'
__credits__ = 'Copyright 2018, TAKEKAWA Takashi'


import numpy as np
import scipy.special as sp

from .basedist import BaseDist


class Dirichlet(BaseDist):

    def __init__(self, alpha):
        self.alpha = alpha

    @property
    def precision(self):
        return alpha.sum(axis=-1)

    @property
    def mean(self):
        return self.alpha / self.precision

    @property
    def mean_log(self):
        return sp.digamma(self.alpha) - sp.digamma(self.precision)

    def log_pdf(self, x):
        log_x = x.mean_log if hasattr(x, 'mean_log') else np.log(x)
        return (
            + sp.gammaln(self.precision)
            - sp.gammaln(self.alpha).sum(axis=-1)
            + sp.xlogy(self.alpha - 1, x).sum(axis=-1)
        )
