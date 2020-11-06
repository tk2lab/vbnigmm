"""Dirichlet Process."""

__author__ = 'TAKEKAWA Takashi <takekawa@tk2lab.org>'
__credits__ = 'Copyright 2020, TAKEKAWA Takashi'


import numpy as np
import scipy.special as sp

from .basedist import BaseDist


class Beta(BaseDist):

    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    @property
    def mean(self):
        a, b = self.alpha, self.beta
        return a / (a + b)

    @property
    def mean_log(self):
        a, b = self.alpha, self.beta
        return sp.digamma(a) - sp.digamma(a + b)

    def log_pdf(self, x):
        a, b = self.alpha, self.beta
        logx = x.mean_log if hasattr(x, 'mean_log') else np.log(x)
        log1mx = x.mean_log1m if hasattr(x, 'mean_log1m') else np.log(1 - x)
        return (
            sp.gammaln(a + b)
            - sp.gammaln(a)
            - sp.gammaln(b)
            + (a - 1) * logx
            + (b - 1) * log1mx
        )
