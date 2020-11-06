"""Gamma distributions."""

__author__ = 'TAKEKAWA Takashi <takekawa@tk2lab.org>'
__credits__ = 'Copyright 2018, TAKEKAWA Takashi'


import numpy as np
import scipy.special as sp

from .basedist import BaseDist


class Gamma(BaseDist):

    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    @property
    def mean(self):
        return self.alpha / self.beta

    @property
    def mean_log(self):
        return sp.digamma(self.alpha) - np.log(self.beta)

    def log_pdf(self, x):
        g, h = self.alpha, self.beta
        x = x.mean if hasattr(x, 'mean') else x
        logx = x.mean_log if hasattr(x, 'mean_log') else x
        return sp.xlogy(g, h) - sp.gammaln(g) + (g - 1) * logx - h * x
