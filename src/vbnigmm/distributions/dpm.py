"""Dirichlet Process."""

__author__ = 'TAKEKAWA Takashi <takekawa@tk2lab.org>'
__credits__ = 'Copyright 2020, TAKEKAWA Takashi'


import numpy as np
import scipy.special as sp

from .basedist import BaseDist


class DirichletProcess(BaseDist):

    def __init__(self, alpha):
        self.base = Dirichlet(alpha)

    @property
    def mean(self):
        m = self.base.mean()
        return m * np.cumprod(1 - m, exclusive=True)

    @property
    def mean_log(self):
        logx = self.base.mean_log()
        log1mx = self.base.mean_log1m()
        return logx + np.cumsum(log1mx, exclusive=True)

    def log_pdf(self, x):
        x = x.mean if hasattr(x, 'mean') else x
        y = x / (1 - np.cumsum(x, exclusive=True))
        return np.sum(self.base.log_pdf(y), axis=-1)
