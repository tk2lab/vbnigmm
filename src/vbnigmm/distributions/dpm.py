"""Dirichlet Process."""

__author__ = 'TAKEKAWA Takashi <takekawa@tk2lab.org>'
__credits__ = 'Copyright 2020, TAKEKAWA Takashi'


import numpy as np
import scipy.special as sp

from .basedist import BaseDist, mean
from .dirichlet import Dirichlet


class DirichletProcess(BaseDist):

    def __init__(self, alpha):
        self.base = Dirichlet(alpha)

    @property
    def mean(self):
        m = self.base.mean
        return m[..., 0] * np.cumprod(m[..., 1], exclusive=True)

    @property
    def mean_log(self):
        logm = self.base.mean_log
        return logm[..., 0] + np.cumsum(logm[..., 1], exclusive=True)

    def log_pdf(self, x):
        x = mean(x)[..., 0]
        y = x / (1 - np.cumsum(x, exclusive=True))
        z = np.stack([y, 1 - y], axis=1)
        return np.sum(self.base.log_pdf(z), axis=-1)
