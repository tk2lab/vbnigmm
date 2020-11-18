"""Dirichlet distributions."""

__author__ = 'TAKEKAWA Takashi <takekawa@tk2lab.org>'
__credits__ = 'Copyright 2018, TAKEKAWA Takashi'


import numpy as np

from .basedist import BaseDist, mean_log
from .math import xlogy, lgamma, digamma


class Dirichlet(BaseDist):

    def __init__(self, alpha):
        self.alpha = alpha

    @property
    def sum(self):
        return self.alpha.sum(axis=-1)

    @property
    def mean(self):
        return self.alpha / self.sum[..., None]

    @property
    def mean_log(self):
        return digamma(self.alpha) - digamma(self.sum)[..., None]

    def log_pdf(self, x):
        return (
            + lgamma(self.sum)
            - lgamma(self.alpha).sum(axis=-1)
            + ((self.alpha - 1) * mean_log(x)).sum(axis=-1)
        )
