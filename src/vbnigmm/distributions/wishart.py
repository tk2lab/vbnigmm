"""Wishart distributions."""

__author__ = 'TAKEKAWA Takashi <takekawa@tk2lab.org>'
__credits__ = 'Copyright 2018, TAKEKAWA Takashi'


import numpy as np
import scipy.special as sp

from .basedist import BaseDist
from .math  import multidigamma


class Wishart(BaseDist):

    def __init__(self, s, t, inv=False):
        s, t = map(np.asarray, (s, t))
        if inv:
            t = np.linalg.inv(t)
        d = t.shape[-1]
        self.params = s, t
        self.prec_factor = s
        self.mean = t * s[..., None, None]

    @property
    def log_det_mean(self):
        s, t = self.params
        d = t.shape[-1]
        return multidigamma(s / 2, d) + np.linalg.slogdet(t * 2)[1]

    def cross_entropy(self, s, t, inv=False):
        s, t = map(np.asarray, (s, t))
        d = t.shape[-1]
        if inv:
            return - (
                + s * np.linalg.slogdet(t / 2)[1] / 2
                - sp.multigammaln(s / 2, d)
                + (s - d - 1) * self.log_det_mean / 2
                - (t * self.mean).sum(axis=(-2, -1)) / 2
            )
        else:
            return - (
                - s * np.linalg.slogdet(t * 2)[1] / 2
                - sp.multigammaln(s / 2, d)
                + (s - d - 1) * self.log_det_mean / 2
                - (np.linalg.inv(t) * self.mean).sum(axis=(-2, -1)) / 2
            )
