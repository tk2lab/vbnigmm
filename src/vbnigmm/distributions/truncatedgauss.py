"""Truncated Gaussian distributions."""

__author__ = 'TAKEKAWA Takashi <takekawa@tk2lab.org>'
__credits__ = 'Copyright 2020, TAKEKAWA Takashi'


import numpy as np
import scipy.special as sp

from .basedist import BaseDist


class TruncatedGaussian(BaseDist):

    def __init__(self, *args):
        m, t = map(np.asarray, args)
        mt = m * t / np.sqrt(2)
        tmp = np.exp(-mt * mt) / (1 + sp.erf(mt)) / np.sqrt(np.pi)
        self.params = m, t
        self.mean = np.sqrt(2) * (mt + tmp) / t
        self.moment = (2 * mt * mt + 2 * tmp * mt + 1) / (t * t)
    
    def cross_entropy(self, *args):
        m, t = map(np.asarray, args)
        mt = m * t / np.sqrt(2)
        return - (
            - mt * mt - np.log((1 + sp.erf(mt)) / t) - np.log(np.pi / 2)
            + t * t * (m * self.mean - self.moment / 2)
        )
