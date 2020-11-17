"""Wishart distributions."""

__author__ = 'TAKEKAWA Takashi <takekawa@tk2lab.org>'
__credits__ = 'Copyright 2018, TAKEKAWA Takashi'

import math

import numpy as np
import scipy.special as sp

from .basedist import BaseDist
from .math  import multilgamma, multidigamma


class Wishart(BaseDist):

    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    @property
    def mean(self):
        a, b = self.alpha, self.beta
        return b * a[..., None, None]

    @property
    def mean_inv(self):
        a, b = self.alpha, self.beta
        d = self.beta.shape[-1]
        return np.linalg.inv(b) / (a - d - 1)[..., None, None]

    @property
    def mean_log_det(self):
        a, b = self.alpha, self.beta
        d = b.shape[-1]
        return (
            + multidigamma(a / 2, d)
            + (math.log(2) * d + np.linalg.logdet(b))
        )

    def log_pdf(self, x):
        a, b = self.alpha, self.beta
        d = b.shape[-1]
        x = val(x)
        logdetx = logdet(x)
        return (
            - multilgamma(a / 2, d)
            - (a / 2) * (math.log(2) * d + np.linalg.logdet(b))
            + ((a - d - 1) / 2) * logdetx
            - (1 / 2) * (x * b).sum(axis=(-2, -1))
        )
