import numpy as np

from .dist import Dist
from ..linbase.scalar import Scalar, wrap_scalar
from ..math import log, lgamma, digamma


class Gamma(Dist, Scalar):

    def __init__(self, alpha, beta):
        self.alpha = np.asarray(alpha)
        self.beta = np.asarray(beta)

    @property
    def mean(self):
        return self.alpha / self.beta

    @property
    def mean_inv(self):
        return self.beta / (self.alpha - 1)

    @property
    def mean_log(self):
        return digamma(self.alpha) - log(self.beta)

    def log_pdf(self, x):
        g, h = self.alpha, self.beta
        x = wrap_scalar(x)
        return g * log(h) - lgamma(g) + (g - 1) * x.mean_log - h * x.mean
