from ..backend import current as tk

from .base import Dist
from ..linalg.scalar import Scalar, wrap_scalar


class Gamma(Dist, Scalar):

    def __init__(self, alpha, beta):
        self.alpha = tk.as_array(alpha)
        self.beta = tk.as_array(beta)

    @property
    def mode(self):
        return (self.a - 1) / self.beta

    @property
    def mean(self):
        return self.alpha / self.beta

    @property
    def mean_inv(self):
        return self.beta / (self.alpha - 1)

    @property
    def mean_log(self):
        return tk.digamma(self.alpha) - tk.log(self.beta)

    @property
    def log_const(self):
        return self.alpha * tk.log(self.beta) - tk.lgamma(self.alpha)

    def log_pdf(self, x):
        x = wrap_scalar(x)
        return (
            self.log_const
            + (self.alpha - 1) * x.mean_log - self.beta * x.mean
        )
