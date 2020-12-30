import vbnigmm.math.base as tk

from .dist import Dist
from ..math.scalar import Scalar, wrap_scalar


class Gamma(Dist, Scalar):

    def __init__(self, alpha, beta):
        self.alpha = tk.as_array(alpha)
        self.beta = tk.as_array(beta)

    @property
    def mean(self):
        return self.alpha / self.beta

    @property
    def mean_inv(self):
        return self.beta / (self.alpha - 1)

    @property
    def mean_log(self):
        return tk.digamma(self.alpha) - tk.log(self.beta)

    def log_pdf(self, x):
        g, h = self.alpha, self.beta
        x = wrap_scalar(x)
        return (
            g * tk.log(h) - tk.lgamma(g)
            + (g - 1) * x.mean_log - h * x.mean
        )
