import numpy as np

from .dist import Dist
from ..linbase.vector import Vector, wrap_vector
from ..math import xlogy, lgamma, digamma


class Dirichlet(Dist, Vector):

    def __init__(self, alpha):
        self.alpha = np.asarray(alpha)

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
        x = wrap_vector(x)
        return (
            + lgamma(self.sum)
            - lgamma(self.alpha).sum(axis=-1)
            + ((self.alpha - 1) * x.mean_log).sum(axis=-1)
        )
