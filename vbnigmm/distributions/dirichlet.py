from ..backend import current as tk

from .base import Dist
from ..linalg.vector import Vector, wrap_vector


class Dirichlet(Dist, Vector):

    def __init__(self, alpha, dtype=None):
        self.alpha = tk.as_array(alpha, dtype)

    @property
    def dtype(self):
        return self.alpha.dtype

    @property
    def sum(self):
        return tk.sum(self.alpha, axis=-1)

    @property
    def mean(self):
        return self.alpha / self.sum[..., None]

    @property
    def mean_log(self):
        return tk.digamma(self.alpha) - tk.digamma(self.sum)[..., None]

    def log_pdf(self, x, condition=None):
        x = wrap_vector(x, self.dtype)
        return (
            tk.lgamma(self.sum)
            - tk.sum(tk.lgamma(self.alpha), axis=-1)
            + tk.sum((self.alpha - 1) * x.mean_log, axis=-1)
        )
