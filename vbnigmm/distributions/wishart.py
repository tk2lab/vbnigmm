from ..backend import current as tk

from .base import Dist
from ..linalg.matrix import Matrix, wrap_matrix


class Wishart(Dist, Matrix):

    def __init__(self, alpha, beta, inv=False, dtype=None):
        self.alpha = tk.as_array(alpha, dtype)
        self.beta = tk.as_array(beta, dtype)
        self.inv_beta = tk.inv(self.beta)
        if inv:
            self.beta, self.inv_beta = self.inv_beta, self.beta

    @property
    def dtype(self):
        return self.alpha.dtype

    @property
    def dim(self):
        return self.beta.shape[-1]

    @property
    def d(self):
        return tk.cast(self.dim, self.alpha.dtype)

    @property
    def mean(self):
        return self.beta * self.alpha[..., None, None]

    @property
    def mean_inv(self):
        return self.inv_beta / (self.alpha - self.d - 1)[..., None, None]

    @property
    def mean_log_det(self):
        return (
            tk.multi_digamma(self.alpha / 2, self.dim)
            + self.d * tk.log2 + tk.log_det(self.beta)
        )

    def log_pdf(self, x):
        x = wrap_matrix(x, self.dtype)
        return (
            - tk.multi_lgamma(self.alpha / 2, self.dim)
            - (self.alpha / 2) * (self.d * tk.log2 + tk.log_det(self.beta))
            + ((self.alpha - self.d - 1) / 2) * x.mean_log_det
            - (1 / 2) * tk.sum(x.mean * self.inv_beta, axis=(-2, -1))
        )
