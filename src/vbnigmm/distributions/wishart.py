import numpy as np

from .dist import Dist
from ..linbase.matrix import Matrix, wrap_matrix
from ..math  import log2, multi_lgamma, multi_digamma, log_det, inv


class Wishart(Dist, Matrix):

    def __init__(self, alpha, beta):
        self.alpha = np.asarray(alpha)
        self.beta = np.asarray(beta)

    @property
    def dim(self):
        return self.beta.shape[-1]

    @property
    def mean(self):
        a, b = self.alpha, self.beta
        return b * a[..., None, None]

    @property
    def mean_inv(self):
        a, b, d = self.alpha, self.beta, self.dim
        return inv(b) / (a - d - 1)[..., None, None]

    @property
    def mean_log_det(self):
        a, b, d = self.alpha, self.beta, self.dim
        return (
            + multi_digamma(a / 2, d)
            + d * log2 + log_det(b)
        )

    def log_pdf(self, x):
        a, b, d = self.alpha, self.beta, self.dim
        x = wrap_matrix(x)
        return (
            - multi_lgamma(a / 2, d)
            - (a / 2) * (d * log2 + log_det(b))
            + ((a - d - 1) / 2) * x.mean_log_det
            - (1 / 2) * (x.mean * inv(b)).sum(axis=(-2, -1))
        )
