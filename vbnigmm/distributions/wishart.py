import vbnigmm.math.base as tk

from .dist import Dist
from ..math.matrix import Matrix, wrap_matrix


class Wishart(Dist, Matrix):

    def __init__(self, alpha, beta):
        self.alpha = tk.as_array(alpha)
        self.beta = tk.as_array(beta)

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
        return tk.inv(b) / (a - d - 1)[..., None, None]

    @property
    def mean_log_det(self):
        a, b, d = self.alpha, self.beta, self.dim
        return (
            tk.multi_digamma(a / 2, d) + d * tk.log2 + tk.log_det(b)
        )

    def log_pdf(self, x):
        a, b, d = self.alpha, self.beta, self.dim
        x = wrap_matrix(x)
        return (
            - tk.multi_lgamma(a / 2, d)
            - (a / 2) * (d * tk.log2 + tk.log_det(b))
            + ((a - d - 1) / 2) * x.mean_log_det
            - (1 / 2) * tk.sum(x.mean * tk.inv(b), axis=(-2, -1))
        )
