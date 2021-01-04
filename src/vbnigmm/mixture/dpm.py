from ..backend import current as tk

from ..distributions.base import Dist
from ..distributions.dirichlet import Dirichlet


class DirichletProcess(Dist):

    def __init__(self, alpha, beta, gamma=0, dtype=None):
        self.alpha = tk.as_array(alpha, dtype)
        self.beta = tk.as_array(beta, dtype)
        self.gamma = gamma

    def base(self, dim=None):
        if dim is None:
            dim = self.dim
        one = tk.ones(dim - 1, dtype=self.dtype)
        idx = tk.range(1, dim, dtype=self.dtype)
        alpha = self.alpha - one * self.gamma
        beta = self.beta + idx * self.gamma
        return Dirichlet(tk.stack((alpha, beta), axis=-1))

    @property
    def dtype(self):
        return self.alpha.dtype

    @property
    def dim(self):
        return tk.size(self.alpha) + 1

    @property
    def mean(self):
        x = self.base().mean
        m = tk.concat((x[..., 0], [1]), 0)
        n = tk.concat(([1], tk.cumprod(x[..., 1])), 0)
        return m * n

    @property
    def mean_log(self):
        logx = self.base().mean_log
        logm = tk.concat((logx[..., 0], [0]), 0)
        logn = tk.concat(([0], tk.cumsum(logx[..., 1])), 0)
        return logm + logn

    def log_pdf(self, x, condition=None):
        if isinstance(x, DirichletProcess):
            return tk.sum(self.base(x.dim).log_pdf(x.base()), axis=-1)
        x = x[:-1]
        y = x / (1 - tk.concat(([0], tk.cumprod(x[:-1]))))
        z = tk.stack((y, 1 - y), axis=1)
        return tk.sum(self.base().log_pdf(z), axis=-1)
