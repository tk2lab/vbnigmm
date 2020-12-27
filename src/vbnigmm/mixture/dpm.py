import numpy as np

from ..distributions.dist import Dist
from ..distributions.dirichlet import Dirichlet


class DirichletProcess(Dist):

    def __init__(self, alpha, beta, gamma=0):
        self.alpha = np.asarray(alpha)
        self.beta = np.asarray(beta)
        self.gamma = gamma

    def base(self, dim=None):
        dim = dim or self.dim
        alpha = (self.alpha - self.gamma) * np.ones(dim - 1)
        beta = self.beta + np.arange(1, dim) * self.gamma
        return Dirichlet(np.stack((alpha, beta), axis=-1))

    @property
    def dim(self):
        return self.alpha.size + 1

    @property
    def mean(self):
        x = self.base().mean
        m = np.hstack((x[..., 0], 1))
        n = np.hstack((1, np.cumprod(x[..., 1])))
        return m * n

    @property
    def mean_log(self):
        logx = self.base().mean_log
        logm = np.hstack((logx[..., 0], 0))
        logn = np.hstack((0, np.cumsum(logx[..., 1])))
        return logm + logn

    def log_pdf(self, x):
        if isinstance(x, DirichletProcess):
            return np.sum(self.base(x.dim).log_pdf(x.base()), axis=-1)
        x = x[:-1]
        y = x / (1 - np.hstack((0, np.cumprod(x[:-1]))))
        z = np.stack((y, 1 - y), axis=1)
        return np.sum(self.base().log_pdf(z), axis=-1)
