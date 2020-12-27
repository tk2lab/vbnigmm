import numpy as np

from .dist import Dist
from ..linbase.vector import Vector, wrap_vector, precision
from ..linbase.matrix import Matrix, wrap_matrix
from ..math import log2pi


class Gauss(Dist, Vector):

    def __init__(self, mu, tau):
        self.mu = wrap_vector(mu)
        self.tau = wrap_matrix(tau)

    @property
    def dim(self):
        return self.mu.dim

    @property
    def mean(self):
        return self.mu.mean

    @property
    def precision(self):
        return self.tau

    def log_pdf(self, x):
        x = wrap_vector(x)
        dx = x.mean - self.mu.mean
        dx2 = dx[..., None, :] * dx[..., :, None]
        return (
            - (self.dim / 2) * log2pi
            + (1 / 2) * self.tau.mean_log_det
            - (1 / 2) * self.tau.trace_dot(dx2)
            - (1 / 2) * self.tau.trace_dot_inv(x.precision)
            - (1 / 2) * self.tau.trace_dot_inv(self.mu.precision)
            + self.tau.trace_dot_inv(precision(self.mu, x))
        )
