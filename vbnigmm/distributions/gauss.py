from ..backend import current as tk

from .base import Dist
from ..linalg.vector import Vector, wrap_vector, precision
from ..linalg.matrix import Matrix, wrap_matrix


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
        return (
            (self.dim / 2) * tk.log2pi + (1 / 2) * self.tau.mean_log_det
            - (1 / 2) * (
                self.tau.trace_dot_outer(x.mean - self.mu.mean)
                + self.tau.trace_dot_inv(x.precision)
                + self.tau.trace_dot_inv(self.mu.precision)
                - 2 * self.tau.trace_dot_inv(precision(x, self.mu))
            )
        )
