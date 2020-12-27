import numpy as np

from .check import check_data
from .check import check_concentration
from .check import check_covariance
from .mixture import Mixture
from .dpm import DirichletProcess
from ..distributions.wishart import Wishart
from ..distributions.gauss import Gauss
from ..distributions.dist import Dist
from ..math import log2pi, inv


class GaussMixtureParameters(Dist):

    def __init__(self, l, r, s, t, u, m, py=0.0):
        self.params = l, r, s, t, u, m
        self.alpha = DirichletProcess(l, r, py)
        self.tau = Wishart(s, inv(t))
        self.mu = Gauss(m, self.tau * u)

    def log_pdf(self, x):
        return (
            + self.alpha.log_pdf(x.alpha)
            + self.tau.log_pdf(x.tau)
            + self.mu.log_pdf(x.mu)
        )


class GaussMixture(Mixture):

    def setup(self, x, mean=None, cov=None,
              concentration=1.0, concentration_decay=0.0,
              mean_scale=1.0, cov_scale=0.3, cov_reliability=2.0):
        m0, cov = check_data(x, mean, cov)
        l0, r0, py = check_concentration(concentration, concentration_decay)
        s0, t0 = check_covariance(cov_reliability, cov_scale, cov)
        u0 = (cov_scale / mean_scale) ** 2
        self.prior = GaussMixtureParameters(l0, r0, s0, t0, u0, m0, py)

    def log_pdf(self, x, q=None):
        q = q or self.posterior
        d = x.shape[-1]
        return (
            + q.alpha.mean_log
            - (d / 2) * log2pi + (1 / 2) * q.tau.mean_log_det
            - (1 / 2) * (
                + q.tau.trace_dot_inv(q.mu.precision)
                + q.tau.trace_dot_outer(x - q.mu.mean)
            )
        )

    def estep(self, x, q):
        rho = self.log_pdf(x[:, None, :], q)
        z, ll, kl = self.eval(rho)
        return z[None, ...], ll, kl

    def mstep(self, x, z):
        z = z[0]
        Yz = z.sum(axis=0)
        Xz = x.T @ z
        X2 = np.tensordot(x, z[:, None, :] * x[:, :, None], (0, 0))
        
        l0, r0, s0, t0, u0, m0 = self.prior.params
        l1 = l0 + Yz[:-1]
        r1 = r0 + np.cumsum(Yz[:0:-1])[::-1]
        u1 = u0 + Yz
        m1 = ((u0 * m0)[:, None] + Xm) / u1
        s1 = s0 + Yz
        t1 = (
            (t0 + u0 * m0 * m0[:, None])[:, :, None]
            - u1 * m1 * m1[:, None, :] + X2
        )
        return GaussMixtureParameters(l1, r1, s1, t1.T, u1, m1.T)
