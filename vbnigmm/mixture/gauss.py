from collections import namedtuple

import vbnigmm.math.base as tk

from .check import check_data
from .check import check_concentration
from .check import check_covariance

from .mixture import Mixture
from .utils import MixtureParameters
from .dpm import DirichletProcess
from ..distributions.wishart import Wishart
from ..distributions.gauss import Gauss


class GaussMixtureParameters(MixtureParameters):

    var_names = 'l', 'r', 's', 'u', 'm', 't'
    var_types = 'o', 'o', 's', 's', 'v', 'm'

    @classmethod
    def make_prior(cls, x, mean=None, cov=None,
              concentration=1.0, concentration_decay=0.0,
              mean_scale=1.0, cov_scale=0.3, cov_reliability=2.0):
        m0, cov = check_data(x, mean, cov)
        l0, r0, py = check_concentration(concentration, concentration_decay)
        s0, t0 = check_covariance(cov_reliability, cov_scale, cov)
        u0 = (cov_scale / mean_scale) ** 2
        return cls(l0, r0, s0, u0, m0, t0, py)

    def __init__(self, l, r, s, u, m, t, py=0.0):
        Params = namedtuple('Params', self.var_names)
        self.params = Params(l, r, s, u, m, t)
        self.size = tk.size(s)
        self.dim = tk.shape(m)[-1]

        self.alpha = DirichletProcess(l, r, py)
        self.tau = Wishart(s, tk.inv(t))
        self.mu = Gauss(m, self.tau * u)
        self.dists = [self.alpha, self.tau, self.mu]


class GaussMixture(Mixture):

    Parameters = GaussMixtureParameters

    def log_pdf(self, x):
        q = self.posterior
        x = x[:, None, :]
        d = float(x.shape[-1])
        p = (
            q.alpha.mean_log
            - (d / 2) * tk.log2pi + (1 / 2) * q.tau.mean_log_det
            - (1 / 2) * (
                q.tau.trace_dot_inv(q.mu.precision)
                + q.tau.trace_dot_outer(x - q.mu.mean)
            )
        )
        return p


    def get_constants(self):
        return dict()

    def init_expect(self, z):
        return z[None, :]

    def call(self, x):
        y = self.log_pdf(x)
        return y, (tk.softmax(y, axis=-1),)

    def mstep(self, x, z):
        z = z[0]
        Y = tk.sum(z, axis=0)
        X1 = tk.transpose(x) @ z
        X2 = tk.tensordot(x, z[:, None, :] * x[:, :, None], (0, 0))
        
        l0, r0, s0, u0, m0, t0 = self.prior.params
        l1 = l0 + Y[:-1]
        r1 = r0 + tk.cumsum(Y[:0:-1])[::-1]
        u1 = u0 + Y
        m1 = ((u0 * m0)[:, None] + X1) / u1
        s1 = s0 + Y
        tx = t0 + u0 * m0[None, :] * m0[:, None]
        ty = X2 - u1 * m1[None, :, :] * m1[:, None, :]
        t1 = tx[:, :, None] + ty
        m1 = tk.transpose(m1)
        t1 = tk.transpose(t1, (2, 0, 1))
        return self.Parameters(l1, r1, s1, u1, m1, t1)
