import vbnigmm.math.base as tk
import tensorflow as tf

from .check import check_data
from .check import check_concentration
from .check import check_covariance
from .mixture import Mixture
from .dpm import DirichletProcess
from ..distributions.wishart import Wishart
from ..distributions.gauss import Gauss
from ..distributions.dist import Dist


class GaussMixtureParameters(Dist):

    def __init__(self, l, r, s, u, m, t, py=0.0):
        self.params = l, r, s, u, m, t
        self.size = tf.size(s)
        self.dim = tf.shape(m)[-1]

        self.alpha = DirichletProcess(l, r, py, tf.float32)
        self.tau = Wishart(s, tk.inv(t))
        self.mu = Gauss(m, self.tau * u)
        self.dists = [self.alpha, self.tau, self.mu]

    def log_pdf(self, x):
        return sum([s.log_pdf(d) for s, d in zip(self.dists, x.dists)], 0)


class GaussMixture(Mixture):

    Parameters = GaussMixtureParameters
    var_names = 'lrsumt'
    var_types = 'aassvm'

    def build_prior(self, x, mean=None, cov=None,
              concentration=1.0, concentration_decay=0.0,
              mean_scale=1.0, cov_scale=0.3, cov_reliability=2.0):
        m0, cov = check_data(x, mean, cov)
        l0, r0, py = check_concentration(concentration, concentration_decay)
        s0, t0 = check_covariance(cov_reliability, cov_scale, cov)
        u0 = (cov_scale / mean_scale) ** 2
        params = l0, r0, s0, u0, m0, t0, py
        params = [tk.as_array(p, tf.float32) for p in params]
        return GaussMixtureParameters(*params)

    def log_pdf(self, x, q=None):
        q = q or self.posterior
        return (
            q.alpha.mean_log
            - tk.cast(q.dim, self.dtype) * (tk.log2pi / 2)
            + (1 / 2) * q.tau.mean_log_det
            - (1 / 2) * (
                q.tau.trace_dot_inv(q.mu.precision)
                + q.tau.trace_dot_outer(x - q.mu.mean)
            )
        )

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
        t1 = (
            (t0 + u0 * m0[None, :] * m0[:, None])[:, :, None]
            - u1 * m1[None, :, :] * m1[:, None, :] + X2
        )
        m1 = tk.transpose(m1)
        t1 = tf.transpose(t1, (2, 0, 1))
        return self.Parameters(l1, r1, s1, u1, m1, t1)
