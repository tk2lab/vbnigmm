import vbnigmm.math.base as tk
import tensorflow as tf

from .check import check_data
from .check import check_concentration
from .check import check_covariance
from .check import check_bias
from .check import check_normality
from .check import check_scale
from .mixture import Mixture
from .dpm import DirichletProcess
from ..distributions.wishart import Wishart
from ..distributions.invgauss import InverseGauss
from ..distributions.gamma import Gamma
from ..distributions.gauss import Gauss
from ..distributions.dist import Dist
from ..math.vector import precision


class NormalInverseGaussMixtureParameters(Dist):

    def __init__(self, l, r, f, g, h, s, u, v, w, m, n, t, py=0.0, dtype=None):
        self.params = l, r, f, g, h, s, u, v, w, m, n, t
        self.size = tk.size(f)
        self.dim = tk.shape(t)[-1]

        self.alpha = DirichletProcess(l, r, py, dtype)
        self.beta = Gamma(h, f / 2) #InverseGauss(f, g, h)
        self.tau = Wishart(s, tk.inv(t))
        self.mu = Gauss(m, self.tau * u)
        self.xi = Gauss(self.mu * w + n, self.tau * v)
        self.dists = [self.alpha, self.beta, self.tau, self.mu, self.xi]

    def log_pdf(self, x):
        return sum([s.log_pdf(d) for s, d in zip(self.dists, x.dists)], 0)


class NormalInverseGaussMixture(Mixture):

    Parameters = NormalInverseGaussMixtureParameters
    var_names = 'lrfghsuvwnmt'
    var_types = 'aasssssssvvm'

    def build_prior(self, x, mean=None, cov=None, bias=None,
                    concentration=1.0, concentration_decay=0.0,
                    normality_mean=3.0, normality_dof=1.0,
                    cov_scale=0.3, cov_dof=2.0,
                    mean_scale=1.0, bias_scale=0.3, mean_bias_factor=0.0):
        m0, cov = check_data(x, mean, cov)
        n0 = check_bias(m0, bias)
        l0, r0, py = check_concentration(concentration, concentration_decay)
        s0, t0 = check_covariance(cov_dof, cov_scale, cov)
        f0, g0, h0 = check_normality('gamma', normality_mean, normality_dof)
        u0, v0 = check_scale(cov_scale, mean_scale, bias_scale)
        w0 = mean_bias_factor
        params = l0, r0, f0, g0, h0, s0, u0, v0, w0, m0, n0, t0, py
        return NormalInverseGaussMixtureParameters(*params, tk.float32)

    def log_pdf_joint(self, x, y, q=None):
        ydist, sz = self._log_pdf(x[..., None, :], q)
        return sz + ydist.log_pdf(y)

    def log_pdf(self, x, q=None):
        ydist, sz = self._log_pdf(x[..., None, :], q)
        return sz - ydist.log_const


    def start(self, x, y=None):
        z = super().start(x, y)
        return tk.tile(z, (3, 1, 1))

    def call(self, x):
        y, sz = self._log_pdf(x[..., None, :])
        z = tk.softmax(sz, axis=-1)
        ym = z * y.mean_inv
        yz = z
        yp = z * y.mean
        return tk.stack((ym, yz, yp))

    def _log_pdf(self, x, q=None):
        q = q or self.posterior
        d = tf.cast(q.dim, self.dtype)
        sz = (
            q.alpha.mean_log
            - (1 / 2) * tk.log2pi + (1 / 2) * q.beta.mean_log
            - (d / 2) * tk.log2pi + (1 / 2) * q.tau.mean_log_det
            + q.beta.mean
            - q.tau.trace_dot_inv(precision(q.xi, q.mu))
            + q.tau.trace_dot_outer(q.xi.mean, x - q.mu.mean)
        )
        sp = (
            q.beta.mean
            + q.tau.trace_dot_inv(q.xi.precision)
            + q.tau.trace_dot_outer(q.xi.mean)
        )
        sm = (
            q.beta.mean
            + q.tau.trace_dot_inv(q.mu.precision)
            + q.tau.trace_dot_outer(x - q.mu.mean)
        )
        y = InverseGauss(sp, sm, - (d + 1) / 2)
        return y, sz

    def mstep(self, x, z):
        ym, yz, yp = z[0], z[1], z[2]
        Ym = tk.sum(ym, axis=0)
        Yz = tk.sum(yz, axis=0)
        Yp = tk.sum(yp, axis=0)
        Xz = tk.transpose(x) @ yz
        Xm = tk.transpose(x) @ ym
        X2 = tk.tensordot(x, ym[:, None, :] * x[:, :, None], (0, 0))
        
        l0, r0, f0, g0, h0, s0, u0, v0, w0, m0, n0, t0 = self.prior.params
        l1 = l0 + Yz[:-1]
        r1 = r0 + tk.cumsum(Yz[:0:-1])[::-1]
        f1 = f0 + Yp + Ym - 2 * Yz
        g1 = g0 * tk.ones_like(f1)
        h1 = h0 + Yz / 2
        u1 = u0 + Ym
        v1 = v0 + Yp
        w1 = (v0 * w0 - Yz) / v1
        n1 = ((v0 * n0)[:, None] + Xz) / v1
        m1 = ((u0 * m0 - v0 * w0 * n0)[:, None] + v1 * w1 * n1 + Xm) / u1
        s1 = s0 + Yz
        t1 = (
            (t0 + u0 * m0 * m0[:, None] + v0 * n0 * n0[:, None])[:, :, None]
            - u1 * m1 * m1[:, None, :] - v1 * n1 * n1[:, None, :] + X2
        )
        m1 = tk.transpose(m1)
        n1 = tk.transpose(n1)
        t1 = tk.transpose(t1, (2, 0, 1))
        return NormalInverseGaussMixtureParameters(
            l1, r1, f1, g1, h1, s1, u1, v1, w1, m1, n1, t1
        )
