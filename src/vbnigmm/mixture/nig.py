import vbnigmm.math.base as tk

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
from ..distributions.gauss import Gauss
from ..distributions.dist import Dist
from ..math.vector import precision


class NormalInverseGaussMixtureParameters(Dist):

    def __init__(self, l, r, f, g, h, s, t, u, v, w, m, n, py=0.0):
        self.params = l, r, f, g, h, s, t, u, v, w, m, n
        self.alpha = DirichletProcess(l, r, py)
        self.beta = InverseGauss(f, g, h)
        self.tau = Wishart(s, inv(t))
        self.mu = Gauss(m, self.tau * u)
        self.xi = Gauss(self.mu * (-w) + n, self.tau * v)

    def log_pdf(self, x):
        return (
            + self.alpha.log_pdf(x.alpha)
            + self.beta.log_pdf(x.beta)
            + self.tau.log_pdf(x.tau)
            + self.mu.log_pdf(x.mu)
            + self.xi.log_pdf(x.xi)
        )


class NormalInverseGaussMixture(Mixture):

    def setup(self, x, mean=None, cov=None, bias=None,
              concentration=1.0, concentration_decay=0.0,
              normality_prior_type='invgauss',
              normality_mean=3.0, normality_reliability=1.0,
              mean_scale=1.0, cov_scale=0.3, cov_reliability=2.0,
              bias_scale=0.3, mean_bias_corr=0.0):
        m0, cov = check_data(x, mean, cov)
        l0, r0, py = check_concentration(concentration, concentration_decay)
        s0, t0 = check_covariance(cov_reliability, cov_scale, cov)
        n0 = check_bias(m0, bias)
        f0, g0, h0 = check_normality(
            normality_prior_type, normality_mean, normality_reliability,
        )
        u0, v0, w0 = check_scale(
            mean_scale, cov_scale, bias_scale, mean_bias_corr,
        )
        self.prior = NormalInverseGaussMixtureParameters(
            l0, r0, f0, g0, h0, s0, t0, u0, v0, w0, m0, n0, py,
        )

    def _log_pdf(self, x, q=None):
        q = q or self.posterior
        d = x.shape[-1]
        sz = (
            + q.alpha.mean_log
            - (1 / 2) * tk.log2pi + (1 / 2) * q.beta.mean_log
            - (d / 2) * tk.log2pi + (1 / 2) * q.tau.mean_log_det
            + q.beta.mean
            - q.tau.trace_dot_inv(precision(q.xi, q.mu))
            + q.tau.trace_dot_outer(q.xi.mean, x - q.mu.mean)
        )
        sp = (
            + q.beta.mean
            + q.tau.trace_dot_inv(q.xi.precision)
            + q.tau.trace_dot_outer(q.xi.mean)
        )
        sm = (
            + q.beta.mean
            + q.tau.trace_dot_inv(q.mu.precision)
            + q.tau.trace_dot_outer(x - q.mu.mean)
        )
        y = InverseGauss(sp, sm, - (d + 1) / 2, halfint=True)
        return y, sz

    def log_pdf_joint(self, x, y, q=None):
        ydist, sz = self._log_pdf(x[..., None, :], q)
        return sz + ydist.log_pdf(y)

    def log_pdf(self, x, q=None):
        ydist, sz = self._log_pdf(x[..., None, :], q)
        return sz - ydist.log_const

    def start(self, x, y=None):
        z = super().start(x, y)
        return tk.tile(z, (3, 1, 1))

    def estep(self, x, q):
        y, sz = self._log_pdf(x[..., None, :], q)
        rho = sz - y.log_const
        z, ll = self.eval(rho)
        ym = z * y.mean_inv
        yz = z
        yp = z * y.mean
        return tk.stack((ym, yz, yp)), ll

    def mstep(self, x, z):
        ym, yz, yp = z
        Ym = ym.sum(axis=0)
        Yz = yz.sum(axis=0)
        Yp = yp.sum(axis=0)
        Xz = x.T @ yz
        Xm = x.T @ ym
        X2 = tk.tensordot(x, ym[:, None, :] * x[:, :, None], (0, 0))
        
        l0, r0, f0, g0, h0, s0, t0, u0, v0, w0, m0, n0 = self.prior.params
        l1 = l0 + Yz[:-1]
        r1 = r0 + tk.cumsum(Yz[:0:-1])[::-1]
        f1 = f0 + Yp + Ym - 2 * Yz
        g1 = tk.full_like(f1, g0)
        h1 = h0 + Yz / 2
        u1 = u0 + Ym
        v1 = v0 + Yp
        w1 = (v0 * w0 + Yz) / v1
        n1 = ((v0 * n0)[:, None] + Xz) / v1
        m1 = ((u0 * m0 + v0 * w0 * n0)[:, None] - v1 * w1 * n1 + Xm) / u1
        s1 = s0 + Yz
        t1 = (
            (t0 + u0 * m0 * m0[:, None] + v0 * n0 * n0[:, None])[:, :, None]
            - u1 * m1 * m1[:, None, :] - v1 * n1 * n1[:, None, :] + X2
        )
        return NormalInverseGaussMixtureParameters(
            l1, r1, f1, g1, h1, s1, t1.T, u1, v1, w1, m1.T, n1.T,
        )
