from .. import backend as tk

from .mixture import Mixture
from .nig_params import NormalInverseGaussMixtureParameters

from ..distributions.invgauss import InverseGauss
from ..linalg.vector import precision


class NormalInverseGaussMixture(Mixture):

    Parameters = NormalInverseGaussMixtureParameters

    def log_pdf_joint(self, x, y):
        sz, ydist = self._log_pdf(x)
        return sz + ydist.log_pdf(y)

    def log_pdf(self, x):
        sz, ydist = self._log_pdf(x)
        return sz - ydist.log_const

    def _log_pdf(self, x):
        q = self.posterior
        x = x[..., None, :]
        d = float(x.shape[-1])
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
        return sz, InverseGauss(sp, sm, - (d + 1) / 2, x.dtype)


    def get_constants(self):
        return dict(g=self.prior.params.g)

    def get_conditions(self, q):
        return dict(tau=q.tau, mu=q.mu)

    def init_expect(self, z):
        return tk.tile(z[None, ...], (3, 1, 1))

    def call(self, x):
        sz, y = self._log_pdf(x)
        l = sz - y.log_const
        z = tk.softmax(l, axis=-1)
        yz = z
        yp = z * y.mean
        ym = z * y.mean_inv
        return l, (yz, yp, ym)

    def mstep(self, x, z):
        yz, yp, ym = z[0], z[1], z[2]
        Ym = tk.sum(ym, axis=0)
        Yz = tk.sum(yz, axis=0)
        Yp = tk.sum(yp, axis=0)
        Xm = tk.transpose(x) @ ym
        Xz = tk.transpose(x) @ yz
        X2 = tk.tensordot(x, ym[:, None, :] * x[:, :, None], (0, 0))
        
        l0, r0, f0, g0, h0, s0, u0, v0, w0, m0, n0, t0 = self.prior.params
        l1 = l0 + Yz[:-1]
        r1 = r0 + tk.cumsum(Yz[:0:-1])[::-1]
        f1 = f0 + Yp + Ym - 2 * Yz
        g1 = g0
        h1 = h0 + Yz
        v1 = v0 + Yp
        w1 = (v0 * w0 - Yz) / v1
        u1 = u0 + v0 * w0 * w0 - v1 * w1 * w1 + Ym
        n1 = ((v0 * n0)[..., None] + Xz) / v1
        m1 = ((u0 * m0 - v0 * w0 * n0)[..., None] + v1 * w1 * n1 + Xm) / u1
        s1 = s0 + Yz
        tx = t0 + u0 * m0 * m0[:, None] + v0 * n0 * n0[:, None]
        ty = X2 - u1 * m1 * m1[:, None, :] - v1 * n1 * n1[:, None, :]
        t1 = tx[..., None] + ty
        m1 = tk.transpose(m1)
        n1 = tk.transpose(n1)
        t1 = tk.transpose(t1, (2, 0, 1))
        return self.Parameters(l1, r1, f1, g1, h1, s1, u1, v1, w1, m1, n1, t1, dtype=x.dtype)
