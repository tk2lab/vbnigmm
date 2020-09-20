__author__ = 'TAKEKAWA Takashi <takekawa@tk2lab.org>'
__credits__ = 'Copyright 2020, TAKEKAWA Takashi'


import numpy as np

from .mixturebase import MixtureBase
from .nig_posterior import BayesianNIGMixturePosterior
from .check import check_data
from .check import check_bias
from .check import check_concentration
from .check import check_normality
from .check import check_covariance
from .check import check_scale


class BayesianNIGMixture(MixtureBase):

    posterior_type = BayesianNIGMixturePosterior

    def __init__(self, yp0=1.0, ym0=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.yp0 = yp0
        self.ym0 = ym0

    def _get_prior(self, x, mean=None, cov=None, bias=None,
                   concentration_prior_type='dpm', l0=None, r0=None,
                   normality_prior_type='invgauss',
                   normality_mean=3.0, normality_reliability=1.0,
                   mean_scale=1.0, cov_scale=0.3, cov_reliability=2.0,
                   bias_scale=0.3, mean_bias_corr=0.0):
        m0, cov = check_data(x, mean, cov)
        n0 = check_bias(m0, bias)
        l0, r0 = check_concentration(concentration_prior_type, l0, r0)
        f0, g0, h0 = check_normality(
            normality_prior_type, normality_mean, normality_reliability,
        )
        s0, t0 = check_covariance(cov_reliability, cov_scale, cov)
        u0, v0, w0 = check_scale(
            mean_scale, cov_scale, bias_scale, mean_bias_corr,
        )
        return l0, r0, f0, g0, h0, s0, t0, u0, v0, w0, m0, n0

    def _init_expect(self, z):
        return z, np.full_like(z, self.yp0), np.full_like(z, self.ym0)

    def _stats(self, x, yz, yp, ym):
        yp = yz * yp
        ym = yz * ym
        Yz = yz.sum(axis=0)
        Yp = yp.sum(axis=0)
        Ym = ym.sum(axis=0)
        Xz = x.T @ yz
        Xm = x.T @ ym
        X2 = np.tensordot(x, ym[:, None, :] * x[:, :, None], (0, 0))
        return yp, ym, Yz, Yp, Ym, Xz, Xm, X2

    def _update_alpha(self, l0, r0, Yz):
        l1 = l0 + Yz
        if r0 is None:
            r1 = None
        else:
            r1 = r0 + np.hstack((np.cumsum(Yz[::-1])[-2::-1], 0))
        return l1, r1

    def _update_lambda(self, f0, g0, h0, Yz, Yp, Ym):
        f1 = f0 + Yp + Ym - 2 * Yz
        g1 = np.full_like(f1, g0)
        h1 = h0 + Yz / 2
        return f1, g1, h1
    
    def _update_tau(self, s0, t0, u0, v0, w0, m0, n0, Yz, Yp, Ym, Xz, Xm, X2):
        u1 = u0 + Ym
        w1 = w0 + Yz
        v1 = v0 + Yp
        s1 = s0 + Yz

        bm = (u0 * m0 + w0 * n0)[:, None] + Xm
        bn = (w0 * m0 + v0 * n0)[:, None] + Xz
        det1 = u1 * v1 - w1 * w1
        m1 = (+ v1 * bm - w1 * bn) / det1
        n1 = (- w1 * bm + u1 * bn) / det1

        t0 = t0.copy()
        t0 += u0 * m0 * m0[:, None]
        t0 += w0 * m0 * n0[:, None]
        t0 += w0 * n0 * m0[:, None]
        t0 += v0 * n0 * n0[:, None]
        t1 = t0[:, :, None] + X2
        t1 -= u1 * m1 * m1[:, None, :]
        t1 -= w1 * m1 * n1[:, None, :]
        t1 -= w1 * n1 * m1[:, None, :]
        t1 -= v1 * n1 * n1[:, None, :]
        return s1, t1, u1, v1, w1, m1, n1
    
    def _update(self, x, yz, yp, ym,
                l0, r0, f0, g0, h0, s0, t0, u0, v0, w0, m0, n0):
        yp, ym, Yz, Yp, Ym, Xz, Xm, X2 = self._stats(x, yz, yp, ym)
        l1, r1 = self._update_alpha(l0, r0, Yz)
        f1, g1, h1 = self._update_lambda(f0, g0, h0, Yz, Yp, Ym)
        s1, t1, u1, v1, w1, m1, n1 = self._update_tau(
            s0, t0, u0, v0, w0, m0, n0, Yz, Yp, Ym, Xz, Xm, X2
        )
        return l1, r1, f1, g1, h1, s1, t1.T, u1, v1, w1, m1.T, n1.T
