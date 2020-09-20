__author__ = 'TAKEKAWA Takashi <takekawa@tk2lab.org>'
__credits__ = 'Copyright 2020, TAKEKAWA Takashi'


import numpy as np

from .nig_model import BayesianNIGMixture
from .fixed_posterior import BayesianFixedNIGMixturePosterior
from .check import check_data
from .check import check_bias
from .check import check_concentration
from .check import check_covariance
from .check import check_scale


class BayesianFixedNIGMixture(BayesianNIGMixture):

    posterior_type = BayesianFixedNIGMixturePosterior

    def _get_prior(self, x, mean=None, cov=None, bias=None,
                   concentration_prior_type='dpm', l0=None, r0=None,
                   normality_mean=5.0, normality_reliability=1.0,
                   mean_scale=1.0, cov_scale=0.3, cov_reliability=2.0,
                   bias_scale=0.3, mean_bias_corr=0.0):
        m0, cov = check_data(x, mean, cov)
        n0 = check_bias(m0, bias)
        l0, r0 = check_concentration(concentration_prior_type, l0, r0)
        f0, g0 = normality_mean, normality_reliability / (normality_mean ** 2)
        s0, t0 = check_covariance(cov_reliability, cov_scale, cov)
        u0, v0, w0 = check_scale(
            mean_scale, cov_scale, bias_scale, mean_bias_corr,
        )
        return l0, r0, f0, g0, s0, t0, u0, v0, w0, m0, n0

    def _update_lambda(self, f0, g0, Yz, Yp, Ym):
        g1 = g0 + Yp
        f1 = (f0 * g0 + Yz) / g1
        return f1, g1
    
    def _update(self, x, yz, yp, ym,
                l0, r0, f0, g0, s0, t0, u0, v0, w0, m0, n0):
        yp, ym, Yz, Yp, Ym, Xz, Xm, X2 = self._stats(x, yz, yp, ym)
        l1, r1 = self._update_alpha(l0, r0, Yz)
        f1, g1 = self._update_lambda(f0, g0, Yz, Yp, Ym)
        s1, t1, u1, v1, w1, m1, n1 = self._update_tau(
            s0, t0, u0, v0, w0, m0, n0, Yz, Yp, Ym, Xz, Xm, X2
        )
        return l1, r1, f1, g1, s1, t1.T, u1, v1, w1, m1.T, n1.T
