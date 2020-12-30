import numpy as np


def check_data(x, mean, cov):
    if x.ndim != 2:
        raise ValueError('x must be 2d')
    if mean is None:
        mean = x.mean(axis=0)
    elif mean.ndim != 1:
        raise ValueError('mean must be 1d')
    if cov is None:
        cov = np.atleast_2d(np.cov(x.T))
    elif cov.ndim != 2:
        raise ValueError('cov must be 2d')
    return mean, cov


def check_concentration(alpha, py):
    return 1, alpha, py


def _check_concentration(prior_type, l0, r0):
    if (l0 == None) and (r0 == None):
        if prior_type == 'dd':
            l0 = 1.0
        elif prior_type == 'dpm':
            l0 = r0 = 1.0
    return l0, r0


def check_covariance(cov_reliability, cov_scale, cov):
    s0 = (cov.shape[0] - 1) + cov_reliability
    t0 = cov * cov_scale ** 2 * s0
    return s0, t0


def check_normality(prior_type, mean, reliability):
    if prior_type == 'invgauss':
        f0 = reliability / mean
        g0 = reliability * mean
        h0 = -1 / 2
    elif prior_type == 'gamma':
        f0 = 2 * reliability / mean
        g0 = 0
        h0 = reliability
    else:
        raise ValueError('normality_prior_type should be invgauss or gamma')
    return f0, g0, h0


def check_bias(mean, bias):
    if bias is None:
        bias = np.zeros_like(mean)
    elif bias.ndim != 1:
        raise ValueError('bias must be 1d')
    return bias


def check_scale(mean_scale, cov_scale, bias_scale, mean_bias_corr):
    corrinv = (1 - mean_bias_corr ** 2)
    covmean = cov_scale / mean_scale
    u0 = covmean ** 2 / corrinv
    w0 = covmean * mean_bias_corr / bias_scale / corrinv
    v0 = 1 / bias_scale ** 2 / corrinv
    return u0, v0, w0
