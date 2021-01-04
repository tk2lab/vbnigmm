from ..backend import current as tk


def check_data(x, mean, cov):
    if x.ndim != 2:
        raise ValueError('x must be 2d')
    if mean is None:
        mean = tk.mean(x, axis=0)
    elif mean.ndim != 1:
        raise ValueError('mean must be 1d')
    if cov is None:
        cov = tk.cov(x)
    elif cov.ndim != 2:
        raise ValueError('cov must be 2d')
    return mean, cov


def check_bias(mean, bias):
    if bias is None:
        bias = tk.zeros_like(mean)
    elif bias.ndim != 1:
        raise ValueError('bias must be 1d')
    return bias


def check_concentration(alpha, py):
    return 1.0, alpha, py


def check_normality(prior_type, mean, reliability):
    if prior_type == 'invgauss':
        f0 = reliability
        g0 = reliability / mean / mean
        h0 = -1
    elif prior_type == 'gamma':
        f0 = reliability / mean
        g0 = 0.0
        h0 = reliability
    else:
        raise ValueError('normality_prior_type should be invgauss or gamma')
    return f0, g0, h0


def check_covariance(cov_reliability, cov_scale, cov):
    s0 = (cov.shape[0] - 1) + cov_reliability
    t0 = cov * cov_scale ** 2 * s0
    return s0, t0


def check_scale(cov, mean, bias):
    u0 = (cov / mean) ** 2
    v0 = 1 / bias ** 2
    return u0, v0
