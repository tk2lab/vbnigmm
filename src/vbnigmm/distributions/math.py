"""Mathematical functions."""

__author__ = 'TAKEKAWA Takashi <takekawa@tk2lab.org>'
__credits__ = 'Copyright 2020, TAKEKAWA Takashi'


import numpy as np
import scipy.special as sp

from scipy.special import multigammaln as multilgamma


def multidigamma(x, d):
    return np.sum([sp.digamma(x - i / 2) for i in range(d)], axis=0)


def khratio(v, x):
    m = np.around(2 * v).astype(np.int32)
    negative = m < 0
    absm = -m if negative else m
    if absm % 2 == 0:
        nu = 1.0
        n = absm // 2
        ratio = sp.k1e(x) / sp.k0e(x)
    else:
        nu = 0.5
        n = (absm + 1) // 2
        ratio = np.full_like(x, 1.0)
    if negative:
        n -= 1
    ratio = update_kratio(nu, x, ratio, n)
    if negative:
        ratio = 1 / ratio
    return ratio


def k05(x):
    return sqrt(0.5 * pi / x) * exp(-x)


def update_kratio(nu, x, ratio, n):
    for i in range(n):
        ratio = 2 * nu / x + 1 / ratio
        nu += 1
    return ratio
