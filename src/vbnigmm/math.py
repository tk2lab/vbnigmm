import math

import numpy as np

from numpy import inf
from numpy import fabs
from numpy import exp
from numpy import log
from numpy import power
from numpy import sqrt
from numpy.linalg import inv
from numpy.linalg import slogdet
from scipy.special import xlogy
from scipy.special import k0e
from scipy.special import k1e
from scipy.special import kve
from scipy.special import digamma
from scipy.special import gammaln as lgamma
from scipy.special import multigammaln as multi_lgamma
from scipy.special import logsumexp as log_sum_exp
from scipy.special import softmax


log2 = math.log(2)
log2pi = math.log(2 * math.pi)
sqrt2 = math.sqrt(2)
sqrtpi = math.sqrt(math.pi)


def log_det(x):
    return slogdet(x)[1]


def multi_digamma(x, d):
    return np.sum([digamma(x - i / 2) for i in range(d)], axis=0)


def khratio(v, x):
    m = np.around(2 * v).astype(np.int32)
    negative = m < 0
    absm = -m if negative else m
    if absm % 2 == 0:
        nu = 1.0
        n = absm // 2
        ratio = k1e(x) / k0e(x)
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
