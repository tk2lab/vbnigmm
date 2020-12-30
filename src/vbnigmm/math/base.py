import math

from .wrap import (
    int32, float32,
    inf,
    as_array,
    zeros, ones, range, zeros_like, ones_like, eye,
    size, shape,
    stack, concat,
    argsort, mask, gather, tensordot,
    fabs, round,
    exp, log, pow, sqrt, xlogy,
    lgamma, digamma,
    k0e, k1e,
    all, sum, max, log_sum_exp, softmax,
    cumsum, cumprod,
    transpose, inv, log_det
)
from .special import multi_lgamma, multi_digamma


log2 = math.log(2)
log2pi = math.log(2 * math.pi)
sqrt2 = math.sqrt(2)
sqrtpi = math.sqrt(math.pi)


class Base(object):
    pass
