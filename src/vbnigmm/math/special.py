from .wrap import sum as _sum
from .wrap import lgamma ,digamma


def multi_lgamma(x, d):
    return _sum([lgamma(x - i / 2) for i in range(d)], axis=0)


def multi_digamma(x, d):
    return _sum([digamma(x - i / 2) for i in range(d)], axis=0)
