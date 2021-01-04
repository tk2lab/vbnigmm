from ..backend import current as _tk


def multi_lgamma(x, d):
    return _tk.sum([_tk.lgamma(x - i / 2) for i in range(d)], axis=0)


def multi_digamma(x, d):
    return _tk.sum([_tk.digamma(x - i / 2) for i in range(d)], axis=0)
