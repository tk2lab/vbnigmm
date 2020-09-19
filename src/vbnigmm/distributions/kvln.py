"""Logarithm of Modified Bessel functions of second kind."""

__author__ = 'TAKEKAWA Takashi <takekawa@tk2lab.org>'
__credits__ = 'Copyright 2020, TAKEKAWA Takashi'


import numpy as np
from numpy import pi, sign, floor, fabs, exp, log, sinh, cosh, tanh
from scipy.special import logsumexp, xlogy, kn, kve
from .sech import sech
from .csch import csch
from .coth import coth
from .coshln import coshln
from .sinhln import sinhln
from .linesearch import line_search
from .search import search
from .newton import newton


def _kvln(v, x):
    shape, v, x = _kv_prev(fabs(v), x)
    out = np.empty_like(x)
    n = np.floor(v - 0.5).astype(np.int32)
    ch = (n == v - 0.5) & (v < 6)
    if np.any(~ch):
        out[~ch] = log(kve(v[~ch], x[~ch])) - x[~ch]
    cx = ~np.isfinite(out)
    if np.any(cx):
        out[cx] = _kvln(v[cx], x[cx])
    return out.reshape(shape)


def kvln(v, x, d=16, n=30, tolp=1e-3):
    def func(t): return _f1(v, x, t)
    def deriv(t): return _f2(v, x, t)
    shape, v, x = _kv_prev(fabs(v), x)
    tp = np.zeros_like(v)
    c = v ** 2 > x
    if np.any(c):
        tp[c] = search(func, deriv, 1e-10, 1.0, tol, True)
    fp = _f0(tp, v, x) - d * log(10)
    t0, t1 = _find_range(v, x, fp)
    return _integrate(lambda t: _f0(t, v, x), t0, t1, n).reshape(shape)


def _khln(n, x):
    shape, n, x = _kv_prev(n, x)
    n = n.astype(np.int32)
    log2x = log(2 * x)
    out = xlogy(1 / 2, pi) - log2x / 2 - x
    ns, idx = np.unique(n, return_inverse=True)
    n = ns.max()
    lr = [0] + [log(n - r + 1) + log (n + r) - log(r) for r in range(1, n + 1)]
    lr = np.cumsum(lr)
    rs = np.arange(n + 1)
    for i, n in enumerate(ns):
        c = idx==i
        out[c] += logsumexp(lr[:n+1] - rs[:n+1] * log2x[c, None], axis=-1)
    return out.reshape(shape)


def kvdvln(v, x, d=16, n=30):
    shape, v, x = _kv_prev(v, x)
    tp = _find_peak(_g1, _g2, v, x)
    fp = _g0(tp, v, x) - log(2) - d * log(10)
    t0, t1 = _find_range(v, x, fp)
    return _integrate(lambda t: _g0(t, v, x), t0, t1, n).reshape(shape)


def kvratioln(v, x, k, d=16, n=30):
    shape, v, x = _kv_prev(v, x)
    vk = v + k
    tp = np.zeros_like(v)
    c = v ** 2 > x
    if np.any(c):
        tp[c] = _find_peak(_f1, _f2, vk[c], x[c])
    fp = _f0(tp, vk, x) - d * log(10)
    t0, t1 = _find_range(vk, x, fp)
    vvk = np.array([v, vk])
    func = lambda t: _f0(t[:, None, :], vvk, x[None, :])
    tmp = _integrate(func, t0, t1, n)
    return (tmp[0] - tmp[1]).reshape(shape)


def _kv_prev(v, x):
    v = fabs(v)
    v, x = np.broadcast_arrays(v, x)
    return v.shape, v.ravel(), x.ravel()


def _find_range(v, x, fp):
    tv = fp / v
    lv = tv - log(2 * v / x)
    t0 = np.zeros_like(lv)
    c = tv > 0.0
    if np.any(c):
        t0[c] = _find_edge(lv[c], 0.5) + tv[c]
    t1 = _find_edge(lv, 2.0) + tv
    return t0, t1


def _find_peak(func, deriv, tol):
    def func(t): return _f1(v, x, t)
    def deriv(t): return _f2(v, x, t)
    return search(func, deriv, 1e-10, 1.0, tol, True)


def _find_edge(tv, lv, t0, t1, tol, expand):
    def func(t): return log(t - tv) - (t - tv) - lv
    def deriv(t): return 1 / (t - tv) - 1
    return search(func, deriv, t0, t1, tol, expand)


def _integrate(func, t0, t1, n):
    h = (t1 - t0) / n
    t0 = np.clip(t0, h / 2, None)
    t = t0 + h * np.arange(n)[:, None]
    return logsumexp(func(t), axis=0) + log(h)


def _f0(t, v, x):
    return coshln(v * t) - x * cosh(t)


def _f1(t, v, x):
    return v * tanh(v * t) - x * sinh(t)


def _f2(t, v, x):
    return (v * sech(v * t)) ** 2 - x * cosh(t)


def _g0(t, v, x):
    return log(t) + sinhln(v * t) - x * cosh(t)


def _g1(t, v, x):
    return 1 / t + v * coth(v * t) - x * sinh(t)


def _g2(t, v, x):
    return -1 / t ** 2 - (v * csch(v * t)) ** 2 - x * cosh(t)
