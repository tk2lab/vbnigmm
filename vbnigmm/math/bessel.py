from ..backend import current as tk
from .search import search, integrate, where_func


def log_kv(v, x):
    def _search_peak(t0, t1, v, x):
        return search(_deriv1, _deriv2, t0, t1, (v, x), tol)
    def _search_bottom(t0, t1, tv, lv):
        return search(_approx_func, _approx_deriv, t0, t1, (tv, lv), tol)

    n = 30
    tol = tk.constant(1e-5, x.dtype)
    eps = tk.constant(2.2e-16 if x.dtype == tk.float64 else 1.2e-7, x.dtype)

    v = tk.as_array(v, x.dtype)
    v = tk.abs(v)
    c = v ** 2 > x
    shape = tk.shape(c)
    v = tk.reshape(tk.broadcast_to(v, shape), [-1])
    x = tk.reshape(tk.broadcast_to(x, shape), [-1])
    c = tk.reshape(c, [-1])

    t0 = tol * tk.ones_like(x)
    t1 = tk.ones_like(x)
    tp = where_func(tk.zeros_like(x), c, _search_peak, (t0, t1, v, x))
    fb = _func(tp, v, x) + tk.log(eps)

    tv = fb / v
    lv = tv - tk.log(2 * v / x)
    cond = (tv > 0) | (_approx_func(0, tv, lv) < 0)
    t0 = tk.where(tv > 0, tv, 0)
    t0 = where_func(t0, cond, _search_bottom, (tp, tv + tol, tv, lv))
    t1 = _search_bottom(tp, tp + 1, tv, lv)
    return tk.reshape(integrate(_func, t0, t1, (v, x), n), shape)


def dv_log_kv(v, x):
    small = 1e-3
    return (log_kv(v + small, x) - log_kv(v, x)) / small


def _func(t, v, x):
    return tk.log_cosh(v * t) - x * tk.cosh(t)


def _deriv1(t, v, x):
    return v * tk.tanh(v * t) - x * tk.sinh(t)


def _deriv2(t, v, x):
    return (v * tk.sech(v * t)) ** 2 - x * tk.cosh(t)


def _approx_func(t, tv, lv):
    t -= tv
    return tk.log(t) - t - lv


def _approx_deriv(t, tv, lv):
    t -= tv
    return 1 / t - 1
