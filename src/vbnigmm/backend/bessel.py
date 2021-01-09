from . import current as tk

from .search import search, integrate
from .tri import sinh, cosh, tanh, sech, csch, coth, log_sinh, log_cosh


def log_kv(v, x, dtype=None):
    def _func(t, v, x, th=0):
        return log_cosh(v * t) - x * cosh(t) - th

    def _deriv1(t, v, x, th=0):
        return v * tanh(v * t) - x * sinh(t)

    def _deriv2(t, v, x):
        return (v * sech(v * t)) ** 2 - x * cosh(t)

    def _search_peak(t0, t1, v, x):
        return search(_deriv1, _deriv2, t0, t1, (v, x), tol)

    def _search_bottom(t0, t1, v, x, fb):
        return search(_func, _deriv1, t0, t1, (v, x, fb), tol)

    x = tk.as_array(x, dtype)
    dtype = x.dtype
    v = tk.as_array(v, dtype)
    orig_shape = tk.shape(v * x)
    v = tk.reshape(tk.broadcast_to(v, orig_shape), [-1])
    x = tk.reshape(tk.broadcast_to(x, orig_shape), [-1])
    shape = tk.shape(v)

    n = 30
    tol = tk.constant(1e-5, dtype)
    eps = tk.constant(2.2e-16 if dtype == tk.float64 else 1.2e-7, dtype)

    v = tk.abs(v)

    t0 = tk.zeros(shape, dtype)
    t1 = tk.ones(shape, dtype)
    tp = tk.where_func(t0, v ** 2 > x, _search_peak, (t0 + eps, t1, v, x))
    fb = _func(tp, v, x) + tk.log(eps)

    t0 = tk.zeros(shape, dtype)
    cond = _func(t0, v, x, fb) < 0
    t0 = tk.where_func(t0, cond, _search_bottom, (tp, t0, v, x, fb))
    t1 = _search_bottom(tp, tp + 1, v, x, fb)
    return tk.reshape(integrate(_func, t0, t1, (v, x), n), orig_shape)


def log_dv_kv(v, x, dtype=None):
    def _func(t, v, x, th=0):
        return tk.log(t) + log_sinh(v * t) - x * cosh(t) - th

    def _deriv1(t, v, x, th=0):
        return 1 / t + v * coth(v * t) - x * sinh(t)

    def _deriv2(t, v, x):
        return - (1 / t) ** 2 + (v * csch(v * t)) ** 2 - x * cosh(t)

    def _search_peak(t0, t1, v, x):
        return search(_deriv1, _deriv2, t0, t1, (v, x), tol)

    def _search_bottom(t0, t1, v, x, fb):
        return search(_func, _deriv1, t0, t1, (v, x, fb), tol)

    x = tk.as_array(x, dtype)
    dtype = x.dtype
    v = tk.as_array(v, dtype)
    orig_shape = tk.shape(v * x)
    v = tk.reshape(tk.broadcast_to(v, orig_shape), [-1])
    x = tk.reshape(tk.broadcast_to(x, orig_shape), [-1])
    shape = tk.shape(v)

    n = 30
    tol = tk.constant(1e-5, dtype)
    eps = tk.constant(2.2e-16 if dtype == tk.float64 else 1.2e-7, dtype)

    sign = tk.sign(v)
    v = tk.abs(v)

    t0 = tk.zeros(shape, dtype)
    t1 = tk.ones(shape, dtype)
    tp = _search_peak(t0, t1, v, x)
    fb = _func(tp, v, x) + tk.log(eps)
    tk.print('tp', tp, fb)

    t0 = tk.zeros(shape, dtype)
    cond = _func(t0, v, x, fb) < 0
    t0 = tk.where_func(t0, cond, _search_bottom, (tp, t0 + eps, v, x, fb))
    tk.print('t0', t0, _func(t0, v, x))
    t1 = _search_bottom(tp, tp + 1, v, x, fb)
    tk.print('t1', t1, _func(t1, v, x))
    return tk.reshape(sign * integrate(_func, t0, t1, (v, x), n), orig_shape)
