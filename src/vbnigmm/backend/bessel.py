from . import current as _tk

from .search import search
from .tri import sinh, cosh, tanh, sech, csch, coth, log_sinh, log_cosh


def log_kv(v, x, dtype=None, n=128, tol=1e-5):

    def func(t, v, x, th=0):
        return log_cosh(v * t) - x * cosh(t) - th

    def deriv1(t, v, x, th=0):
        return v * tanh(v * t) - x * sinh(t)

    def deriv2(t, v, x):
        return (v * sech(v * t)) ** 2 - x * cosh(t)

    return _log_kv(func, deriv1, deriv2, False, v, x, dtype, n, tol)


def log_dv_kv(v, x, dtype=None, n=64, tol=1e-5):

    def func(t, v, x, th=0):
        return _tk.log(t) + log_sinh(v * t) - x * cosh(t) - th

    def deriv1(t, v, x, th=0):
        return 1 / t + v * coth(v * t) - x * sinh(t)

    def deriv2(t, v, x):
        return - (1 / t) ** 2 + (v * csch(v * t)) ** 2 - x * cosh(t)
    
    return _log_kv(func, deriv1, deriv2, True, v, x, dtype, n, tol)


def _log_kv(func, deriv1, deriv2, sign, v, x, dtype, n, tol):

    def search_peak(t0, t1, v, x):
        return search(deriv1, deriv2, t0, t1, (v, x), tol)

    def search_bottom(t0, t1, v, x, fb):
        return search(func, deriv1, t0, t1, (v, x, fb), tol)

    x = _tk.as_array(x, dtype)
    dtype = x.dtype
    v = _tk.as_array(v, dtype)
    orig_shape = _tk.shape(v * x)
    v = _tk.reshape(_tk.broadcast_to(v, orig_shape), [-1])
    x = _tk.reshape(_tk.broadcast_to(x, orig_shape), [-1])
    shape = _tk.shape(v)

    eps = _tk.constant(2.2e-16 if dtype == _tk.float64 else 1.2e-7, dtype)
    zero = _tk.zeros(shape, dtype)
    one = _tk.ones(shape, dtype)

    sign = _tk.sign(v) if sign else 1
    v = _tk.abs(v)

    cond = deriv1(zero, v, x) > 0
    tp = _tk.where_func(zero, cond, search_peak, (zero + eps, one, v, x))
    fb = func(tp, v, x) + _tk.log(eps)

    cond = func(zero, v, x, fb) < 0
    t0 = _tk.where_func(zero, cond, search_bottom, (zero, tp, v, x, fb))
    t1 = search_bottom(tp, tp + 1, v, x, fb)

    h = (t1 - t0) / n
    t = t0 + h / 2 + h * _tk.range(n, dtype=dtype)[:, None]
    out = _tk.log_sum_exp(func(t, v, x), axis=0) + _tk.log(h)
    return _tk.reshape(sign * out, orig_shape)
