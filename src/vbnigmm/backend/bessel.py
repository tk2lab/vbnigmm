from . import current as _tk

from .search import search, integrate
from .tri import sinh, cosh, tanh, sech, csch, coth, log_sinh, log_cosh


def log_kv(v, x, dtype=None):

    def func(t, v, x, th=0):
        return log_cosh(v * t) - x * cosh(t) - th

    def deriv1(t, v, x, th=0):
        return v * tanh(v * t) - x * sinh(t)

    def deriv2(t, v, x):
        return (v * sech(v * t)) ** 2 - x * cosh(t)

    return _log_kv(func, deriv1, deriv2, False, v, x, dtype)


def log_dv_kv(v, x, dtype=None):

    def func(t, v, x, th=0):
        return _tk.log(t) + log_sinh(v * t) - x * cosh(t) - th

    def deriv1(t, v, x, th=0):
        return 1 / t + v * coth(v * t) - x * sinh(t)

    def deriv2(t, v, x):
        return - (1 / t) ** 2 + (v * csch(v * t)) ** 2 - x * cosh(t)
    
    return _log_kv(func, deriv1, deriv2, True, v, x, dtype)


def _log_kv(func, deriv1, deriv2, sign, v, x, dtype):

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

    n = 30
    tol = _tk.constant(1e-5, dtype)
    eps = _tk.constant(2.2e-16 if dtype == _tk.float64 else 1.2e-7, dtype)

    sign = _tk.sign(v) if sign else 1
    v = _tk.abs(v)

    t0 = _tk.zeros(shape, dtype)
    t1 = _tk.ones(shape, dtype)
    cond = deriv1(t0, v, x) > 0
    tp = _tk.where_func(t0, cond, search_peak, (t0 + eps, t1, v, x))
    fb = func(tp, v, x) + _tk.log(eps)

    t0 = _tk.zeros(shape, dtype)
    cond = func(t0, v, x, fb) < 0
    t0 = _tk.where_func(t0, cond, search_bottom, (tp, t0, v, x, fb))
    t1 = search_bottom(tp, tp + 1, v, x, fb)
    return _tk.reshape(sign * integrate(func, t0, t1, (v, x), n), orig_shape)
