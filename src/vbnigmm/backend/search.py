from . import current as tk


def search_extend(func, t0, t1, args):
    def cond(t0, f0, t1, f1, fac):
        return tk.any(f0 * f1 > 0.0)
    def body(t0, f0, t1, f1, fac):
        cond = f0 * f1 > 0.0
        _t0 = tk.where(cond, t1, t0)
        _f0 = tk.where_func(f0, cond, func, (_t0,) + args)
        _t1 = tk.where(cond, t1 + fac * (t1 - t0), t1)
        _f1 = tk.where_func(f1, cond, func, (_t1,) + args)
        return _t0, _f0, _t1, _f1, fac + 1

    f0 = func(t0, *args)
    f1 = func(t1, *args)
    fac = tk.constant(1.0, t0.dtype)
    t0, _, t1, _, _ = tk.while_loop(cond, body, (t0, f0, t1, f1, fac))
    return t0, t1


def search_shrink(func, deriv, t0, t1, tol):
    def cond(t0, f0, t1, f1):
        dratio = deriv(t1) * (t1 - t0) / (f1 - f0)
        return tk.any(((dratio < 1.0) | (2.0 < dratio)) & (tk.abs(f1) > tol))
    def body(t0, f0, t1, f1):
        tm = t0 + 0.5 * (t1 - t0)
        fm = func(tm)
        cond = f0 * fm > 0.0
        _t0 = tk.where(cond, tm, t0)
        _f0 = tk.where(cond, fm, f0)
        _t1 = tk.where(cond, t1, tm)
        _f1 = tk.where(cond, f1, fm)
        return _t0, _f0, _t1, _f1

    f0 = func(t0)
    f1 = func(t1)
    t0, _, t1, _ = tk.while_loop(cond, body, (t0, f0, t1, f1))
    return t0, t1


def newton(func, deriv, t, tol):
    def cond(newt, oldt):
        return tk.any(tk.abs(newt - oldt) > tol)
    def body(newt, oldt):
        return newt - func(newt) / deriv(newt), newt

    return tk.while_loop(cond, body, (t, t + 1))[0]


def search(func, deriv, t0, t1, args, tol):
    def _func(t):
        return func(t, *args)
    def _deriv(t):
        return deriv(t, *args)

    t0, t1 = search_extend(func, t0, t1, args)
    t0, t1 = search_shrink(_func, _deriv, t0, t1, tol)
    return newton(_func, _deriv, t1, tol)
