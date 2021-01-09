from ..backend import current as _tk


sinh = _tk.sinh
cosh = _tk.cosh
tanh = _tk.tanh


def log_sinh(x):
    return _tk.log(_tk.sinh(x))


def log_cosh(x):
    return x + _tk.log1p(_tk.expm1(-2 * x) / 2)


def sech(x):
    e = _tk.exp(-x)
    return 2 * e / (1 + e ** 2)


def csch(x):
    return -2 * _tk.exp(-x) / _tk.expm1(-2 * x)


def coth(x):
    return 1 / _tk.tanh(x)
