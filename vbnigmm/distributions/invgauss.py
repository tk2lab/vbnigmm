from ..backend import current as tk

from .base import Dist
from .gamma import Gamma
from ..linalg.scalar import Scalar, wrap_scalar


class InverseGauss(Dist, Scalar):

    def __new__(cls, a, b, c=-1 / 2, dtype=None):
        try:
            if tk.native_all(b == 0):
                return Gamma(c, a / 2, dtype)
        except Exception:
            pass
        return super().__new__(cls)

    def __init__(self, a, b, c=-1 / 2, dtype=None):
        self.a = tk.as_array(a, dtype)
        self.b = tk.as_array(b, dtype)
        self.c = c

    @property
    def dtype(self):
        return self.a.dtype
        
    @property
    def mode(self):
        a, b, c = self.a, self.b, self.c
        return ((c - 1) + tk.sqrt(tk.pow(c - 1, 2) + a * b)) / a

    @property
    def mean(self):
        return self._kv_ratio * self._omega
    
    @property
    def mean_inv(self):
        return self._kv_ratio / self._omega - 2 * self.c / self.b

    @property
    def mean_log(self):
        c = tk.as_array(self.c, self.dtype)
        return tk.log(self._omega) + tk.dv_log_kv(c, self._mu)

    @property
    def log_const(self):
        c = tk.as_array(self.c, self.dtype)
        return (
            - tk.log2
            - self.c * tk.log(self._omega)
            - tk.log_kv(c, self._mu)
        )

    def log_pdf(self, x):
        x = wrap_scalar(x, self.dtype)
        return (
            self.log_const
            + (self.c - 1) * x.mean_log
            - (self.a * x.mean + self.b * x.mean_inv) / 2
        )

    @property
    def _omega(self):
        return tk.sqrt(self.b / self.a)

    @property
    def _mu(self):
        return tk.sqrt(self.a * self.b)

    @property
    def _kv_ratio(self):
        return tk.kv_ratio(self.c, self._mu)
