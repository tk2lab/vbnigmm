from .. import backend as tk

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
        self.c = tk.as_array(c, dtype)

    @property
    def dtype(self):
        return self.a.dtype
        
    @property
    def mode(self):
        a, b, c = self.a, self.b, self.c
        return ((c - 1) + tk.sqrt(tk.pow(c - 1, 2) + a * b)) / a

    @property
    def mean(self):
        return tk.exp(
            (1 / 2) * (tk.log(self.b) - tk.log(self.a))
            + tk.log_kv(self.c + 1, self._mu)
            - tk.log_kv(self.c, self._mu)
        )
    
    @property
    def mean_inv(self):
        return tk.exp(
            (1 / 2) * (tk.log(self.a) - tk.log(self.b))
            + tk.log_kv(self.c - 1, self._mu)
            - tk.log_kv(self.c, self._mu)
        )

    @property
    def mean_log(self):
        return (
            (1 / 2) * (tk.log(self.b) - tk.log(self.a))
            + tk.dv_log_kv(self.c, self._mu)
        )

    @property
    def log_const(self):
        return (
            (self.c / 2) * (tk.log(self.a) - tk.log(self.b))
            - tk.log2
            - tk.log_kv(self.c, self._mu)
        )

    def log_pdf(self, x, condition=None):
        x = wrap_scalar(x, self.dtype)
        return (
            self.log_const
            + (self.c - 1) * x.mean_log
            - (1 / 2) * (self.a * x.mean + self.b * x.mean_inv)
        )

    @property
    def _mu(self):
        return tk.sqrt(self.a * self.b)
