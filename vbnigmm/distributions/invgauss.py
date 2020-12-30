import vbnigmm.math.base as tk

from .dist import Dist
from .gamma import Gamma
from ..math.scalar import Scalar, wrap_scalar


def InverseGauss(a, b, c=-1 / 2):
    if tk.all(b == 0):
        return Gamma(c, a / 2)
    if tk.all(tk.fabs(c - tk.floor(c) - 0.5) < 1e-6):
        return _InverseGaussHalfInt(a, b, c)
    else:
        return _InverseGauss(a, b, c)


class _InverseGauss(Dist, Scalar):

    def __init__(self, a, b, c):
        self.a = tk.as_array(a)
        self.b = tk.as_array(b)
        self.c = tk.as_array(c)
        
    @property
    def mode(self):
        a, b, c = self.a, self.b, self.c
        return ((c - 1) + tk.sqrt(tk.pow(c - 1, 2) + a * b)) / a

    @property
    def mean(self):
        return self._kratio * self._omega
    
    @property
    def mean_inv(self):
        return self._kratio / self._omega - 2 * self.c / self.b

    @property
    def mean_log(self):
        return self._klndv + tk.log(self._omega)

    @property
    def log_const(self):
        return (
            self._mu - log2 - self.c * tk.log(self._omega) - tk.log(self._kve)
        )

    def log_pdf(self, x):
        x = wrap_scalar(x)
        return (
            + self.log_const
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
    def _kve(self):
        return tk.kve(self.c, self._mu)

    @property
    def _kratio(self):
        return tk.kve(self.c + 1, self._mu) / self._kve

    @property
    def _klndv(self):
        small = 1e-7
        kvdv = (tk.kve(self.c + small, self._mu) - self._kve) / small
        return kvdv / self._kve


class _InverseGaussHalfInt(_InverseGauss):

    @property
    def _kratio(self):
        return tk.khratio(self.c, self._mu)
