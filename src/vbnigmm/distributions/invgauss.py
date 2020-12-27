import numpy as np

from .dist import Dist
from .gamma import Gamma
from ..linbase.scalar import Scalar, wrap_scalar
from ..math import log2, log, sqrt, power, kve, khratio


def InverseGauss(a, b, c=-1 / 2, halfint=False):
    if np.all(b == 0):
        return Gamma(c, a / 2)
    if halfint:
        return _InverseGaussHalfInt(a, b, c)
    else:
        return _InverseGauss(a, b, c)


class _InverseGauss(Dist, Scalar):

    def __init__(self, a, b, c=-1 / 2):
        self.a = np.asarray(a)
        self.b = np.asarray(b)
        self.c = np.asarray(c)
        
    @property
    def mode(self):
        a, b, c = self.a, self.b, self.c
        return ((c - 1) + sqrt(power(c - 1, 2) + a * b)) / a

    @property
    def mean(self):
        return self._kratio * self._omega
    
    @property
    def mean_inv(self):
        return self._kratio / self._omega - 2 * self.c / self.b

    @property
    def mean_log(self):
        return self._klndv + log(self._omega)

    @property
    def log_const(self):
        return self._mu - log2 - self.c * log(self._omega) - log(self._kve)

    def log_pdf(self, x):
        x = wrap_scalar(x)
        return (
            + self.log_const
            + (self.c - 1) * x.mean_log
            - (self.a * x.mean + self.b * x.mean_inv) / 2
        )

    @property
    def _omega(self):
        return sqrt(self.b / self.a)

    @property
    def _mu(self):
        return sqrt(self.a * self.b)

    @property
    def _kve(self):
        return kve(self.c, self._mu)

    @property
    def _kratio(self):
        return kve(self.c + 1, self._mu) / self._kve

    @property
    def _klndv(self):
        small = 1e-10
        kvdv = (kve(self.c + small, self._mu) - self._kve) / small
        return kvdv / self._kve


class _InverseGaussHalfInt(_InverseGauss):

    @property
    def _kratio(self):
        return khratio(self.c, self._mu)
