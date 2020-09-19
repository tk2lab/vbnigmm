"""(Generalized) Inverse Gaussian distributions."""

__author__ = 'TAKEKAWA Takashi <takekawa@tk2lab.org>'
__credits__ = 'Copyright 2018, TAKEKAWA Takashi'


import numpy as np
import scipy.special as sp

from .basedist import BaseDist
from .gamma import Gamma
from .math import khratio
from .cache import cache_property


def InverseGaussian(a, b, c=-1 / 2, halfint=False):
    if np.all(b == 0):
        return InverseGaussianB0(a, c)
    if halfint:
        return InverseGaussianHalfInt(a, b, c)
    else:
        return InverseGaussianDist(a, b, c)


class InverseGaussianDist(BaseDist):

    def __init__(self, a, b, c=-1 / 2):
        a, b, c = map(np.asarray, (a, b, c))
        self.params = a, b, c
        
    @cache_property
    def mode(self):
        a, b, c = self.params
        return ((c - 1) + np.sqrt(np.power(c - 1, 2) + a * b)) / a

    @cache_property
    def mean(self):
        return self._kratio * self._omega
    
    @cache_property
    def inv_mean(self):
        a, b, c = self.params
        return self._kratio / self._omega - 2 * c / b

    @cache_property
    def log_mean(self):
        a, b, c = self.params
        return self._klndv + np.log(self._omega)

    @cache_property
    def log_const(self):
        a, b, c = self.params
        return sp.xlogy(-c, self._omega) + self._mu - np.log(2 * self._kve)

    def log_pdf(self, x):
        a, b, c = self.params
        return self.log_const + sp.xlogy(c - 1, x) - (a * x + b / x) / 2

    def cross_entropy(self, *args):
        a, b, c = map(np.asarray, args)
        mu = np.sqrt(a * b)
        omega = np.sqrt(b / a)
        return - (
            + sp.xlogy(-c, omega) + mu - np.log(2 * sp.kve(c, mu))
            + (c - 1) * self.log_mean
            - (a * self.mean + b * self.inv_mean) / 2
        )

    @cache_property
    def _omega(self):
        a, b, c = self.params
        return np.sqrt(b / a)

    @cache_property
    def _mu(self):
        a, b, c = self.params
        return np.sqrt(a * b)

    @cache_property
    def _kve(self):
        a, b, c = self.params
        return sp.kve(c, self._mu)

    @cache_property
    def _kratio(self):
        a, b, c = self.params
        return sp.kve(c + 1, self._mu) / self._kve

    @cache_property
    def _klndv(self):
        small = 1e-10
        a, b, c = self.params
        kvdv = (sp.kve(c + small, self._mu) - self._kve) / small
        return kvdv / self._kve


class InverseGaussianHalfInt(InverseGaussianDist):

    @cache_property
    def _kratio(self):
        a, b, c = self.params
        return khratio(c, self._mu)


class InverseGaussianB0(Gamma):
    
    def __init__(self, a, c):
        super().__init__(c, a / 2)
        
    def cross_entropy(self, *args):
        a, b, c = map(np.asarray, args)
        if np.all(b == 0):
            return super().cross_entropy(c, a / 2)
        raise NotImplementedError()
