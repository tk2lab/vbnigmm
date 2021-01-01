from ..backend import current as tk

from .base import Base
from .matrix import InfMatrix, mul_matrix


class Vector(Base):

    def __add__(self, other):
        return affine_vector(1., self, other)

    def __sub__(self, other):
        return affine_vector(1., self, -other)

    def __mul__(self, other):
        return affine_vector(other, self, 0.)


def precision(self, other, dtype=None):
    self = wrap_vector(self, dtype)
    other = wrap_vector(other, dtype)
    if isinstance(self, AffineVector):
        return mul_matrix(1 / tk.abs(self.a), precision(self.x, other))
    if isinstance(other, AffineVector):
        return mul_matrix(1 / tk.abs(other.a), precision(self, other.x))
    if self is other:
        return self.precision
    return InfMatrix()


class WrapVector(Vector):

    def __init__(self, x, dtype=None):
        self.x = tk.as_array(x, dtype)

    @property
    def dim(self):
        return self.x.shape[-1]

    @property
    def mean(self):
        return self.x

    @property
    def mean_log(self):
        return tk.log(self.x)

    @property
    def precision(self):
        return InfMatrix()


def wrap_vector(x, dtype=None):
    if isinstance(x, Vector):
        return x
    return WrapVector(x, dtype)


class AffineVector(Vector):

    def __init__(self, a, x, b):
        self.a = tk.as_array(a, x.dtype)
        self.b = tk.as_array(b, x.dtype)
        self.x = x

    @property
    def dim(self):
        return self.x.dim

    @property
    def mean(self):
        return self.x.mean * self.a[..., None]

    @property
    def precision(self):
        return mul_matrix(1 / self.a**2, self.x.precision)


def affine_vector(a, x, b):
    try:
        if tk.native_all(a == 0):
            return WrapVector(b, x.dtype)
    except Exception:
        pass
    if isinstance(x, WrapVector):
        return WrapVector(a * x.x + b, x.dtype)
    if isinstance(x, AffineVector):
        return AffineVector(a * x.a, x.x, a * x.b + b)
    return AffineVector(a, x, b)
