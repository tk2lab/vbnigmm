import vbnigmm.math.base as tk

from .base import Base
from .matrix import InfMatrix, mul_matrix


class Vector(Base):

    @property
    def dim(self):
        raise NotImplementedError()

    @property
    def mean(self):
        raise NotImplementedError()

    @property
    def mean_log(self):
        raise NotImplementedError()

    @property
    def precision(self):
        raise NotImplementedError()

    def __add__(self, other):
        return affine_vector(1, self, other)

    def __sub__(self, other):
        return affine_vector(1, self, -other)

    def __mul__(self, other):
        return affine_vector(other, self, 0)


def precision(self, other):
    self = wrap_vector(self)
    other = wrap_vector(other)
    if isinstance(self, AffineVector):
        return mul_matrix(1 / fabs(self.a), precision(self.x, other))
    if isinstance(other, AffineVector):
        return mul_matrix(1 / fabs(other.a), precision(self, other.x))
    if self is other:
        return self.precision
    return InfMatrix()


class WrapVector(Vector):

    def __init__(self, x):
        self.x = tk.as_array(x)

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


def wrap_vector(x):
    if isinstance(x, Vector):
        return x
    return WrapVector(x)


class AffineVector(Vector):

    def __init__(self, a, x, b):
        self.a = tk.as_array(a)
        self.b = tk.as_array(b)
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
    x = wrap_vector(x)
    if tk.all(a == 0):
        return WrapVector(b)
    if isinstance(x, WrapVector):
        return WrapVector(a * x.x + b)
    if isinstance(x, AffineVector):
        return AffineVector(a * x.a, x.x, a * x.b + b)
    return AffineVector(a, x, b)
