from ..backend import current as tk

from .base import Base


class Matrix(Base):

    def trace_dot_outer(self, a, b=None):
        if b is None:
            b = a
        ab = a[..., :, None]  * b[..., None, :]
        return tk.sum(self.mean * ab, axis=(-2, -1))

    def trace_dot(self, other):
        other = wrap_matrix(other)
        if isinstance(self, MultiplyMatrix):
            return self.x.trace_dot(other) * self.a
        if isinstance(other, MultiplyMatrix):
            return self.trace_dot(other.x) * other.a
        if self is other:
            raise NotImplementedError()
        return tk.sum(self.mean * other.mean, axis=(-2, -1))

    def trace_dot_inv(self, other):
        other = wrap_matrix(other)
        if isinstance(other, InfMatrix):
            return 0
        if isinstance(self, MultiplyMatrix):
            return self.x.trace_dot_inv(other) * self.a
        if isinstance(other, MultiplyMatrix):
            return self.trace_dot_inv(other.x) / other.a
        if self is other:
            return self.dim
        return tk.sum(self.mean * other.mean_inv, axis=(-2, -1))

    def __mul__(self, other):
        return mul_matrix(other, self)


class InfMatrix(Matrix):
    pass


class WrapMatrix(Matrix):

    def __init__(self, x, dtype=None):
        self.x = tk.as_array(x, dtype)

    @property
    def dim(self):
        return self.x.shape[-1]

    @property
    def mean(self):
        return self.x

    @property
    def mean_inv(self):
        return tk.inv(self.x)

    @property
    def mean_log_det(self):
        return tk.log_det(self.x)


def wrap_matrix(x, dtype=None):
    if isinstance(x, Matrix):
        return x
    return WrapMatrix(x, dtype)


class MultiplyMatrix(Matrix):

    def __init__(self, a, x):
        self.a = tk.as_array(a, x.dtype)
        self.x = x

    @property
    def dim(self):
        return self.x.dim

    @property
    def mean(self):
        return self.x.mean * self.a[..., None, None]

    @property
    def mean_inv(self):
        return self.x.mean_inv / self.a[..., None, None]

    @property
    def mean_log_det(self):
        return self.x.dim * tk.log(self.a) + self.x.mean_log_det


def mul_matrix(a, x):
    try:
        if tk.natieve_all(a == 0):
            return WrapMatrix(tk.zeros_like(x.mean), x.dtype)
    except Exception:
        pass
    if isinstance(x, InfMatrix):
        return x
    if isinstance(x, WrapMatrix):
        return WrapMatrix(a * x.x, x.dtype)
    if isinstance(x, MultiplyMatrix):
        return MultiplyMatrix(a * x.a, x.x)
    return MultiplyMatrix(a, x)
