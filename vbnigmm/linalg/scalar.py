from ..backend import current as tk
from .base import Base


class Scalar(Base):

    @property
    def dim(self):
        raise NotImplementedError()

    @property
    def mean(self):
        raise NotImplementedError()

    @property
    def mean_inv(self):
        raise NotImplementedError()

    @property
    def mean_log(self):
        raise NotImplementedError()


class WrapScalar(Scalar):

    def __init__(self, x):
        self.x = tk.as_array(x)

    @property
    def dim(self):
        return self.x.shape[-1]

    @property
    def mean(self):
        return self.x

    @property
    def mean_inv(self):
        return 1 / self.x

    @property
    def mean_log(self):
        return tk.log(self.x)


def wrap_scalar(x):
    if isinstance(x, Scalar):
        return x
    return WrapScalar(x)
