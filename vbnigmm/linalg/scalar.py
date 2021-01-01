from ..backend import current as tk
from .base import Base


class Scalar(Base):
    pass


class WrapScalar(Scalar):

    def __init__(self, x, dtype=None):
        self.x = tk.as_array(x, dtype)

    @property
    def mean(self):
        return self.x

    @property
    def mean_inv(self):
        return 1 / self.x

    @property
    def mean_log(self):
        return tk.log(self.x)


def wrap_scalar(x, dtype):
    if isinstance(x, Scalar):
        return x
    return WrapScalar(x, dtype)
