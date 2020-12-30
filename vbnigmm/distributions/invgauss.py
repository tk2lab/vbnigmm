import vbnigmm.math.base as tk

from .dist import Dist
from .gamma import Gamma
from ..math.scalar import Scalar, wrap_scalar


class InverseGauss(Dist, Scalar):

    def __init__(self, a, b, c=-1 / 2):
        self.a = tk.as_array(a)
        self.b = tk.as_array(b)
        self.c = tk.as_array(c)
        self.gamma = Gamma(c, a / 2)
        
    @property
    def mode(self):
        a, b, c = self.a, self.b, self.c
        v0 = self.gamma.mode
        v1 = ((c - 1) + tk.sqrt(tk.pow(c - 1, 2) + a * b)) / a
        return tk.where(b == 0, v0, v1)

    @property
    def mean(self):
        v0 = self.gamma.mean
        v1 = self._kv_ratio * self._omega
        return tk.where(self.b == 0, v0, v1)
    
    @property
    def mean_inv(self):
        v0 = self.gamma.mean_inv
        v1 = self._kv_ratio / self._omega - 2 * self.c / self.b
        return tk.where(self.b == 0, v0, v1)

    @property
    def mean_log(self):
        v0 = self.gamma.mean_log
        v1 = tk.log(self._omega) + tk.dv_log_kv(self.c, self._mu)
        return tk.where(self.b == 0, v0, v1)

    @property
    def log_const(self):
        v0 = self.gamma.log_const
        v1 = (
            - tk.log2
            - self.c * tk.log(self._omega)
            - tk.log_kv(self.c, self._mu)
        )
        return tk.where(self.b == 0, v0, v1)

    def log_pdf(self, x):
        v0 = self.gamma.log_pdf(x)
        x = wrap_scalar(x)
        v1 = (
            + self.log_const
            + (self.c - 1) * x.mean_log
            - (self.a * x.mean + self.b * x.mean_inv) / 2
        )
        return tk.where(self.b == 0, v0, v1)

    @property
    def _omega(self):
        return tk.sqrt(self.b / self.a)

    @property
    def _mu(self):
        return tk.sqrt(self.a * self.b)

    @property
    def _kv_ratio(self):
        return tk.kv_ratio_h(self.c, self._mu)
