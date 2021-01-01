from collections import namedtuple

from ..backend import current as tk

from .check import check_data
from .check import check_concentration
from .check import check_covariance
from .check import check_bias
from .check import check_normality
from .check import check_scale

from .utils import MixtureParameters
from .dpm import DirichletProcess
from ..distributions.wishart import Wishart
from ..distributions.invgauss import InverseGauss
from ..distributions.gauss import Gauss


class NormalInverseGaussMixtureParameters(MixtureParameters):

    var_names = 'l', 'r', 'f', 'g', 'h', 's', 'u', 'v', 'w', 'm', 'n', 't'
    var_types = 'o', 'o', 's', 's', 's', 's', 's', 's', 's', 'v', 'v', 'm'

    @classmethod
    def make_prior(cls, x, mean=None, cov=None, bias=None,
                    concentration=1.0, concentration_decay=0.0,
                    normality_mean=3.0, normality_dof=1.0,
                    cov_scale=0.3, cov_dof=2.0,
                    mean_scale=1.0, bias_scale=0.3, mean_bias_factor=0.0):
        m0, cov = check_data(x, mean, cov)
        n0 = check_bias(m0, bias)
        l0, r0, py = check_concentration(concentration, concentration_decay)
        s0, t0 = check_covariance(cov_dof, cov_scale, cov)
        f0, g0, h0 = check_normality('gamma', normality_mean, normality_dof)
        u0, v0 = check_scale(cov_scale, mean_scale, bias_scale)
        w0 = mean_bias_factor
        return cls(l0, r0, f0, g0, h0, s0, u0, v0, w0, m0, n0, t0, py, x.dtype)

    def __init__(self, l, r, f, g, h, s, u, v, w, m, n, t, py=0.0, dtype=None):
        Params = namedtuple('Params', self.var_names)
        self.params = Params(l, r, f, g, h, s, u, v, w, m, n, t)
        self.size = tk.size(f)
        self.dim = tk.shape(t)[-1]

        self.alpha = DirichletProcess(l, r, py, dtype)
        self.beta = InverseGauss(f, g, h, dtype)
        self.tau = Wishart(s, t, inv=True, dtype=dtype)
        self.mu = Gauss(m, self.tau * u, dtype)
        self.xi = Gauss(self.mu * w + n, self.tau * v, dtype)
        self.dists = [self.alpha, self.beta, self.tau, self.mu, self.xi]
