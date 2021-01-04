from .. import backend as tk

from .utils import MixtureParameters
from .dpm import DirichletProcess, gen_dpm_params
from ..distributions.wishart import Wishart
from ..distributions.gauss import Gauss
from ..distributions.invgauss import InverseGauss, gen_gig_params


class NormalInverseGaussMixtureParameters(MixtureParameters):

    var_names = 'l', 'r', 'f', 'g', 'h', 's', 'u', 'v', 'w', 'm', 'n', 't'
    var_types = 'o', 'o', 's', 's', 's', 's', 's', 's', 's', 'v', 'v', 'm'

    @classmethod
    def make_prior(cls, x, mean=None, bias=None, cov=None,
                   concentration_args=(1.0, 0.0),
                   normality_args=('gamma', 10.0, 20.0),
                   mean_scale=1.0, cov_scale=0.3, bias_scale=0.3,
                   mean_bias_factor=0.0, cov_ddof=0.0):
        num, dim = x.shape
        m0 = mean or tk.mean(x, axis=0)
        n0 = bias or tk.zeros_like(m0)
        cov = cov or (tk.transpose(x - m0) @ (x - m0)) / (num - 1)
        l0, r0, py = gen_dpm_params(*concentration_args)
        f0, g0, h0 = gen_gig_params(*normality_args)
        s0 = dim + cov_ddof
        t0 = cov * (cov_scale ** 2) * s0
        u0 = (cov_scale ** 2) / (mean_scale ** 2)
        v0 = 1 / (bias_scale ** 2)
        w0 = mean_bias_factor
        return cls(l0, r0, f0, g0, h0, s0, u0, v0, w0, m0, n0, t0, py, x.dtype)

    def __init__(self, l, r, f, g, h, s, u, v, w, m, n, t, py=0.0, dtype=None):
        Params = tk.namedtuple('Params', self.var_names)
        self.params = Params(l, r, f, g, h, s, u, v, w, m, n, t)
        self.size = tk.size(f)
        self.dim = tk.shape(t)[-1]

        self.alpha = DirichletProcess(l, r, py, dtype)
        self.beta = InverseGauss(f, g, h / 2, dtype)
        self.tau = Wishart(s, t, inv=True, dtype=dtype)
        self.mu = Gauss(
            m, self.tau * u, dtype,
            condition=dict(tau=self.tau),
        )
        self.xi = Gauss(
            self.mu * w + n, self.tau * v, dtype,
            condition=dict(tau=self.tau, mu=self.mu),
        )
        self.dists = [self.alpha, self.beta, self.tau, self.mu, self.xi]
