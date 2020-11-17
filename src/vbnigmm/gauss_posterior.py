__author__ = 'TAKEKAWA Takashi <takekawa@tk2lab.org>'
__credits__ = 'Copyright 2020, TAKEKAWA Takashi'

import math

import numpy as np
import scipy.special as sp

from .distributions import BaseDist
from .distributions import Dirichlet
from .distributions import Wishart
from .distributions import MultivariateNormal


class GaussianMixture(BaseDist):

    def __init__(self, alpha, tau, mu):
        self.alpha = alpha
        self.tau = tau
        self.mu = mu

    def log_pdf(self, x):
        dim = x.shape[-1]
        return (
            + self.alpha.mean_log
            - (dim / 2) * math.log(2 * math.pi)
            + (1 / 2) * self.tau.mean_log_det
            - (1 / 2) * self.mu.mahalanobis(x)
        )


class BayesianGaussianMixturePosterior(BaseDist):

    def __init__(self, l, r, s, t, u, m):
        self.alpha = Dirichlet(l, r)
        self.tau = Wishart(s, t, inv=True)
        self.mu = WishartGauss(m, u, self.tau) 

    def log_pdf(self, alpha, tau, mu):
        return self.alpha.log_pdf(alpha) + self.tau.log_pdf(tau) + self.mu.log_pdf(mu)


class WishartGauss(BaseDist):

    def 
