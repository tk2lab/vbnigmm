"""Function of statistical distributions."""

__author__ = 'TAKEKAWA Takashi <takekawa@tk2lab.org>'
__credits__ = 'Copyright 2018, TAKEKAWA Takashi'


import numpy as np


class BaseDist(object):

    def pdf(self, x):
        return np.exp(self.log_pdf(x))

    def cross_entropy(self, other):
        return -other.log_pdf(self)

    def entropy(self):
        return self.cross_entropy(self)

    def kl(self, other):
        return self.cross_entropy(other) - self.entropy()


def mean(x):
    return x.mean if hasattr(x, 'mean') else x


def mean_log(x):
    return x.mean_log if hasattr(x, 'mean_log') else np.log(x)


def mean_logdet(x):
    return x.mean_logdet if hasattr(x, 'mean_logdet') else np.linalg.logdet(x)
