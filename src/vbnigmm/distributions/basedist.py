"""Function of statistical distributions."""

__author__ = 'TAKEKAWA Takashi <takekawa@tk2lab.org>'
__credits__ = 'Copyright 2018, TAKEKAWA Takashi'


import numpy as np


class BaseDist(object):

    def pdf(self, x):
        return np.exp(self.log_pdf(x))

    def entropy(self):
        return self.cross_entropy(*self.params)

    def kl(self, *params):
        return self.cross_entropy(*params) - self.entropy()
