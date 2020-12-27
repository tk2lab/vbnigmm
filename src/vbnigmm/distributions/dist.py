import numpy as np

from ..math import exp


class Dist(object):

    def pdf(self, x):
        return exp(self.log_pdf(x))

    def cross_entropy(self, other):
        return -other.log_pdf(self)

    def entropy(self):
        return self.cross_entropy(self)

    def kl(self, other):
        return self.cross_entropy(other) - self.entropy()
