from .. import backend as tk


class Dist(object):

    def pdf(self, x, condition=None):
        return tk.exp(self.log_pdf(x, condition))

    def cross_entropy(self, other, condition=None):
        return -other.log_pdf(self, condition)

    def entropy(self, condition=None):
        return self.cross_entropy(self, condition)

    def kl(self, other, condition=None):
        return self.cross_entropy(other, condition) - self.entropy(condition)

    def update(self, condition=None):
        return self
