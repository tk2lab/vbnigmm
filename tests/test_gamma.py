import numpy as np

from vbnigmm.distributions.gamma import Gamma

g0 = Gamma(5, 5)
g1 = Gamma(10, 7)
s0 = 9

print(g0.mean)
print(g0.mean_inv)
print(g0.mean_log)

print(g1.mean)
print(g1.mean_inv)
print(g1.mean_log)

print(g0.log_pdf(g0))
print(g0.log_pdf(g1))
print(g0.log_pdf(s0))

print(g1.log_pdf(g0))
print(g1.log_pdf(g1))
print(g1.log_pdf(s0))

print(g1.kl(g0))
print(g1.kl(g1))
