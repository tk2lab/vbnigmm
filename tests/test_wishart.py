import numpy as np

from vbnigmm.linbase.matrix import wrap_matrix
from vbnigmm.linbase.matrix import mul_matrix
from vbnigmm.distributions.wishart import Wishart

dim = 5
x = np.random.randn(100, dim)
x = x.T @ x
y = np.random.randn(100, dim)
y = y.T @ y

w0 = Wishart(dim + 2, np.eye(dim))
w1 = Wishart(dim + 10, x)
m0 = wrap_matrix(y)

print(w0.mean)
print(w0.mean_inv)
print(w0.mean_log_det)

print(w1.mean)
print(w1.mean_inv)
print(w1.mean_log_det)

print(w0.log_pdf(m0))
print(w0.log_pdf(w0))
print(w0.log_pdf(w1))

print(w1.log_pdf(m0))
print(w1.log_pdf(w0))
print(w1.log_pdf(w1))

print(w0.log_pdf(mul_matrix(2, w0)))
print(w0.log_pdf(mul_matrix(3, w1)))
print(w0.log_pdf(mul_matrix(4, m0)))

print(w1.kl(w0))
print(w1.kl(w1))
