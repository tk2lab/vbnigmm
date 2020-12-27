import numpy as np

from vbnigmm.linbase.matrix import mul_matrix
from vbnigmm.linbase.vector import affine_vector
from vbnigmm.distributions.wishart import Wishart
from vbnigmm.distributions.gauss import Gauss

dim = 5
x = np.random.randn(100, dim)
t0 = x.T @ x
t1 = Wishart(dim + 2, t0)
t2 = mul_matrix(7, t1)
t3 = mul_matrix(5, t1)

g = []

m0 = np.random.randn(dim)
g.append(Gauss(m0, t0))
g.append(Gauss(m0, t1))
g.append(Gauss(m0, t2))

m1 = affine_vector(2, g[0], m0)
m2 = affine_vector(2, g[1], m0)
m3 = affine_vector(2, g[2], m0)
g.append(Gauss(m1, t3))
g.append(Gauss(m2, t3))
g.append(Gauss(m3, t3))

for p in g:
    print(p.mean)
    print(p.precision.mean)
    print(p.kl(g[0]))
