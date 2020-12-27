import numpy as np

from vbnigmm.dpm import DirichletProcess


dp0 = DirichletProcess(1, 1, 0.1)
x = np.arange(1, 10)
y = np.cumsum(x[:0:-1])
x = x[:-1]
dp1 = DirichletProcess(x, y)
p = np.random.rand(9)
p /= p.sum()
print(dp1.log_pdf(p))
#print(dp1.log_pdf(dp0))
#print(dp1.log_pdf(dp1))

print(dp1.kl(dp0))
print(dp1.kl(dp1))
