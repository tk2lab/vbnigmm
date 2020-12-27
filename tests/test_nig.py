import numpy as np

from vbnigmm.driver.nig import NormalInverseGaussMixture

n, d = 100, 5
x = np.random.randn(n, d)

solver = NormalInverseGaussMixture()
solver.fit(x)
solver.predict_proba(x)
