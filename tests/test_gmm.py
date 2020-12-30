import numpy as np

from vbnigmm import GaussMixture as Model

n, d = 10000, 12
x = np.random.randn(n, d).astype(np.float32)

solver = Model()
solver.fit(x)
solver.predict_proba(x)
