import numpy as np
np.set_printoptions(3, suppress=True)

from vbnigmm import GaussMixture as Model
from make_data import make_data


#n, d = 10000, 12
#x = np.random.randn(n, d).astype(np.float32)
x, y = make_data(d=3, seed=1234)

solver = Model(init_e='random')
solver.fit(x, epochs=3)
print(solver.history.history)
z = solver.predict_proba(x)
print(z)
print(np.sum(z, axis=0))
