from vbnigmm import NormalInverseGaussMixture as Model
from make_data import make_data


#n, d = 10000, 12
#x = np.random.randn(n, d).astype(np.float32)
x, y = make_data(d=3)

solver = Model()
solver.fit(x, epochs=3)
print(solver.history.history)
print(solver.predict_proba(x))
