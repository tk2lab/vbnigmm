import tensorflow as tf
import numpy as np
np.set_printoptions(3, suppress=True)

from vbnigmm import NormalInverseGaussMixture as Model
from make_data import make_data

tf.keras.backend.set_floatx('float64')

#n, d = 10000, 12
#x = np.random.randn(n, d).astype(np.float32)
x, y = make_data(d=4, normality=10, difficulty=0.1, seed=1234, population=[100]*10)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        'loss', patience=3, restore_best_weights=True, verbose=1,
    )
]
solver = Model(init_e='kmeans')
solver.fit(x, steps_per_epoch=100, epochs=100, callbacks=callbacks)
q = solver.posterior
print(q.alpha.mean)
print(q.beta.mean)
print(q.mu.mean)
print(q.xi.mean)
#print(q.tau.mean_inv)
z = solver.predict_proba(x)
print(solver.history.history)
print(np.sum(z, axis=0))
