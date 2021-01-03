import tensorflow as tf
import numpy as np
np.set_printoptions(3, suppress=True)

from vbnigmm import NormalInverseGaussMixture as Model
from make_data import make_data

tf.keras.backend.set_floatx('float64')

#n, d = 10000, 12
#x = np.random.randn(n, d).astype(np.float32)
x, y = make_data(d=4, normality=3, difficulty=0.3, seed=1234, population=[100]*10)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        'loss', patience=3, restore_best_weights=True, verbose=1,
    )
]
solver = Model(init_e='kmeans', normality_type='gamma')
solver.fit(x, steps_per_epoch=100, epochs=100, callbacks=callbacks)
print(solver.history.history)
q = solver.posterior
print('alpha', q.alpha.mean)
print('beta', q.beta.mean)
print('mu', q.mu.mean)
print('xi', q.xi.mean)
print('tau', tf.exp(q.tau.mean_log_det))
z = solver.predict_proba(x)
print(np.sum(z, axis=0))
