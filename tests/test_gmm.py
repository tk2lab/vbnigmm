import tensorflow as tf
import numpy as np
np.set_printoptions(3, suppress=True)

from vbnigmm import GaussMixture as Model
from make_data import make_data


#n, d = 10000, 12
#x = np.random.randn(n, d).astype(np.float32)
x, y = make_data(d=3, seed=1234)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        'loss', patience=3, restore_best_weights=True, verbose=1,
    )
]
solver = Model(init_e='kmeans')
solver.fit(x, epochs=100, callbacks=callbacks)
print(solver.history.history)
z = solver.predict_proba(x)
print(z)
print(np.sum(z, axis=0))
