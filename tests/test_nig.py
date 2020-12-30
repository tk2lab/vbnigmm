import numpy as np
import tensorflow as tf

from vbnigmm import NormalInverseGaussMixture as Model
from vbnigmm.math.base import kv_ratio_h


print(tf.function(kv_ratio_h)(11/2, np.ones(10, np.float32)))

n, d = 10000, 12
x = np.random.randn(n, d).astype(np.float32)

solver = Model()
solver.fit(x)
solver.predict_proba(x)
