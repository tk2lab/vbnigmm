import tensorflow as tf

import vbnigmm.math.base as tk
from ..distributions.dist import Dist


class LogLikelihood(tf.keras.losses.Loss):

    def call(self, y_true, y_pred):
        return -tk.log_sum_exp(y_pred, axis=-1)


class Size(tf.keras.metrics.Metric):

    def __init__(self):
        super().__init__()
        self._result = self.add_weight('result', (), dtype=tk.int32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        self._result.assign(tk.shape(y_pred)[-1])

    def result(self):
        return self._result


class MixtureParameters(Dist):

    def log_pdf(self, x):
        return sum([s.log_pdf(d) for s, d in zip(self.dists, x.dists)], 0)

    def build_weights(self, target):
        def create(n, t, v):
            if t[0] == 'c':
                return v
            shape = dict(r=(size - 1,), f=(size,))[t[0]]
            shape = shape + dict(s=(), v=(dim,), m=(dim, dim))[t[1]]
            return target.add_weight(
                n, shape, initializer=tf.keras.initializers.Constant(v),
            )
        size, dim = self.size, self.dim
        _size = target.add_weight(
                'size', (), tk.int32, tf.keras.initializers.Constant(size),
            )
        return [_size] + [
            create(n, t, v)
            for n, t, v in zip(self.var_names, self.var_types, self.params)
        ]

    def assign_weights(self, params):
        params[0].assign(self.size)
        for dst, src, t in zip(params[1:], self.params, self.var_types):
            if t[0] != 'c':
                update = tf.IndexedSlices(src, tk.range(tk.shape(src)[0]))
                dst.scatter_update(update)
