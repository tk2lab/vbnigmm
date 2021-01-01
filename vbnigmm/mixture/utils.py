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


def make_one_hot(y):
    u, label = tk.unique(y)
    return tk.gather(tk.eye(tk.size(u), dtype=tk.float32), label)


def kmeans(x, z, n=100):
    one_hot = tk.eye(tk.shape(z)[1])
    for i in range(n):
        mean = tk.transpose(z) @ x / tk.sum(z, axis=0)[:, None]
        dist = tk.sum((x[:, None, :] - mean) ** 2, axis=2)
        z = tf.gather(one_hot, tk.argmax(dist, axis=1))
    return z
