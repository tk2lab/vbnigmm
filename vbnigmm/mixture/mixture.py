import tensorflow as tf
from sklearn.cluster import KMeans

import vbnigmm.math.base as tk
from .utils import LogLikelihood, Size


class Mixture(tf.keras.Model):

    def __init__(self, init_n=20, init_e='kmeans', **args):
        super().__init__()
        self.init_n = init_n
        self.init_e = init_e
        self.prior_config = args

    def fit(self, x, y=None, seed=None, steps_per_epoch=100, **kwargs):
        if seed is not None:
            tf.random.set_seed(seed)

        self.prior = self.Parameters.create_prior(x, **self.prior_config)
        z = self.init_expect(x, y)

        x = tk.as_array(x, self.dtype)
        q = self.mstep(x, z)
        self._params = q.build_weights(self)
        self.add_loss(self.kl)

        self.compile(loss=LogLikelihood(), metrics=[Size()])
        data = tf.data.Dataset.from_tensors(x).repeat(steps_per_epoch)
        super().fit(data, **kwargs)

    def predict(self, x):
        return tk.argmax(self.predict_proba(x), axis=-1)

    def predict_proba(self, x):
        return tk.softmax(super().predict(x)[0], axis=-1)


    @property
    def posterior(self):
        s = self._params[0]
        return self.Parameters(*[
            dict(c=lambda: p, r=lambda: p[:s - 1], f=lambda: p[:s])[t[0]]()
            for p, t in zip(self._params[1:], self.Parameters.var_types)
        ])

    def kl(self):
        return tk.sum(self.posterior.kl(self.prior))


    def init_label(self, x, y=None):
        if y is None:
            num, dim = x.shape
            size = int((num + self.init_n - 1) / self.init_n)
            y = dict(
                random=lambda: tf.random.uniform((num,), 0, size, tf.int32),
                kmeans=lambda: KMeans(size).fit(x).predict(x),
            )[self.init_e]()
        u, label = tk.unique(y)
        return tk.gather(tk.eye(tk.size(u), dtype=tk.float32), label)

    def train_step(self, x):
        dummy = tk.zeros((1,))
        y = self(x)
        l, z = self.calc_expect(y)
        self.compiled_loss(dummy, l, regularization_losses=self.losses)

        z = self.sort_and_remove(z)
        q = self.mstep(x, z)
        q.assign_weights(self._params)

        self.compiled_metrics.update_state(dummy, z[0])
        return {m.name: m.result() for m in self.metrics}

    def sort_and_remove(self, z):
        zsum = tk.sum(z[0], axis=0)
        idx = tk.argsort(zsum)[::-1]
        z = tk.gather(z, idx, axis=2)
        zsum = tk.gather(zsum, idx)
        return tk.mask(z, zsum > 2, axis=2)
