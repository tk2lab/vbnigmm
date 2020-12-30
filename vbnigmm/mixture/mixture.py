import numpy as np
import tensorflow as tf
import vbnigmm.math.base as tk
from sklearn.cluster import KMeans

from .history import History


class Mixture(tf.keras.Model):

    def __init__(self, init_n=20, init_e='kmeans', init_r=None,
                 tol=1e-5, max_iter=1000, **prior_config):
        super().__init__()
        if (init_r is None) or isinstance(init_r, int):
            init_r = np.random.RandomState(init_r)
        self.init_n = init_n
        self.init_e = init_e
        self.init_r = init_r
        self.tol = tol
        self.max_iter = max_iter
        self.prior_config = prior_config

    def fit(self, x, y=None, epochs=10, steps_per_epoch=100):
        self.prior = self.build_prior(x, **self.prior_config)
        num = tf.shape(x)[0]
        dim = tf.shape(x)[1]
        self.max_size = int((num + self.init_n - 1) / self.init_n)
        z = self.start(x, y)
        q = self.mstep(x, z)

        self._size = self.add_weight('size', (), tf.int32)
        self._params = self.build_posterior(self.max_size, dim)
        self.assign_posterior(q)

        self.compile()
        x = tf.data.Dataset.from_tensors(x).repeat(steps_per_epoch)
        super().fit(x, epochs=epochs)

    @property
    def posterior(self):
        params = [
            p[:self.var_shape[t](self._size, None)[0]]
            for t, p in zip(self.var_types, self._params)
        ]
        return self.Parameters(*params)

    def predict(self, x, q=None):
        return tk.argmax(self.predict_proba(x, q), axis=-1)

    def predict_proba(self, x, q=None):
        return tk.softmax(self.log_pdf(x[..., None, :], q), axis=-1)


    var_shape = dict(
        a=lambda size, dim: (size - 1,),
        s=lambda size, dim: (size,),
        v=lambda size, dim: (size, dim),
        m=lambda size, dim: (size, dim, dim),
    )

    def build_posterior(self, size, dim):
        return [
            self.add_weight(n, self.var_shape[t](size, dim))
            for n, t in zip(self.var_names, self.var_types)
        ]

    def start(self, x, y=None):
        if y is None:
            get_init = dict(
                random=self._init_label_random,
                kmeans=self._init_label_kmeans,
            )[self.init_e]
            y = get_init(x)
        u, label = np.unique(y, return_inverse=True)
        return tk.gather(tk.eye(u.size), label)[None, :]

    def _init_label_random(self, x):
        return self.init_r.randint(self.max_size, size=x.shape[0])

    def _init_label_kmeans(self, x):
        kmeans = KMeans(self.max_size, random_state=self.init_r)
        return kmeans.fit(x).labels_

    def call(self, x):
        return tk.softmax(self.log_pdf(x[..., None, :]), axis=-1)[None, ...]

    def train_step(self, x):
        z = self(x)
        q = self.mstep(x, z)
        #q = self.q.sort_and_remove()
        self.assign_posterior(q)
        return dict(size=q.size)

    def assign_posterior(self, q):
        self._size.assign(q.size)
        for d, s in zip(self._params, q.params):
            d.scatter_update(tf.IndexedSlices(s, tf.range(tk.shape(s)[0])))


    def ll(self, x, q):
        return tk.log_sum_exp(self.log_pdf(x[:, None, :], q), axis=-1)

    def kl(self, q):
        return q.kl(self.prior)

    def _metrix(self, x):
        num = tf.shape(x)[0]
        q = self.posterior
        ll = self.ll(x, q)
        kl = self.kl(q)
        lb = (ll - kl) / num
        self.add_metrix(q.size)
        self.add_metrix(ll, 'll')
        self.add_metrix(kl, 'kl')
        self.add_metrix(lb, 'lb')
