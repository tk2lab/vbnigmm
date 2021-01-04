from .utils import LogLikelihood, Size, kmeans, make_one_hot, dummy
from ..backend import current as tk


class Mixture(tk.Model):

    def __init__(self, init_n=20, init_e='kmeans', **args):
        super().__init__()
        self.init_n = init_n
        self.init_e = init_e
        self.prior_config = args

    def fit(self, x, y=None, steps_per_epoch=100, **kwargs):
        x = tk.as_array(x, self.dtype)
        num, dim = x.shape
        size = int((num + self.init_n - 1) / self.init_n)

        self._prior = self.Parameters.make_prior(x, **self.prior_config)
        self.Parameters.add_weights(self, size, dim)
        self.add_loss(self.kl)
        self.compile(loss=LogLikelihood(), metrics=[Size()])

        data = tk.Dataset.from_tensors(x).repeat(steps_per_epoch)
        self._y = self.init_e if y is None else y
        self._max_size = size
        super().fit(data, **kwargs)

    def predict(self, x):
        return tk.argmax(self.predict_proba(x), axis=-1, dtype=tk.int32)

    def predict_proba(self, x):
        return super().predict(x)[1][0]

    @property
    def prior(self):
        return self._prior

    @property
    def posterior(self):
        return self.Parameters.create_from_weights(self)

    def kl(self):
        q = self.posterior
        return tk.sum(q.kl(self.prior, self.get_conditions(q)))


    def assign_posterior(self, q):
        self._size.assign(q.size)
        for dst, src in zip(self._params, q.params):
            if hasattr(dst, 'scatter_update'):
                tk.scatter_update(dst, src)

    def train_step(self, x):
        def _initialize():
            self._initialized.assign(True)
            if not isinstance(self._y, str):
                y = self._y
            else:
                num, size = x.shape[0], self._max_size
                y = tk.random_uniform((num,), 0, size, tk.int32)
                if self._y == 'kmeans':
                    y = kmeans(x, y)
            z = make_one_hot(y, self.dtype)
            return self.init_expect(z)

        def _train_step():
            l, z = self(x)
            self.compiled_loss(dummy, l, regularization_losses=self.losses)
            zsum = tk.sum(z[0], axis=0)
            idx = tk.argsort(zsum)[::-1]
            z = tk.gather(z, idx, axis=2)
            zsum = tk.gather(zsum, idx)
            return tk.mask(z, zsum > 2, axis=2)

        z = tk.cond(self._initialized, _train_step, _initialize)
        q = self.mstep(x, z)
        self.assign_posterior(q)
        self.compiled_metrics.update_state(dummy, z)
        return {m.name: m.result() for m in self.metrics}
