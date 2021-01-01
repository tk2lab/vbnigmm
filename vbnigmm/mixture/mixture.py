from ..backend import current as tk
from .utils import LogLikelihood, Size, kmeans, make_one_hot, dummy


class Mixture(tk.Model):

    def __init__(self, init_n=30, init_e='kmeans', **args):
        super().__init__()
        self.init_n = init_n
        self.init_e = init_e
        self.prior_config = args

    def fit(self, x, y=None, seed=None, steps_per_epochs=100, **kwargs):
        num, dim = x.shape
        size = int((num + self.init_n - 1) / self.init_n)
        x = tk.as_array(x, self.dtype)
        self._y = self.init_e if y is None else y
        if seed is not None:
            tk.random.set_seed(seed)

        self.prior = self.Parameters.make_prior(x, **self.prior_config)
        self.build_posterior(size, dim)
        self.compile(loss=LogLikelihood(), metrics=[Size()])

        data = tk.Dataset.from_tensors(x).repeat(steps_per_epochs)
        super().fit(data, **kwargs)

    @property
    def posterior(self):
        q = self.Parameters
        constants = self.get_constants()
        s = self._params[0]
        params = []
        for n, t, p in zip(q.var_names, q.var_types, self._params[1:]):
            c = constants.get(n, None)
            if c is not None:
                params.append(c)
            elif t == 'o':
                params.append(p[:s - 1])
            else:
                params.append(p[:s])
        return q(*params)

    def kl(self):
        return tk.sum(self.posterior.kl(self.prior))

    def predict(self, x):
        return tk.argmax(self.predict_proba(x), axis=-1, dtype=tk.int32)

    def predict_proba(self, x):
        return super().predict(x)[1][0]


    def build_posterior(self, size, dim):
        s, d, q = size, dim, self.Parameters
        shapes = dict(o=(s - 1,), s=(s,), v=(s, d,), m=(s, d, d))
        constants = self.get_constants()
        self._initialized = self.add_weight('init', (), tk.bool, 'Zeros')
        self._max_size = size
        self._params = []
        self._params.append(self.add_weight('size', (), tk.int32))
        for n, t in zip(q.var_names, q.var_types):
            c = constants.get(n, None)
            if c is not None:
                self._params.append(c)
            else:
                self._params.append(self.add_weight(n, shapes[t], self.dtype))
        self.add_loss(self.kl)

    def assign_posterior(self, q):
        self._params[0].assign(q.size)
        for dst, src in zip(self._params[1:], q.params):
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
            return _sort_and_remove(z)

        def _sort_and_remove(z):
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
