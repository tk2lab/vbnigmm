from .. import backend as tk

from ..distributions.base import Dist


dummy = tk.zeros((1,))


class LogLikelihood(tk.Loss):

    def call(self, y_true, y_pred):
        return -tk.log_sum_exp(y_pred, axis=-1)


class Size(tk.Metric):

    def __init__(self):
        super().__init__()
        self._result = self.add_weight('result', (), dtype=tk.int32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        self._result.assign(tk.shape(y_pred)[-1])

    def result(self):
        return self._result


class MixtureParameters(Dist):

    @classmethod
    def create_from_weights(cls, self):
        size, params, dtype = self._size, self._params, self.dtype
        constants = self.get_constants()
        _params = []
        for n, t, p in zip(cls.var_names, cls.var_types, params):
            c = constants.get(n, None)
            if c is not None:
                _params.append(c)
            elif t == 'o':
                _params.append(p[:size - 1])
            else:
                _params.append(p[:size])
        return cls(*_params, dtype=dtype)

    @classmethod
    def add_weights(cls, target, size, dim):
        s, d = size, dim
        shapes = dict(o=(s - 1,), s=(s,), v=(s, d,), m=(s, d, d))
        dtype = target.dtype
        constants = target.get_constants()
        target._initialized = target.add_weight('init', (), tk.bool, 'Zeros')
        target._size = target.add_weight('size', (), tk.int32)
        target._params = []
        for n, t in zip(cls.var_names, cls.var_types):
            c = constants.get(n, None)
            if c is not None:
                target._params.append(c)
            else:
                target._params.append(target.add_weight(n, shapes[t], dtype))

    def log_pdf(self, x, condition=None):
        out = 0
        for s, d in zip(self.dists, x.dists):
            s = s.update(condition)
            x = x.update(condition)
            out += s.log_pdf(d)
        return out


def make_one_hot(y, dtype):
    u, label = tk.unique(y)
    return tk.gather(tk.eye(tk.size(u), dtype=dtype), label)


def kmeans(x, y):
    def cond(o, y):
        return ~tk.all(tk.equal(o, y))
    def update(o, y):
        z = make_one_hot(y, x.dtype)
        mean = (tk.transpose(z) @ x) / tk.sum(z, axis=0)[:, None]
        dist = tk.sum((x[:, None, :] - mean) ** 2, axis=2)
        return y, tk.argmin(dist, axis=1, output_type=y.dtype)
    _, y = tk.while_loop(cond, update, (tk.zeros_like(y), y))
    return y
