from ..backend import current as tk
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
