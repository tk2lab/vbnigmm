import numpy as np
import vbnigmm.math.base as tk
from sklearn.cluster import KMeans

from .history import History


class Mixture(object):

    def __init__(self, init_n=20, init_e='kmeans', init_r=None,
                 tol=1e-5, max_iter=1000, **prior_config):
        if (init_r is None) or isinstance(init_r, int):
            init_r = np.random.RandomState(init_r)
        self.init_n = init_n
        self.init_e = init_e
        self.init_r = init_r
        self.tol = tol
        self.max_iter = max_iter
        self.prior_config = prior_config

    def fit(self, x, y=None):
        num = x.shape[0]
        self.setup(x, **self.prior_config)
        z = self.start(x, y)
        history = History(self.max_iter, self.tol)
        for n in history:
            # update
            size = z.shape[-1]
            q = self.mstep(x, z)
            z, ll = self.estep(x, q)
            kl = tk.sum(q.kl(self.prior))
            lb = (ll - kl) / num
            # test converge
            if history.append(size, lb):
                break
            # sort and remove
            zsum = tk.sum(z[0], axis=0)
            idx = tk.argsort(zsum)[::-1]
            z = tk.gather(z, idx, axis=-1)
            z = tk.mask(z, zsum > 2, axis=2)
        self.history = dict(lb=history.lb, size=history.size)
        self.posterior = q

    def predict(self, x, q=None):
        return tk.argmax(self.predict_proba(x, q), axis=-1)

    def predict_proba(self, x, q=None):
        return tk.softmax(self.log_pdf(x[..., None, :], q), axis=-1)


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
        n_components = int((x.shape[0] + self.init_n - 1) / self.init_n)
        return self.init_r.randint(n_components, size=x.shape[0])

    def _init_label_kmeans(self, x):
        n_components = int((x.shape[0] + self.init_n - 1) / self.init_n)
        kmeans = KMeans(n_components, random_state=self.init_r)
        return kmeans.fit(x).labels_

    def eval(self, rho):
        max_rho = tk.max(rho, axis=-1)
        r = tk.exp(rho - max_rho[:, None])
        sum_r = tk.sum(r, axis=-1)
        ll = tk.sum(tk.log(sum_r)) + tk.sum(max_rho)
        z = r / sum_r[:, None]
        return z, ll
