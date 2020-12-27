import numpy as np
from sklearn.cluster import KMeans

from ..math import exp


class Mixture(object):

    def __init__(self, init_n=20, init_e='kmeans', init_r=None,
                 tol=1e-5, max_iter=1000, **prior_config):
        if (init_r is None) or isinstance(init_r, int):
            init_r = np.random.RandomState(init_r)
        self.prior_config = prior_config
        self.init_n = init_n
        self.init_e = init_e
        self.init_r = init_r
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, x, y=None):
        self.setup(x, **self.prior_config)
        z = self.start(x, y)
        for n in range(self.max_iter):
            q = self.mstep(x, z)
            z, ll = self.estep(x, q)
            kl = q.kl(self.prior).sum()
            zsum = z[0].sum(axis=0)
            idx = np.argsort(zsum)[::-1]
            z = z[..., idx]
            z = z[..., zsum > 2]
            print(z.shape[-1], ll - kl)
        self.posterior = q

    def predict(self, x, q=None):
        return np.argmax(self.predict_proba(x, q), axis=-1)

    def predict_proba(self, x, q=None):
        return sofmax(self.log_pdf(x[..., None, :], q), axis=-1)


    def start(self, x, y=None):
        if y is None:
            get_init = dict(
                random=self._init_label_random,
                kmeans=self._init_label_kmeans,
            )[self.init_e]
            y = get_init(x)
        u, label = np.unique(y, return_inverse=True)
        return np.eye(u.size)[label][None, :]

    def _init_label_random(self, x):
        n_components = int((x.shape[0] + self.init_n - 1) / self.init_n)
        return self.init_r.randint(n_components, size=x.shape[0])

    def _init_label_kmeans(self, x):
        n_components = int((x.shape[0] + self.init_n - 1) / self.init_n)
        kmeans = KMeans(n_components, random_state=self.init_r)
        return kmeans.fit(x).labels_

    def eval(self, rho):
        max_rho = rho.max(axis=-1)
        r = exp(rho - max_rho[:, None])
        sum_r = r.sum(axis=-1)
        ll = np.log(sum_r).sum() + max_rho.sum()
        z = r / sum_r[:, None]
        return z, ll
