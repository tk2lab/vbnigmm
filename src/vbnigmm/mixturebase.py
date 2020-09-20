__author__ = 'TAKEKAWA Takashi <takekawa@tk2lab.org>'
__credits__ = 'Copyright 2020, TAKEKAWA Takashi'


import numpy as np
from sklearn.cluster import KMeans

from .mainloop import _fit


class MixtureBase(object):

    def __init__(self, num_try=1, progress=0,
                 init_n=20, init_e='kmeans', init_r=None,
                 tol=1e-5, max_iter=1000, **prior_config):
        if (init_r is None) or isinstance(init_r, int):
            init_r = np.random.RandomState(init_r)
        self.prior_config = prior_config
        self.init_n = init_n
        self.init_e = init_e
        self.init_r = init_r
        self.num_try = num_try
        self.tol = tol
        self.max_iter = max_iter
        self.progress = progress

    def fit(self, x, y=None):
        get_init = dict(
            random=self._init_label_random,
            kmeans=self._init_label_kmeans,
        )[self.init_e]

        def _try():
            label = y or get_init(x)
            posterior = self._get_posterior(x, prior, label)
            return _fit(self, x, prior, posterior)

        num_try = self.num_try if y is None else 1
        prior = self._get_prior(x, **self.prior_config)
        results = [_try() for i in range(num_try)]

        self.prior = prior
        self._results = sorted(results, key=lambda x: x[2], reverse=True)

    def predict(self, x):
        return np.argmax(self.predict_proba(x), axis=-1)

    def predict_proba(self, x):
        return self.stats.expect(x)

    def pdf(self, x):
        return np.exp(self.log_pdf(x))

    def log_pdf(self, x):
        return self.stats.log_pdf(x)

    @property
    def posterior(self):
        return self._results[0][1].params

    @property
    def lower_bound(self):
        return self._results[0][2]

    @property
    def converged(self):
        return self._results[0][0]

    @property
    def n_iter(self):
        return len(self._results[0][3]) - 1

    @property
    def stats(self):
        return self._results[0][1]


    def _init_label_random(self, x):
        n_components = int((x.shape[0] + self.init_n - 1) / self.init_n)
        return self.init_r.randint(n_components, size=x.shape[0])

    def _init_label_kmeans(self, x):
        n_components = int((x.shape[0] + self.init_n - 1) / self.init_n)
        kmeans = KMeans(n_components, random_state=self.init_r)
        return kmeans.fit(x).labels_


    def _get_posterior(self, x, prior, label):
        u, label = np.unique(label, return_inverse=True)
        onehot = np.eye(u.size)[label]
        expect = self._init_expect(onehot)
        return self._mstep(x, prior, expect)

    def _mstep(self, x, prior, expect):
        size = expect[0].sum(axis=0)
        idx = np.argsort(size)[::-1]
        idx = idx[size[idx] >= 2]
        expect = [e[:, idx] for e in expect]
        post = self._update(x, *expect, *prior)
        return post
