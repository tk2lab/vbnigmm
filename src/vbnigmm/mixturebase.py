__author__ = 'TAKEKAWA Takashi <takekawa@tk2lab.org>'
__credits__ = 'Copyright 2020, TAKEKAWA Takashi'


import numpy as np
from sklearn.cluster import KMeans

from .logger import get_logger


class MixtureBase(object):

    def __init__(self, num_try=1, try_remove=False, progress=0,
                 init_n=20, init_e='kmeans', init_m='once', init_r=None,
                 tol=1e-5, max_iter=1000, **prior_config):
        if (init_r is None) or isinstance(init_r, int):
            init_r = np.random.RandomState(init_r)
        self.prior_config = prior_config
        self.init_n = init_n
        self.init_e = init_e
        self.init_m = init_m
        self.init_r = init_r
        self.num_try = num_try
        self.try_remove = try_remove
        self.tol = tol
        self.max_iter = max_iter
        self.progress = progress

    def fit(self, x, y=None):
        prior = self._get_prior(x, **self.prior_config)
        results = []
        if y is not None:
            results.append(self._fit(x, y, prior))
        else:
            for i in range(self.num_try):
                y = dict(random=self._init_label_random,
                         kmeans=self._init_label_kmeans)[self.init_e](x)
                results.append(self._fit(x, y, prior))
        results.sort(key=lambda x: x[3], reverse=True)
        self._prior = prior
        self._results = results

    @property
    def prior(self):
        return self._prior

    @property
    def converged(self):
        return self._results[0][0]

    @property
    def n_iter(self):
        return self._results[0][1]

    @property
    def posterior(self):
        return self._results[0][2].params

    @property
    def stats(self):
        return self._results[0][2]

    @property
    def lower_bound(self):
        return self._results[0][3]

    def predict(self, x):
        return np.argmax(self.predict_proba(x), axis=-1)

    def predict_proba(self, x):
        return self.stats.expect(x)

    def pdf(self, x):
        return np.exp(self.log_pdf(x))

    def log_pdf(self, x):
        return self.stats.log_pdf(x)

    @staticmethod
    def _check_data(x, mean, cov):
        if x.ndim != 2:
            raise ValueError('x must be 2d')
        if mean is None:
            mean = x.mean(axis=0)
        elif mean.ndim != 1:
            raise ValueError('mean must be 1d')
        if cov is None:
            cov = np.atleast_2d(np.cov(x.T))
        elif cov.ndim != 2:
            raise ValueError('cov must be 2d')
        return x, mean, cov

    @staticmethod
    def _check_concentration(prior_type, l0, r0):
        if (l0 == None) and (r0 == None):
            if prior_type == 'dd':
                l0 = 1.0
            elif prior_type == 'dpm':
                l0 = r0 = 1.0
        return l0, r0

    @staticmethod
    def _check_covariance(cov_reliability, cov_scale, cov):
        s0 = (cov.shape[0] - 1) + cov_reliability
        t0 = cov * cov_scale ** 2 * s0
        return s0, t0

    def _fit(self, x, y, prior):

        def make_dist():
            dist = self.posterior_type(*post)
            size = post[0].size
            lb = (dist.log_pdf(x).sum() - dist.kl(*prior).sum()) / x.shape[0]
            history.append((size, lb))
            return dist, size, lb

        def update_opt():
            nonlocal opt_post, opt_dist, opt_size, opt_lb, removed_id
            opt_post = post
            opt_dist = dist
            opt_size = size
            opt_lb = lb
            removed_id = -1

        def make_removed_post():
            nonlocal removed_id
            removed_id += 1
            idx = np.arange(opt_size)
            removed_idx = np.delete(idx, -(removed_id + 1))
            return [p[removed_idx] for p in opt_post]

        def update_status(status):
            msg = f'opt:{opt_lb: 10.5f} ({opt_size:3d})'
            msg += f', score:{lb: 10.5f} ({size:3d})'
            if size != old_size:
                msg += f', diff=--------'
                msg += ' !' if removed else ' >'
                status = 0
            else:
                diff = lb - old_lb
                msg += f', diff:{diff: .5f}'
                if -self.tol <= diff <= self.tol:
                    status = status + 1 if status >= 0 else +1
                    msg += ' o'
                elif diff < -self.tol:
                    status = status - 1 if status <= 0 else -1
                    msg += ' x'
                else:
                    msg += ' -'
            opt = lb > opt_lb
            msg += ' *' if opt else '  '
            return status, msg, opt
        
        history = []

        opt_post, opt_dist, opt_size, opt_lb, removed_id = [None] * 5
        post = dict(once=self._get_params_once,
                    ind=self._get_params_ind)[self.init_m](x, y, prior)
        dist, size, lb = make_dist()
        status = 0
        update_opt()

        converged = False
        with get_logger(self.progress, self.max_iter) as logger:
            for n_iter in range(self.max_iter):
                success = status >= 5
                try_remove = (self.try_remove and (1 < opt_size)
                              and (removed_id + 1 < opt_size))
                if success and not try_remove:
                    converged = True
                    break
                if success and try_remove:
                    old_size = opt_size
                    old_lb = opt_lb
                    post = make_removed_post()
                    removed = True
                else:
                    old_size = size
                    old_lb = lb
                    post = self._mstep(x, prior, dist.expect_all(x))
                    removed = False
                dist, size, lb = make_dist()
                status, msg, opt = update_status(status)
                if opt:
                    update_opt()
                logger.update(msg)
        return converged, n_iter, opt_dist, opt_lb, history

    def _init_label_random(self, x):
        n_components = int((x.shape[0] + self.init_n - 1) / self.init_n)
        return self.init_r.randint(n_components, size=x.shape[0])

    def _init_label_kmeans(self, x):
        n_components = int((x.shape[0] + self.init_n - 1) / self.init_n)
        kmeans = KMeans(n_components, random_state=self.init_r)
        return kmeans.fit(x).labels_

    def _get_params_once(self, x, y, prior):
        u, y = np.unique(y, return_inverse=True)
        yz = np.eye(u.size)[y]
        expect = self._init_expect(yz)
        return self._mstep(x, prior, expect)

    def _get_params_ind(self, x, y, prior):
        tmp = []
        for i in np.unique(y):
            xi = x[y == i]
            yz = np.ones((xi.shape[0], 1))
            expect = self._init_expect(yz)
            tmp.append(self._mstep(xi, prior, expect))
        post = []
        for p in zip(*tmp):
            post.append(np.concatenate(p, axis=0))
        return post

    def _mstep(self, x, prior, expect):
        size = expect[0].sum(axis=0)
        idx = np.argsort(size)[::-1]
        idx = idx[size[idx] >= 2]
        expect = [e[:, idx] for e in expect]
        post = self._update(x, *expect, *prior)
        return post
