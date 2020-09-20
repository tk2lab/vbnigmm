__author__ = 'TAKEKAWA Takashi <takekawa@tk2lab.org>'
__credits__ = 'Copyright 2020, TAKEKAWA Takashi'


import numpy as np

from .logger import get_logger


def _fit(self, x, prior, post):

    def make_dist():
        dist = self.posterior_type(*post)
        size = post[0].size
        lb = (dist.log_pdf(x).sum() - dist.kl(*prior).sum()) / x.shape[0]
        return dist, size, lb

    def update_status(status):
        msg = f'opt:{opt_lb: 10.5f} ({opt_size:3d})'
        msg += f', score:{lb: 10.5f} ({size:3d})'
        if size != history[-1]['size']:
            msg += f', diff=--------'
            msg += ' >'
            status = 0
        else:
            diff = lb - history[-1]['lb'] 
            msg += f', diff:{diff: .5f}'
            if -self.tol <= diff <= self.tol:
                status = status + 1 if status >= 0 else +1
                msg += ' o'
            elif diff < -self.tol:
                status = status - 1 if status <= 0 else -1
                msg += ' x'
            else:
                msg += ' -'
        msg += ' *' if lb > opt_lb else '  '
        logger.update(msg)
        return status
    
    dist, size, lb = make_dist()
    status = 0
    history = [dict(size=size, lb=lb)]
    opt_post, opt_dist, opt_size, opt_lb = post, dist, size, lb
    with get_logger(self.progress, self.max_iter) as logger:
        while (len(history) <= self.max_iter) and (status < 5):
            post = self._mstep(x, prior, dist.expect_all(x))
            dist, size, lb = make_dist()
            status = update_status(status)
            history.append(dict(size=size, lb=lb))
            if lb > opt_lb:
                opt_post, opt_dist, opt_size, opt_lb = post, dist, size, lb
    return status >= 5, opt_dist, opt_lb, history
