from tqdm import tqdm

from ..math import inf


class History(tqdm):

    def __init__(self, max_iter, tol):
        super().__init__(range(max_iter))
        self.tol = tol
        self.opt_size = None
        self.opt_lb = -inf
        self.size = []
        self.lb = []

    def append(self, size, lb):
        self.size.append(size)
        self.lb.append(lb)
        if len(self.size) < 2:
            self.status = 0
        elif self.size[-1] != self.size[-2]:
            self.status = 0
            #self.write(f'size changed: {self.size[-2]} -> {self.size[-1]}')
        else:
            diff = self.lb[-1] - self.lb[-2]
            if -self.tol <= diff <= self.tol:
                self.status = (0 if self.status < 0 else self.status) + 1
            elif diff < -self.tol:
                self.status = (0 if self.status > 0 else self.status) - 1
        if lb > self.opt_lb:
            self.opt_lb = lb
            self.opt_size = size
        self.set_postfix_str(f'{self.opt_lb: 10.5f}({self.opt_size:3d})')
        if self.status >= 5:
            self.total = self.n
            return True
        return False
