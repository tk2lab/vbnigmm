__author__ = 'TAKEKAWA Takashi <takekawa@tk2lab.org>'
__credits__ = 'Copyright 2020, TAKEKAWA Takashi'


from tqdm import tqdm


class NullLogger(object):

    def __enter__(self):
        return self

    def __exit__(self, *info):
        pass

    def update(self, msg):
        pass


class PrintLogger(NullLogger):

    def update(self, msg):
        print(msg)


class TqdmLogger(tqdm):

    def update(self, msg):
        self.set_description(msg)
        super().update()


def get_logger(progress, max_iter):
    if progress == 1:
        logger = TqdmLogger(total=max_iter)
    elif progress == 2:
        logger = PrintLogger()
    else:
        logger = NullLogger()
    return logger
