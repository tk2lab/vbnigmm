"""cache decorator."""

__author__ = 'TAKEKAWA Takashi <takekawa@tk2lab.org>'
__credits__ = 'Copyright 2020, TAKEKAWA Takashi'


def cache_property(func):
    name = f'_cache{func.__name__}'
    @property
    def _wraper(self, *args, **kwargs):
        if not hasattr(self, name):
            setattr(self, name, func(self, *args, **kwargs))
        return getattr(self, name)
    return _wraper