class Base(object):

    @property
    def dtype(self):
        return self.x.dtype

    def get(self, replace=None):
        replace = replace or dict()
        return replace.get(id(self), self)
