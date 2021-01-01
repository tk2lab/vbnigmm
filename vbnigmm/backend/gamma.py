import tensorflow as _tf


lgamma = _tf.math.lgamma
digamma = _tf.math.digamma


def multi_lgamma(x, d):
    return _tf.math.reduce_sum([lgamma(x - i / 2) for i in range(d)], axis=0)


def multi_digamma(x, d):
    return _tf.math.reduce_sum([digamma(x - i / 2) for i in range(d)], axis=0)
