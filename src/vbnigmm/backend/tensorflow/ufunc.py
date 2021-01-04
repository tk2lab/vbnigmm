import tensorflow as _tf
import tensorflow_probability as _tfp


abs = _tf.math.abs
exp = _tf.math.exp
log = _tf.math.log
pow = _tf.math.pow
fabs = _tf.math.abs
sqrt = _tf.math.sqrt
round = _tf.math.round

sinh = _tf.math.sinh
cosh = _tf.math.cosh
tanh = _tf.math.tanh

lgamma = _tf.math.lgamma
digamma = _tf.math.digamma

k0_e = _tf.math.special.bessel_k0e
k1_e = _tf.math.special.bessel_k1e
kv_e = _tfp.math.bessel_kve


def log_cosh(x):
    return x + _tf.math.log1p(_tf.math.expm1(-2 * x) / 2)


def sech(x):
    e = _tf.math.exp(-x)
    return 2 * e / (1 + e ** 2)
