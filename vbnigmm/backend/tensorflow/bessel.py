import tensorflow as _tf
import tensorflow_probability as _tfp

#from .kvln import kv_ratio as _kv_ratio
#from .kvln import log_kv as _log_kv


k0_e = _tf.math.special.bessel_k0e
k1_e = _tf.math.special.bessel_k1e
kv_e = _tfp.math.bessel_kve


def kv_ratio(v, z):
    v = _tf.convert_to_tensor(v, z.dtype)
    return kv_e(v + 1, z) / kv_e(v, z)
#kv_ratio = _kv_ratio


def log_kv(v, z):
    v = _tf.convert_to_tensor(v, z.dtype)
    return _tf.math.log(kv_e(v, z)) - z
#log_kv = _log_kv


def dv_kv_e(v, z, name=None):
    small = 1e-6
    return (tkv_e(v + small, z) - kv_e(v, z)) / small


def dv_log_kv(v, z, name=None):
    small = 1e-6
    return (log_kv(v + small, z) - log_kv(v, z)) / small
