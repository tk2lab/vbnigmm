import tensorflow as _tf
import tensorflow_probability as _tfp


k0_e = _tf.math.special.bessel_k0e
k1_e = _tf.math.special.bessel_k1e
kv_e = _tfp.math.bessel_kve


def is_integer(n):
    try:
        return float(n).is_integer()
    except TypeError:
        return False


def kv_ratio(v, z):
    try:
        if (2. * v).is_integer():
            n = int(round(v + 0.1))
            nu = v - n
            if nu == -0.5:
                ratio = 1.0
            elif nu == 0.0:
                ratio = k1_e(z) / k0_e(z)
            else:
                raise ValueError()
            if n > 0:
                for i in range(n):
                    ratio = 1 / ratio + 2 * nu / z
                    nu += 1
            else:
                for i in range(n):
                    ratio = 1 / (ratio - 2 * nu / z)
                    nu += 1
            return ratio
    except:
        pass
    return kv_e(v + 1, z) / kv_e(v, z)


def log_kv(v, z, name=None):
    return _tf.math.log(kv_e(v, z)) - z


def dv_kv_e(v, z, name=None):
    small = 1e-6
    return (tkv_e(v + small, z) - kv_e(v, z)) / small

def dv_log_kv(v, z, name=None):
    small = 1e-6
    return (log_kv(v + small, z) - log_kv(v, z)) / small
