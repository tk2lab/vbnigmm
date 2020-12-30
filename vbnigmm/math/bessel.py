import tensorflow as _tf
import tensorflow_probability as _tfp


k0_e = _tf.math.special.bessel_k0e
k1_e = _tf.math.special.bessel_k1e
kv_e = _tfp.math.bessel_kve


def kv_ratio(v, z, name=None):
    with _tf.name_scope(name or 'kv_ratio'):
        log_ratio = log_kv(v + 1, z) - log_kv(v, z)
        return _tf.math.exp(log_ratio)


def kv_ratio_h(v, z, name=None):
    with _tf.name_scope(name or 'kv_ratio_h'):
        m = _tf.cast(_tf.round(2 * v), _tf.int32)
        absm = _tf.math.abs(m)
        even = absm % 2 == 0
        negative = m < 0
        nu = _tf.where(even, 1.0, 0.5)
        ratio = _tf.where(even, k1_e(z) / k0_e(z), 1.0)
        n = _tf.where(even, absm // 2, (absm + 1) // 2)
        n = _tf.where(negative, n - 1, n)
        ratio = update_kv_ratio(nu, z, ratio, n)
        return _tf.where(negative, 1 / ratio, ratio)


def update_kv_ratio(nu, z, ratio, n):
    def update(i, nu, ratio):
        ratio = 2 * nu / z + 1 / ratio
    _, ratio = _tf.while_loop(
        cond=lambda i, ratio: i < _tf.cast(n, _tf.float32),
        body=lambda i, ratio: (i + 1, 2 * (i + nu) / z + 1 / ratio),
        loop_vars=(_tf.constant(0.), ratio)
    )
    return ratio


def log_kv(v, z, name=None):
    with _tf.name_scope(name or 'log_kv'):
        return _tf.math.log(kv_e(v, z)) - z


def dv_kv_e(v, z, name=None):
    with _tf.name_scope(name or 'dv_kv_e'):
        small = 1e-6
        return (tkv_e(v + small, z) - kv_e(v, z)) / small

def dv_log_kv(v, z, name=None):
    with _tf.name_scope(name or 'dv_log_kv'):
        small = 1e-6
        return (log_kv(v + small, z) - log_kv(v, z)) / small
