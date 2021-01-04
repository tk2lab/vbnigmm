import tensorflow as _tf
import tensorflow_probability as _tfp
import numpy as _np


print = _tf.print

# random
random_seed = _tf.random.set_seed
random_uniform = _tf.random.uniform

# dtype
bool = _tf.bool
int32 = _tf.int32
float32 = _tf.float32
float64 = _tf.float64

# shape
size = _tf.size
shape = _tf.shape
reshape = _tf.reshape
broadcast_to = _tf.broadcast_to

# constractor
as_array = _tf.convert_to_tensor
constant = _tf.constant
cast = _tf.cast
zeros = _tf.zeros
ones = _tf.ones
fill = _tf.fill
range = _tf.range
zeros_like = _tf.zeros_like
ones_like = _tf.ones_like
eye = _tf.eye
one_hot = _tf.one_hot
unique = _tf.unique

# modify 
mask = _tf.boolean_mask
tile = _tf.tile
stack = _tf.stack
concat = _tf.concat
gather = _tf.gather
argsort = _tf.argsort
argmin = _tf.argmin
argmax = _tf.argmax
transpose = _tf.transpose

# reduce
all = _tf.math.reduce_all
any = _tf.math.reduce_any
min = _tf.math.reduce_min
max = _tf.math.reduce_max
sum = _tf.math.reduce_sum
mean = _tf.math.reduce_mean
cumsum = _tf.math.cumsum
cumprod = _tf.math.cumprod
softmax = _tf.nn.softmax
log_sum_exp = _tf.math.reduce_logsumexp

# linalg
tensordot = _tf.tensordot
cov = _tfp.stats.covariance
inv = _tf.linalg.inv
log_det = _tf.linalg.logdet

# 
equal = _tf.equal
case = _tf.case
cond = _tf.cond
where = _tf.where
while_loop = _tf.while_loop
is_native = lambda x: not _tf.is_tensor(x)
native_all = _np.all


def scatter_update(dst, src):
    update =_tf.IndexedSlices(src, _tf.range(_tf.shape(src)[0]))
    dst.scatter_update(update)


def where_func(base, cond, func, args):
    idx = _tf.where(cond)
    args = [_tf.gather(x, idx[:, 0]) for x in args]
    return _tf.tensor_scatter_nd_update(base, idx, func(*args))
