import tensorflow as _tf


# dtype
int32 = _tf.int32
float32 = _tf.float32

# shape
size = _tf.size
shape = _tf.shape

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

# modify 
mask = _tf.boolean_mask
tile = _tf.tile
stack = _tf.stack
concat = _tf.concat
gather = _tf.gather
argsort = _tf.argsort
transpose = _tf.transpose

# reduce
all = _tf.math.reduce_all
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
inv = _tf.linalg.inv
log_det = _tf.linalg.logdet

# 
case = _tf.case
where = _tf.where
