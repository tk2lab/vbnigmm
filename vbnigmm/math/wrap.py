import math

from numpy import inf
from tensorflow import int32
from tensorflow import float32
from tensorflow import convert_to_tensor as as_array
from tensorflow import size
from tensorflow import shape
from tensorflow import zeros
from tensorflow import ones
from tensorflow import range
from tensorflow import fill
from tensorflow import zeros_like
from tensorflow import ones_like
from tensorflow import eye
from tensorflow import argsort
from tensorflow import gather
from tensorflow import boolean_mask as mask
from tensorflow import stack
from tensorflow import concat
from tensorflow import transpose
from tensorflow import tensordot
from tensorflow.math import abs as fabs
from tensorflow.math import round
from tensorflow.math import exp
from tensorflow.math import log
from tensorflow.math import pow
from tensorflow.math import sqrt
from tensorflow.math import xlogy
from tensorflow.math import digamma
from tensorflow.math import lgamma
from tensorflow.math import reduce_all as all
from tensorflow.math import reduce_sum as sum
from tensorflow.math import reduce_max as max
from tensorflow.math import reduce_logsumexp as log_sum_exp
from tensorflow.math import cumsum
from tensorflow.math import cumprod
from tensorflow.nn import softmax
from tensorflow.math.special import bessel_k0e as k0e
from tensorflow.math.special import bessel_k1e as k1e
from tensorflow.linalg import inv
from tensorflow.linalg import logdet as log_det


log2 = math.log(2)
log2pi = math.log(2 * math.pi)
sqrt2 = math.sqrt(2)
sqrtpi = math.sqrt(math.pi)
