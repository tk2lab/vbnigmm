"""Function of statistical distributions."""

__author__ = 'TAKEKAWA Takashi <takekawa@tk2lab.org>'
__credits__ = 'Copyright 2020, TAKEKAWA Takashi'


from .basedist import BaseDist
from .dirichlet import Dirichlet
from .wishart import Wishart
from .multinormal import MultivariateNormal
from .blocknormal import BlockNormal
from .truncatedgauss import TruncatedGaussian
from .gamma import Gamma
from .invgauss import InverseGaussian


__all__ = [
    'BaseDist', 'Dirichlet', 'Wishart',
    'MultivariateNormal', 'BlockNormal',
    'TruncatedGaussian' 'Gamma', 'InverseGaussian',
]
