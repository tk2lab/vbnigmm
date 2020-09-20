"""Function of statistical distributions."""

__author__ = 'TAKEKAWA Takashi <takekawa@tk2lab.org>'
__credits__ = 'Copyright 2020, TAKEKAWA Takashi'


from .basedist import BaseDist
from .dirichlet import Dirichlet
from .normal import Normal
from .wishart import Wishart
from .truncatedgauss import TruncatedGaussian
from .gamma import Gamma
from .invgauss import InverseGaussian


__all__ = [
    'BaseDist', 'Dirichlet', 'Normal', 'Wishart',
    'TruncatedGaussian' 'Gamma', 'InverseGaussian',
]
