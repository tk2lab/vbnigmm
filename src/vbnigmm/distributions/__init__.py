"""Function of statistical distributions."""

__author__ = 'TAKEKAWA Takashi <takekawa@tk2lab.org>'
__credits__ = 'Copyright 2020, TAKEKAWA Takashi'


from .basedist import BaseDist
from .dirichlet import Dirichlet
from .gamma import Gamma
from .wishart import Wishart
from .invgauss import InverseGaussian
from .truncatedgauss import TruncatedGaussian


__all__ = [
    'BaseDist', 'Dirichlet', 'Gamma', 'Wishart',
    'InverseGaussian', 'TruncatedGaussian'
]
