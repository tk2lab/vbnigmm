__author__ = 'TAKEKAWA Takashi <takekawa@tk2lab.org>'
__credits__ = 'Copyright 2020, TAKEKAWA Takashi'


from .gauss_model import BayesianGaussianMixture
from .fixed_model import BayesianFixedNIGMixture
from .nig_model import BayesianNIGMixture


__all__ = [
    'BayesianGaussianMixture',
    'BayesianFixedNIGMixture',
    'BayesianNIGMixture',
]
