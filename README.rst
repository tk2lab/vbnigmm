vbnigmm
=======

Variational Bayes algorithm for Normal Inverse Gaussian Mixture Models

Installation
------------

The package can be build using poetry and installed using pip:

.. code-block:: bash

    pip install vbnigmm

Examples
--------

If you want to apply vbnigmm to your data,
you can run the following code:

.. code-block:: python

    from vbnigmm import NormalInverseGaussMixture as Model

    # x is numpy.ndarray of 2D

    model = Model()
    model.fit(x)
    label = model.predict(x)

Citation
--------

If you use vbnigmm in a scientific paper,
please consider citing the following paper:

Takashi Takekawa, `Clustering of non-Gaussian data by variational Bayes for normal inverse Gaussian mixture models. <https://arxiv.org/abs/2009.06002>`_ arXiv preprint arXiv:2009.06002 (2020).
