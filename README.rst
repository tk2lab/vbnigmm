vbnigmm
=======

Variational Bayes algorithm for Normal Inverse Gaussian Mixture Models

Instration:
-----------

The package can be build using poetry and installed using pip:

.. code-block:: bash

    poetry build
    pip install dist/vbnigmm-1.0.0-py3-none-any.whl

Examples:
---------

.. code-block:: python

    from vbnigmm import BaysianNIGMixture as Model

    model = Model()
    model.fit(x)
    label = model.predict(x)

Citation:
---------

If you use vbnigmm in a scientific paper,
please consider citing the following paper:

Takashi Takekawa, `Clustering of non-Gaussian data by variational Bayes for normal inverse Gaussian mixture models. <https://arxiv.org/abs/2009.06002>`_ arXiv preprint arXiv:2009.06002 (2020).
