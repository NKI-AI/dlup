.. highlight:: shell


Installation
============

General
-------
Dlup is the Deep Learning Utilities for Pathology toolkit. dlup contains functions useful for apply deep learning
in computational pathology, such as image normalization, pathology specific augmentations and models. But it also
contains command line tools to download and upload data from data repositories. For a more general introduction
head over to the :doc:`/readme` page.

Dlup uses `OpenSlide`_ to read Whole-Slide Images (WSI). We have some specific changes in our `fork of OpenSlide`_ which
improve the compatibility with other images.

Usage
#####
:doc:`/usage` collects some best practices on how to use dlup in your project.



Build from source
-----------------
The sources for dlup can be downloaded from the `Github repo`_.
You can clone the public repository and install it as follows

.. code-block:: console

    $ git clone git://github.com/NKI-AI/dlup
    $ cd dlup
    $ python setup.py install

It can be beneficial to install our `fork of OpenSlide`_.


Build Docker
------------
Building the docker container can be done as follows (from the dlup directory)

.. code-block:: console

    $ docker build -t dlup:latest . -f docker/Dockerfile

This will also install our OpenSlide version, and include a juypter environment. In non-privileged environments it
might be useful to convert the container to a `Singularity`_ container.


.. _Singularity: https://sylabs.io/singularity/
.. _OpenSlide: https://openslide.org
.. _fork of OpenSlide: https://github.com/NKI-AI/OpenSlide
.. _Github repo: https://github.com/NKI-AI/dlup
