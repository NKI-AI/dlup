.. highlight:: shell


Installation
============

Usage
#####
:doc:`/usage` collects some best practices on how to use dlup in your project.



Build from source
-----------------
The sources for dlup can be downloaded from the `Github repo`_.
You can clone the public repository and install the latest version as follows

.. code-block:: console

    $ git clone git@github.com:NKI-AI/dlup.git dlup
    $ cd dlup
    $ pip install -e .

Adding the :code:`-e` allows you to update the repository and have the latest version in your python
environment. If you want to use a specific version of dlup (v0.1.0 in the example),
use something along the following

.. code-block:: console

    $ git clone git@github.com:NKI-AI/dlup.git dlup
    $ git checkout tags/v0.1.0 -b v0.1.0
    $ cd dlup
    $ pip install -e .


Also install our `fork of OpenSlide`_ if you want broader support for different image types.


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
