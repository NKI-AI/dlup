.. role:: bash(code)
   :language: bash


Installation and Usage
======================

Build from Source
-----------------
The sources for dlup can be downloaded from the `Github repo`_.
You can clone the public repository and install the latest version as follows:

.. code-block:: console

    git clone git@github.com:NKI-AI/dlup.git dlup
    cd dlup
    pip install -e .

Adding the :code:`-e` flag allows you to update the repository and see those changes reflected in your python
environment. If you want to use a specific version of dlup (v0.1.0 in the example), use the following:

.. code-block:: console

    git clone git@github.com:NKI-AI/dlup.git dlup
    git checkout tags/v0.1.0 -b v0.1.0
    cd dlup
    pip install -e .

Replace any occurrences of :code:`v0.1.0` in the example with the version you want to install.

Also install our `fork of OpenSlide`_ if you want broader support for different image types.


Build Docker
------------
To build the docker container, navigate to the repo's root directory and run:

.. code-block:: console

    docker build -t dlup:latest . -f docker/Dockerfile

This will also install our OpenSlide version, and include a juypter environment. This container can be
subsequently included in your project using the :code:`FROM` statement.

In non-privileged environments it might be useful to convert the container to a `Singularity`_ container.


Using dlup as a Git Submodule
-----------------------------
If you want to use dlup in your project, it can be advantageous to freeze the version and include dlup
in your Git repository. You can do this by running:

.. code-block:: console

    git submodule add git://github.com/NKI-AI/dlup third_party/dlup
    git commit -m "Added dlup as submodule to the project."
    git push

If you want to update the submodule to the latest version, you need to execute :code:`git pull` in the
submodules directory :code:`third_party/dlup`.






.. _Singularity: https://sylabs.io/singularity/
.. _OpenSlide: https://openslide.org
.. _fork of OpenSlide: https://github.com/NKI-AI/OpenSlide
.. _Github repo: https://github.com/NKI-AI/dlup
