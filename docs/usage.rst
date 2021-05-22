How to use dlup in your project
===============================
As dlup is a helper library, it is convenient to include dlup in your project. Recommended is to use a Git submodule or
a docker container (or both!).


As a Git submodule
------------------
If you want to use dlup in your project, it can be advantageous to freeze the version, and include dlup
in your Git repository. For instance, you can use something like:

.. code-block:: console

    $ git submodule add git@github.com:NKI-AI/dlup.git third_party/dlup
    $ git submodule init
    $ git submodule update

If you want the latest version, you can now commit:

.. code-block:: console

    $ git commit -a

However, if you would want a specific version, for instance :code:`v0.1.0`, do the following before you commit:

.. code-block:: console

    $ cd third_party/dlup
    $ git reset --hard v0.1.0

When committed, you can check the status of your submodule

.. code-block:: console

    $ cd third_party/dlup
    $ git submodule status

If succesful, you should see something like:

.. code-block:: console

    $ e50b81a5fbfcef63eb214d344438ada86a4e5dc0 dlup (v0.1.0)

The module can subsequently be installed. Please visit the :doc:`/installation` page for further details.


As a Docker container
---------------------
The docker can be built using:

.. code-block:: console

    $ docker build -t dlup:latest . -f docker/Dockerfile

This container can be subsequently included in your project using the :code:`FROM` statement.
