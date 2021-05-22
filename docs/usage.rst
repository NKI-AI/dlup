How to use DLUP in your project
===============================
There are several ways to include DLUP in your project, of which Git submodules or dockers are likely
the most convenient for you.


As a python package
-------------------
To use DLUP in a project:

.. code-block:: python

    import dlup


As a Git submodule
------------------
If you want to use DLUP in your project, it can be advantageous to freeze the version, and include DLUP
in your Git repository. For instance, you can use something like:

.. code-block:: console

    $ git submodule add git://github.com/NKI-AI/dlup third_party/dlup
    $ git commit -m "Added DLUP as submodule to the project."
    $ git push

If you want to update the submodule to the latest version, you need to execute :code:`git pull` in the
submodules directory :code:`third_party/dlup`.


As a Docker container
---------------------
The docker can be built using:

.. code-block:: console

    $ docker build -t dlup:latest . -f docker/Dockerfile

This container can be subsequently included in your project using the :code:`FROM` statement.
