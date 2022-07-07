Command line tools
==================

Dlup provides several command-line tools, all starting with :code:`dlup <command>`. The documentation can
be found using :code:`dlup --help`, or similarly by appending :code:`--help` to any of the subcommands.

WSI CLI
-------
The command-line interface has some utilities to work with whole-slide images (WSI). For instance,
we can extract important metadata (e.g., mpp, magnification, shape) from a collection of WSIs with:

.. code-block:: console

    dlup wsi info [args]

Or we can store the WSIs along with the metadata as tiles to disk. This can be done using:

.. code-block:: console

    dlup wsi tile [args]
