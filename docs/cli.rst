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


SlideScore CLI
--------------
The SlideScore CLI tools require an API key, which can be set to the environmental variable
:code:`SLIDESCORE_API_KEY` (recommended) or through the flag :code:`-t token.txt`.

The following utilities to interact with SlideScore are implemented:

* :code:`dlup slidescore download-wsi`: Download WSIs from SlideScore.
  This requires more permissions than most other API calls.
* :code:`dlup slidescore download-labels`: Download labels from SlideScore.
* :code:`dlup slidescore upload-labels`: Upload labels to SlideScore.
