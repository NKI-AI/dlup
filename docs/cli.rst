Command line tools
==================

Dlup provides several command-line tools, all starting with :code:`dlup <command>`. The documentation can
be found using :code:`dlup --help` or similar with the subcommands.

WSI CLI
-------
The command-line interface has some utilities to work with whole-slide images (WSI). For instance,
we can extract important metadata (e.g., mpp, magnification, shape) from a collection of WSIs or store them
as tiles to disk.
The following utilities are implemented:

* :code:`dlup wsi tile`: Tile WSIs to disk.
* :code:`dlup wsi info`: Extract metadata from a WSI.


SlideScore CLI
--------------
The SlideScore CLI tools require an API key, which can be set to the environmental variable
:code:`SLIDESCORE_API_KEY` (recommended) or through the flag :code:`-t token.txt`.
The following utilities are implemented:

* :code:`dlup slidescore download-wsi`: Download WSIs from SlideScore.
  Requires more permissions than most other API calls.
* :code:`dlup slidescore download-labels`: Download labels from SlideScore.
* :code:`dlup slidescore upload-labels`: Upload labels to SlideScore.
