.. role:: bash(code)
   :language: bash

Deep Learning Utilities for Pathology
=====================================

The evaluation of histopathological tissue sections is an often used tool in the diagnosis and prognosis of cancer.
While historically studied with a microscope, these images are digitized at a high pace. These image are often so
large that a collection of them will not fit into today's computer memory, as well as resulting in complex shapes
with hard-to-define boundaries. Dlup is a collection of software tools used to facilitate computational
pathology research, with a special focus on deep learning. Dlup aims to ease the development of new models
by simplifying the retrieval and management of whole-slide images, so that they can be quickly turned into
datasets ready to be used for model training and inference.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   tiling
   writers
   backends
   cli
   examples
   contributing
   modules

Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`



.. _OpenSlide: https://openslide.org
.. _fork of OpenSlide: https://github.com/NKI-AI/OpenSlide
.. _Github repo: https://github.com/NKI-AI/dlup
