.. role:: bash(code)
   :language: bash

Deep Learning Utilities for Pathology
=====================================

The evaluation of histopathological tissue sections are often used in the diagnosis and prognosis of cancer.
While historically studied with a microscope, these images are digitized at a high pace. These image are often so
large, that a collection of them often cannot fit into today's computer memory as well as resulting in complex shapes
with hard-to-define boundaries. Dlup is a collection of software tools used to research computational pathology with
special focus on deep learning that aims to ease the development of new models by simplifying the retrieval and
the management of these slide images as datasets ready to be used for model training and inference.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   tiling
   cli
   contributing
   modules

Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`




.. _Singularity: https://sylabs.io/singularity/
.. _OpenSlide: https://openslide.org
.. _fork of OpenSlide: https://github.com/NKI-AI/OpenSlide
.. _Github repo: https://github.com/NKI-AI/dlup
