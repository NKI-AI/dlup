DLUP Documentation
==================

.. raw:: html

   <div style="text-align: center; margin-bottom: 2rem;">
      <img src="_static/dlup-logo.png" alt="DLUP Logo" style="max-width: 200px; height: auto; margin-bottom: 1rem;" />
   </div>

**Deep Learning Utilities for Pathology**

DLUP (Deep Learning Utilities for Pathology) offers a comprehensive set of utilities to ease the process of running Deep Learning algorithms on Whole Slide Images (WSI). It provides seamless reading of whole-slide images at any arbitrary resolution, supports multiple backends, and includes powerful annotation and dataset classes.

.. grid:: 1 2 2 2
   :gutter: 3
   :margin: 3
   :padding: 0

   .. grid-item-card:: 🚀 **Quick Start**
      :class-card: card-quick-start

      Get started with DLUP in minutes with our comprehensive guides and examples.

      +++

      .. button-ref:: quickstart
         :ref-type: doc
         :click-parent:
         :color: primary

         View Quick Start Guide

   .. grid-item-card:: 📚 **API Reference**
      :class-card: card-api

      Complete API documentation for Python and C++ interfaces.

      +++

      .. button-ref:: api-reference
         :ref-type: doc
         :click-parent:
         :color: primary

         Browse API Docs

   .. grid-item-card:: 🔧 **User Guide**
      :class-card: card-guide

      Learn how to use DLUP's advanced features and customize for your needs.

      +++

      .. button-ref:: user-guide
         :ref-type: doc
         :click-parent:
         :color: primary

         Read User Guide

   .. grid-item-card:: 💻 **Examples**
      :class-card: card-examples

      Practical examples and tutorials to get you started quickly.

      +++

      .. button-ref:: examples
         :ref-type: doc
         :click-parent:
         :color: primary

         View Examples

Key Features
------------

🎯 **Arbitrary Resolution Reading**
   Seamlessly read whole-slide images at any arbitrary resolution by interpolating between pyramidal levels.

🔌 **Multiple Backend Support**
   Supports multiple backends including OpenSlide, FastSlide, and custom backends for maximum flexibility.

📊 **Dataset Classes**
   Tile-by-tile dataset classes compatible with PyTorch for efficient training pipelines.

🏷️ **Annotation Support**
   Load and manipulate annotations in multiple formats: GeoJSON, V7 Darwin, HALO, and ASAP.

🔄 **Transforms & Augmentation**
   Built-in transforms for handling annotations per tile and creating mask targets.

⚡ **High Performance**
   Optimized for speed with efficient caching and memory management.

Installation
------------

Install DLUP using pip:

.. code-block:: bash

   pip install dlup

For development or to access the latest features:

.. code-block:: bash

   git clone https://github.com/NKI-AI/dlup.git
   cd dlup
   pip install -e .

Quick Example
-------------

Here's a simple example to get you started:

.. code-block:: python

   import dlup

   # Open a whole slide image
   with dlup.SlideImage.from_file_path('path/to/slide.svs') as slide:
       # Read a region at arbitrary resolution
       region = slide.read_region(
           location=(1000, 2000),
           size=(512, 512),
           resolution=0.5  # 0.5 μm/px
       )

       # Get slide properties
       print(f"Dimensions: {slide.dimensions}")
       print(f"Spacing: {slide.spacing}")

   # Use with PyTorch DataLoader
   from torch.utils.data import DataLoader

   dataset = dlup.tiling.SlideImageTilingDataset(
       slide_path='path/to/slide.svs',
       tile_size=(256, 256),
       tile_overlap=(0, 0)
   )

   dataloader = DataLoader(dataset, batch_size=8, num_workers=4)

Supported Formats
-----------------

DLUP supports a wide range of whole slide image formats:

- **SVS** (Aperio ScanScope)
- **NDPI** (Hamamatsu NanoZoomer)
- **MRXS** (3DHistech MIRAX)
- **SCN** (Leica SCN400)
- **VMS** and **VMU** (Hamamatsu)
- **TIF** and **TIFF** (Generic TIFF)
- **BIF** (Ventana BIF)

Backend Support
---------------

Choose from multiple backends based on your needs:

- **OpenSlide**: Fast, widely supported backend for most formats
- **VIPS**: Advanced image processing with excellent performance
- **Custom backends**: Extend DLUP with your own backend implementation

Citation
--------

If you use DLUP in your research, please cite:

.. code-block:: bibtex

   @software{dlup,
     author = {Teuwen, J., Romor, L., Pai, A., Schirris, Y., Marcus, E.},
     month = {8},
     title = {{DLUP: Deep Learning Utilities for Pathology}},
     url = {https://github.com/NKI-AI/dlup},
     version = {0.9.4},
     year = {2024}
   }

Contributing
------------

We welcome contributions! See our `GitHub repository <https://github.com/NKI-AI/dlup>`_ for guidelines.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   overview
   quickstart
   user_guide
   api/index
   examples/index
   contributing

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
