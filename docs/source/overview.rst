Overview
========

DLUP (Deep Learning Utilities for Pathology) is a comprehensive Python library designed to simplify working with whole slide images (WSI) in deep learning applications. It provides seamless reading of whole-slide images at any arbitrary resolution, supports multiple backends, and includes powerful annotation and dataset classes for building efficient training pipelines.

Core Components
---------------

DLUP consists of several key modules that work together to provide a complete solution for pathology image analysis:

.. grid:: 1 2 3 3
   :gutter: 2

   .. grid-item-card:: 🖼️ **SlideImage**
      :class-card: overview-card

      Core class for reading and manipulating whole slide images. Supports arbitrary resolution reading and multiple backends.

   .. grid-item-card:: 📐 **Geometry**
      :class-card: overview-card

      Geometric utilities for handling regions, points, polygons, and spatial operations on whole slide images.

   .. grid-item-card:: 🏷️ **Annotations**
      :class-card: overview-card

      Support for loading, manipulating, and exporting annotations in multiple formats (GeoJSON, HALO, ASAP, V7 Darwin).

   .. grid-item-card:: 🔲 **Tiling**
      :class-card: overview-card

      Dataset classes and utilities for creating tile-based datasets compatible with PyTorch DataLoader.

   .. grid-item-card:: 🎨 **Background**
      :class-card: overview-card

      Background removal and foreground detection algorithms for whole slide images.

   .. grid-item-card:: ✍️ **Writers**
      :class-card: overview-card

      Utilities for writing processed images and annotations to various formats.

Key Features
------------

🎯 **Arbitrary Resolution Reading**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

DLUP can read whole slide images at any arbitrary resolution by intelligently interpolating between the available pyramidal levels. This eliminates the need to work with fixed resolution levels and provides maximum flexibility for your analysis.

.. code-block:: python

   # Read at exactly 0.5 μm/px resolution
   region = slide.read_region(
       location=(1000, 2000),
       size=(512, 512),
       resolution=0.5
   )

🔌 **Multiple Backend Support**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

DLUP supports multiple backends for reading whole slide images, allowing you to choose the best option for your specific use case:

- **OpenSlide Backend**: Fast, widely supported backend for most formats
- **VIPS Backend**: Advanced image processing with excellent performance
- **TIFFile Backend**: For generic TIFF files
- **Custom Backends**: Easy to implement your own backend

.. code-block:: python

   # Use VIPS backend for better performance
   slide = dlup.SlideImage.from_file_path(
       'slide.svs',
       backend='vips'
   )

📊 **PyTorch Integration**
^^^^^^^^^^^^^^^^^^^^^^^^^^

DLUP provides dataset classes that are fully compatible with PyTorch's DataLoader, making it easy to create efficient training pipelines:

.. code-block:: python

   from torch.utils.data import DataLoader

   # Create a tiled dataset
   dataset = dlup.tiling.SlideImageTilingDataset(
       slide_path='slide.svs',
       tile_size=(256, 256),
       tile_overlap=(0, 0),
       transform=transforms.ToTensor()
   )

   # Use with DataLoader for multi-worker loading
   dataloader = DataLoader(
       dataset,
       batch_size=8,
       num_workers=4,
       shuffle=True
   )

🏷️ **Annotation Support**
^^^^^^^^^^^^^^^^^^^^^^^^^^

DLUP supports loading and manipulating annotations in multiple formats commonly used in pathology:

- **GeoJSON**: Standard geographic JSON format
- **HALO**: Indica Labs HALO annotation format
- **ASAP**: Automated Slide Analysis Platform format
- **V7 Darwin**: V7 Darwin annotation format

.. code-block:: python

   # Load annotations from multiple formats
   annotations = dlup.annotations.AnnotationSet.from_geojson('annotations.json')

   # Filter annotations by type
   nuclei = annotations.filter_by_type('nuclei')
   glands = annotations.filter_by_type('gland')

   # Create masks from annotations
   mask = annotations.to_mask(size=(1024, 1024))

⚡ **High Performance**
^^^^^^^^^^^^^^^^^^^^^^^

DLUP is optimized for performance with:

- Efficient caching mechanisms
- Memory-mapped file access
- Lazy loading of image data
- Multi-threaded operations where appropriate

.. code-block:: python

   # Enable caching for better performance
   slide.set_cache_manager(dlup.CacheManager.create(capacity=1000))

   # Monitor cache performance
   stats = slide.cache_manager.get_basic_stats()
   print(f"Hit ratio: {stats.hit_ratio:.2%}")

Supported Formats
-----------------

DLUP supports a wide range of whole slide image formats through its backend system:

.. list-table:: Supported Formats
   :header-rows: 1
   :widths: 20 30 50

   * - Format
     - Extension
     - Description
   * - Aperio SVS
     - .svs
     - Aperio ScanScope virtual slide format
   * - Hamamatsu NDPI
     - .ndpi
     - Hamamatsu NanoZoomer format
   * - 3DHistech MRXS
     - .mrxs
     - 3DHistech MIRAX format
   * - Leica SCN
     - .scn
     - Leica SCN400 format
   * - Hamamatsu VMS/VMU
     - .vms, .vmu
     - Hamamatsu format
   * - Generic TIFF
     - .tif, .tiff
     - Standard TIFF format (including BigTIFF)
   * - Ventana BIF
     - .bif
     - Ventana BIF format

Architecture
------------

DLUP is organized into several key modules:

.. code-block:: text

   dlup/
   ├── SlideImage           # Core image reading functionality
   ├── backends/            # Backend implementations (OpenSlide, VIPS, etc.)
   ├── geometry/            # Geometric utilities and spatial operations
   ├── annotations/         # Annotation loading and manipulation
   ├── tiling/              # Dataset classes for PyTorch integration
   ├── background/          # Background removal algorithms
   ├── writers/             # Image and annotation writers
   └── utils/               # Utility functions

Each module is designed to be independent yet work seamlessly with others, providing maximum flexibility for your specific use case.

Use Cases
---------

DLUP is particularly well-suited for:

🔬 **Digital Pathology Research**
   Building deep learning models for pathology image analysis, including classification, segmentation, and detection tasks.

🏥 **Clinical Applications**
   Developing tools for clinical decision support, biomarker discovery, and automated analysis of pathology slides.

📈 **Large-Scale Studies**
   Processing large cohorts of whole slide images for epidemiological studies and clinical trials.

🎓 **Educational Tools**
   Creating interactive tools for pathology education and training.

Getting Help
------------

- **Documentation**: This comprehensive documentation covers all features and provides examples.
- **GitHub Issues**: Report bugs or request features at `https://github.com/NKI-AI/dlup/issues`_
- **Discussions**: Join the community discussion at `https://github.com/NKI-AI/dlup/discussions`_

License
-------

DLUP is released under the Apache License 2.0. See the LICENSE file for details.
