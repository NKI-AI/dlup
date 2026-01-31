Quick Start Guide
=================

This guide will help you get started with DLUP quickly. We'll cover the basics of installation, basic usage, and common workflows.

Installation
------------

Install DLUP using pip:

.. code-block:: bash

   pip install dlup

For the latest development version:

.. code-block:: bash

   git clone https://github.com/NKI-AI/dlup.git
   cd dlup
   pip install -e .

Basic Usage
-----------

Reading Whole Slide Images
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The core functionality of DLUP is reading whole slide images at arbitrary resolutions. Here's how to get started:

.. code-block:: python

   import dlup

   # Open a whole slide image
   with dlup.SlideImage.from_file_path('path/to/slide.svs') as slide:
       print(f"Slide dimensions: {slide.dimensions}")
       print(f"Available resolutions: {slide.spacings}")

       # Read a region at a specific resolution (0.5 μm/px)
       region = slide.read_region(
           location=(1000, 2000),  # (x, y) coordinates
           size=(512, 512),        # (width, height) in pixels
           resolution=0.5          # Resolution in μm/px
       )

       print(f"Region shape: {region.shape}")

Using Different Backends
~~~~~~~~~~~~~~~~~~~~~~~~~

DLUP supports multiple backends for reading whole slide images. Choose the best one for your use case:

.. code-block:: python

   # OpenSlide backend (default, widely supported)
   slide = dlup.SlideImage.from_file_path('slide.svs', backend='openslide')

   # Fastslide backend
   slide = dlup.SlideImage.from_file_path('slide.svs', backend='fastslide')

Working with Annotations
~~~~~~~~~~~~~~~~~~~~~~~~

DLUP supports loading annotations in multiple formats:

.. code-block:: python

   # Load GeoJSON annotations
   annotations = dlup.annotations.AnnotationSet.from_geojson('annotations.json')

   # Load HALO annotations
   annotations = dlup.annotations.AnnotationSet.from_halo_xml('annotations.xml')

   # Filter annotations by type
   nuclei = annotations.filter_by_type('nuclei')
   glands = annotations.filter_by_type('gland')

   # Create a mask from annotations
   mask = annotations.to_mask(size=(1024, 1024))

PyTorch Integration
-------------------

Creating Tiled Datasets
~~~~~~~~~~~~~~~~~~~~~~~

For training deep learning models, you'll want to create tiled datasets:

.. code-block:: python

   from torch.utils.data import DataLoader

   # Create a dataset that generates tiles from a slide
   dataset = dlup.tiling.SlideImageTilingDataset(
       slide_path='path/to/slide.svs',
       tile_size=(256, 256),      # Size of each tile
       tile_overlap=(0, 0),       # Overlap between tiles
       resolution=0.5,            # Resolution for tiles
       transform=None             # Optional transforms
   )

   # Use with PyTorch DataLoader
   dataloader = DataLoader(
       dataset,
       batch_size=8,
       num_workers=4,
       shuffle=True
   )

   # Iterate through batches
   for batch in dataloader:
       images, masks = batch  # If using annotation masks
       # Process your batch here

Custom Transforms
~~~~~~~~~~~~~~~~~

Apply transforms to your data:

.. code-block:: python

   import torchvision.transforms as T

   # Define transforms
   transforms = T.Compose([
       T.ToTensor(),
       T.Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])
   ])

   # Create dataset with transforms
   dataset = dlup.tiling.SlideImageTilingDataset(
       slide_path='slide.svs',
       tile_size=(256, 256),
       transform=transforms
   )

Background Removal
------------------

Remove background from slides for better training:

.. code-block:: python

   # Create background estimator
   background_estimator = dlup.background.BackgroundEstimator()

   # Estimate background
   background_mask = background_estimator.estimate(slide)

   # Apply background removal
   foreground_slide = slide.apply_background_mask(~background_mask)

Performance Optimization
-------------------------

Caching
~~~~~~~

Enable caching for better performance when reading overlapping regions:

.. code-block:: python

   # Create a cache manager
   cache = dlup.CacheManager.create(capacity=1000)

   # Apply to slide
   slide.set_cache_manager(cache)

   # Monitor cache performance
   stats = cache.get_basic_stats()
   print(f"Hit ratio: {stats.hit_ratio:.2%}")

Memory Management
~~~~~~~~~~~~~~~~~

For large slides, use memory-efficient reading:

.. code-block:: python

   # Read only the region you need
   region = slide.read_region(
       location=(x, y),
       size=(width, height),
       resolution=resolution
   )

   # Process and release
   # The region data will be automatically cleaned up

Batch Processing
~~~~~~~~~~~~~~~~

Process multiple slides efficiently:

.. code-block:: python

   slide_paths = ['slide1.svs', 'slide2.svs', 'slide3.svs']

   for slide_path in slide_paths:
       with dlup.SlideImage.from_file_path(slide_path) as slide:
           # Process each slide
           # Context manager ensures proper cleanup

Common Workflows
----------------

Classification Workflow
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # 1. Load slide and annotations
   slide = dlup.SlideImage.from_file_path('slide.svs')
   annotations = dlup.annotations.AnnotationSet.from_geojson('annotations.json')

   # 2. Create dataset with labels
   dataset = dlup.tiling.SlideImageTilingDataset(
       slide_path='slide.svs',
       tile_size=(256, 256),
       annotations=annotations,
       label_extractor=lambda ann: 1 if ann.type == 'tumor' else 0
   )

   # 3. Train your model
   # ... your training code here

Segmentation Workflow
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # 1. Load slide and create annotation masks
   slide = dlup.SlideImage.from_file_path('slide.svs')
   annotations = dlup.annotations.AnnotationSet.from_geojson('annotations.json')

   # 2. Create dataset for segmentation
   dataset = dlup.tiling.SlideImageTilingDataset(
       slide_path='slide.svs',
       tile_size=(256, 256),
       mask_annotations=annotations,
       mask_types=['nuclei', 'glands']
   )

   # 3. Train segmentation model
   # ... your training code here

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**Import Error**: Make sure all dependencies are installed:

.. code-block:: bash

   pip install numpy shapely pillow packaging tifffile
   # Also install openslide-python if using the OpenSlide backend

**Memory Issues**: For large slides, process in smaller batches:

.. code-block:: python

   # Use smaller tile sizes or process slides one at a time
   dataset = dlup.tiling.SlideImageTilingDataset(
       slide_path='slide.svs',
       tile_size=(128, 128),  # Smaller tiles use less memory
       max_tiles_per_slide=1000  # Limit tiles per slide
   )

**Performance Issues**: Enable caching and use appropriate backends:

.. code-block:: python

   # Use VIPS backend for better performance
   slide = dlup.SlideImage.from_file_path('slide.svs', backend='vips')

   # Enable caching
   cache = dlup.CacheManager.create(capacity=2000)
   slide.set_cache_manager(cache)

Next Steps
----------

Now that you have the basics, explore these areas:

- :doc:`user_guide` - Detailed usage guide with advanced features
- :doc:`../api/index` - Complete API reference
- :doc:`../examples/index` - Practical examples and tutorials

For more help:

- Check the `GitHub repository <https://github.com/NKI-AI/dlup>`_
- Report issues at `GitHub Issues <https://github.com/NKI-AI/dlup/issues>`_
- Join discussions at `GitHub Discussions <https://github.com/NKI-AI/dlup/discussions>`_
