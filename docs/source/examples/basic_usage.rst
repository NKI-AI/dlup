Basic Usage Examples
====================

This page demonstrates the most common usage patterns for DLUP.

Opening and Reading Slides
--------------------------

The most basic operation - opening a slide and reading regions:

.. code-block:: python

   import dlup

   # Open a slide file
   with dlup.SlideImage.from_file_path('path/to/slide.svs') as slide:
       print(f"Slide dimensions: {slide.dimensions}")
       print(f"Number of levels: {slide.level_count}")

       # Read a region at level 0 (highest resolution)
       region = slide.read_region(
           location=(1000, 2000),  # (x, y) coordinates
           size=(512, 512),        # (width, height) in pixels
           level=0                 # Pyramid level
       )

       print(f"Region shape: {region.shape}")

Working with Different Resolutions
---------------------------------

DLUP supports reading at arbitrary resolutions, not just predefined levels:

.. code-block:: python

   with dlup.SlideImage.from_file_path('slide.svs') as slide:
       # Read at exactly 0.5 μm/px resolution
       region = slide.read_region(
           location=(1000, 2000),
           size=(512, 512),
           resolution=0.5  # μm/px
       )

       # Read at 2x magnification
       region = slide.read_region(
           location=(500, 1000),
           size=(256, 256),
           scaling=2.0  # Scale factor
       )

Using Different Backends
------------------------

Choose the backend that best fits your needs:

.. code-block:: python

   # OpenSlide backend (default, widely supported)
   slide = dlup.SlideImage.from_file_path('slide.svs', backend='openslide')

   # VIPS backend (advanced image processing)
   slide = dlup.SlideImage.from_file_path('slide.svs', backend='vips')

   # PyVIPS backend (pure Python)
   slide = dlup.SlideImage.from_file_path('slide.svs', backend='pyvips')

Loading and Using Annotations
-----------------------------

Work with annotations in various formats:

.. code-block:: python

   # Load GeoJSON annotations
   annotations = dlup.annotations.AnnotationSet.from_geojson('annotations.json')

   # Load HALO XML annotations
   annotations = dlup.annotations.AnnotationSet.from_halo_xml('annotations.xml')

   # Filter by annotation type
   nuclei = annotations.filter_by_type('nuclei')
   glands = annotations.filter_by_type('gland')

   # Create masks from annotations
   mask = annotations.to_mask(size=(1024, 1024))

Creating Simple Datasets
------------------------

For machine learning applications:

.. code-block:: python

   from torch.utils.data import DataLoader

   # Create a basic tiled dataset
   dataset = dlup.tiling.SlideImageTilingDataset(
       slide_path='slide.svs',
       tile_size=(256, 256),
       tile_overlap=(0, 0)
   )

   # Use with PyTorch DataLoader
   dataloader = DataLoader(dataset, batch_size=8, num_workers=4)

   # Iterate through batches
   for batch in dataloader:
       images = batch  # Shape: (batch_size, 256, 256, 3)
       # Process your batch

Performance Optimization
------------------------

Enable caching for better performance:

.. code-block:: python

   # Create and configure cache
   cache = dlup.CacheManager.create(capacity=1000)
   slide.set_cache_manager(cache)

   # Monitor cache performance
   stats = cache.get_basic_stats()
   print(f"Hit ratio: {stats.hit_ratio:.2%}")

Error Handling
--------------

Proper error handling for robust applications:

.. code-block:: python

   try:
       slide = dlup.SlideImage.from_file_path('slide.svs')
       region = slide.read_region((0, 0), (256, 256), 0.5)
   except FileNotFoundError:
       print("Slide file not found")
   except dlup.DlupError as e:
       print(f"DLUP error: {e}")
   except Exception as e:
       print(f"Unexpected error: {e}")

Batch Processing
----------------

Process multiple slides efficiently:

.. code-block:: python

   slide_paths = ['slide1.svs', 'slide2.svs', 'slide3.svs']

   for slide_path in slide_paths:
       with dlup.SlideImage.from_file_path(slide_path) as slide:
           # Process each slide
           process_slide(slide)

Memory Management
-----------------

For large slides, manage memory carefully:

.. code-block:: python

   # Process in smaller chunks for large slides
   chunk_size = 1000

   with dlup.SlideImage.from_file_path('large_slide.svs') as slide:
       for y in range(0, slide.dimensions[1], chunk_size):
           for x in range(0, slide.dimensions[0], chunk_size):
               region = slide.read_region(
                   (x, y),
                   (min(chunk_size, slide.dimensions[0] - x),
                    min(chunk_size, slide.dimensions[1] - y)),
                   0.5
               )
               # Process chunk and release memory
