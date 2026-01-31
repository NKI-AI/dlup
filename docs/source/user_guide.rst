User Guide
==========

This comprehensive guide covers advanced usage of DLUP, including detailed explanations of features, best practices, and advanced workflows.

Advanced Slide Reading
----------------------

Coordinate Systems
~~~~~~~~~~~~~~~~~~

DLUP uses **level-native coordinates** for reading regions. Understanding coordinate systems is crucial for correct usage:

.. code-block:: python

   import dlup

   with dlup.SlideImage.from_file_path('slide.svs') as slide:
       # Get slide properties
       print(f"Level 0 dimensions: {slide.level_dimensions[0]}")
       print(f"Level 1 dimensions: {slide.level_dimensions[1]}")

       # Convert between coordinate systems
       level0_x, level0_y = slide.convert_level_native_to_level0(
           x=100, y=200, level=2
       )

       # Convert from level 0 to specific level
       level2_x, level2_y = slide.convert_level0_to_level_native(
           x=400, y=800, level=2
       )

Resolution and Scaling
~~~~~~~~~~~~~~~~~~~~~~

DLUP supports reading at arbitrary resolutions, not just the predefined pyramid levels:

.. code-block:: python

   # Read at exactly 0.5 μm/px resolution
   region = slide.read_region(
       location=(1000, 2000),
       size=(512, 512),
       resolution=0.5  # μm/px
   )

   # Read at 2x magnification of level 0
   region = slide.read_region(
       location=(500, 1000),
       size=(256, 256),
       scaling=2.0  # 2x scale factor
   )

Efficient Reading Patterns
~~~~~~~~~~~~~~~~~~~~~~~~~~

For optimal performance, use these patterns:

.. code-block:: python

   # 1. Batch multiple reads when possible
   regions = []
   for x, y in coordinates:
       regions.append(slide.read_region((x, y), (256, 256), 0.5))

   # 2. Use context managers for proper cleanup
   with dlup.SlideImage.from_file_path('slide.svs') as slide:
       # All operations within this block
       pass  # Slide automatically closed

   # 3. Enable caching for overlapping reads
   cache = dlup.CacheManager.create(capacity=1000)
   slide.set_cache_manager(cache)

Backend Selection Guide
-----------------------

Choosing the Right Backend
~~~~~~~~~~~~~~~~~~~~~~~~~~

OpenSlide Backend
^^^^^^^^^^^^^^^^^

**Best for**: Most standard use cases, compatibility, stability

.. code-block:: python

   slide = dlup.SlideImage.from_file_path(
       'slide.svs',
       backend='openslide'
   )

**Features**:
- Wide format support
- Stable and well-tested
- Good performance for most cases

VIPS Backend
^^^^^^^^^^^^

**Best for**: Advanced image processing, high-performance scenarios

.. code-block:: python

   slide = dlup.SlideImage.from_file_path(
       'slide.svs',
       backend='vips'
   )

**Features**:
- Excellent performance
- Advanced color management
- Memory efficient
- Support for complex image operations

PyVIPS Backend
^^^^^^^^^^^^^^

**Best for**: Pure Python environments, custom processing

.. code-block:: python

   slide = dlup.SlideImage.from_file_path(
       'slide.svs',
       backend='OPENSLIDE'
   )

**Features**:
- Pure Python implementation
- Full PyVIPS feature set
- Easy to extend and customize

Annotation Handling
-------------------

Loading Annotations
~~~~~~~~~~~~~~~~~~~

GeoJSON Format
^^^^^^^^^^^^^^

.. code-block:: python

   # Load GeoJSON annotations
   annotations = dlup.annotations.AnnotationSet.from_geojson(
       'annotations.json',
       coordinate_system='level0'  # or 'pixel'
   )

   # Access annotation properties
   for annotation in annotations:
       print(f"Type: {annotation.type}")
       print(f"Properties: {annotation.properties}")
       print(f"Geometry: {annotation.geometry}")

HALO XML Format
^^^^^^^^^^^^^^^

.. code-block:: python

   # Load HALO XML annotations
   annotations = dlup.annotations.AnnotationSet.from_halo_xml(
       'annotations.xml'
   )

ASAP XML Format
^^^^^^^^^^^^^^^

.. code-block:: python

   # Load ASAP XML annotations
   annotations = dlup.annotations.AnnotationSet.from_asap_xml(
       'annotations.xml'
   )

V7 Darwin Format
^^^^^^^^^^^^^^^^

.. code-block:: python

   # Load V7 Darwin JSON annotations
   annotations = dlup.annotations.AnnotationSet.from_darwin_json(
       'annotations.json'
   )

Annotation Processing
~~~~~~~~~~~~~~~~~~~~

Filtering and Querying
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Filter by annotation type
   nuclei = annotations.filter_by_type('nuclei')
   glands = annotations.filter_by_type('gland')

   # Filter by properties
   large_annotations = annotations.filter_by_property(
       'area', lambda x: x > 1000
   )

   # Spatial queries
   region = dlup.geometry.Region(x=0, y=0, width=1000, height=1000)
   annotations_in_region = annotations.within_region(region)

Creating Masks
^^^^^^^^^^^^^^

.. code-block:: python

   # Create binary mask from annotations
   mask = annotations.to_mask(
       size=(2048, 2048),
       annotation_types=['nuclei', 'glands']
   )

   # Create instance mask (each annotation gets unique ID)
   instance_mask = annotations.to_instance_mask(
       size=(2048, 2048),
       background_value=0
   )

Annotation Export
^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Export to different formats
   annotations.to_geojson('output.json')
   annotations.to_halo_xml('output.xml')

Advanced Tiling Strategies
--------------------------

Custom Tiling Patterns
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Create custom tiling dataset
   dataset = dlup.tiling.SlideImageTilingDataset(
       slide_path='slide.svs',
       tile_size=(256, 256),
       tile_overlap=(64, 64),  # Overlapping tiles
       resolution=0.5,
       # Custom coordinate generator
       coordinate_generator=dlup.tiling.GridCoordinateGenerator(
           grid_size=(256, 256),
           overlap=(64, 64)
       )
   )

Multi-Resolution Tiling
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Create dataset with multiple resolutions
   dataset = dlup.tiling.MultiResolutionTilingDataset(
       slide_path='slide.svs',
       tile_size=(256, 256),
       resolutions=[0.25, 0.5, 1.0, 2.0],
       max_level=2
   )

Annotation-Aware Tiling
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Create dataset that respects annotation boundaries
   dataset = dlup.tiling.AnnotationAwareTilingDataset(
       slide_path='slide.svs',
       annotations=annotations,
       tile_size=(256, 256),
       min_annotation_coverage=0.1  # At least 10% annotation coverage
   )

Background Processing
---------------------

Background Estimation
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Create background estimator
   estimator = dlup.background.BackgroundEstimator(
       method='otsu',  # or 'adaptive', 'manual'
       **kwargs
   )

   # Estimate background
   background_mask = estimator.estimate(slide)

   # Apply threshold
   foreground_mask = ~background_mask

Advanced Background Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Adaptive thresholding
   estimator = dlup.background.BackgroundEstimator(
       method='adaptive',
       block_size=35,
       offset=10
   )

   # Manual thresholding
   estimator = dlup.background.BackgroundEstimator(
       method='manual',
       threshold=0.1
   )

Custom Background Algorithms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class CustomBackgroundEstimator(dlup.background.BackgroundEstimator):
       def estimate(self, slide):
           # Your custom algorithm
           # Return binary mask (True = background)
           return background_mask

Performance Optimization
------------------------

Memory Management
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Monitor memory usage
   import psutil
   import os

   process = psutil.Process(os.getpid())
   memory_info = process.memory_info()
   print(f"Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")

   # Use memory-efficient reading
   with dlup.SlideImage.from_file_path('slide.svs') as slide:
       # Process in chunks
       chunk_size = 1000
       for y in range(0, slide.dimensions[1], chunk_size):
           for x in range(0, slide.dimensions[0], chunk_size):
               region = slide.read_region(
                   (x, y),
                   (min(chunk_size, slide.dimensions[0] - x),
                    min(chunk_size, slide.dimensions[1] - y)),
                   0.5
               )
               # Process chunk

Parallel Processing
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from concurrent.futures import ThreadPoolExecutor
   import threading

   def process_slide(slide_path):
       with dlup.SlideImage.from_file_path(slide_path) as slide:
           # Thread-safe processing
           return process_slide_data(slide)

   # Process multiple slides in parallel
   slide_paths = ['slide1.svs', 'slide2.svs', 'slide3.svs']

   with ThreadPoolExecutor(max_workers=4) as executor:
       results = list(executor.map(process_slide, slide_paths))

Caching Strategies
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Global cache (shared across all readers)
   global_cache = dlup.RuntimeGlobalCacheManager.instance()
   global_cache.set_capacity(5000)

   # Per-slide cache
   slide_cache = dlup.CacheManager.create(
       capacity=1000,
       strategy='lru'  # or 'lfu', 'random'
   )

   # Monitor cache performance
   stats = global_cache.get_basic_stats()
   print(f"Global cache hit ratio: {stats.hit_ratio:.2%}")

Custom Backend Implementation
-----------------------------

Creating a Custom Backend
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class CustomBackend(dlup.backends.Backend):
       def __init__(self, path):
           self.path = path
           # Initialize your backend

       def read_region(self, location, size, resolution):
           # Implement region reading
           # Return numpy array
           pass

       def get_dimensions(self, resolution):
           # Return (width, height) at given resolution
           pass

       def get_spacings(self):
           # Return list of available spacings
           pass

   # Register the backend
   dlup.backends.register_backend('custom', CustomBackend)

Error Handling and Debugging
-----------------------------

Exception Handling
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   try:
       slide = dlup.SlideImage.from_file_path('slide.svs')
       region = slide.read_region((0, 0), (256, 256), 0.5)
   except dlup.DlupError as e:
       print(f"DLUP Error: {e}")
   except FileNotFoundError:
       print("Slide file not found")
   except Exception as e:
       print(f"Unexpected error: {e}")

Debug Information
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Enable debug logging
   import logging
   logging.basicConfig(level=logging.DEBUG)

   # Get detailed slide information
   slide.print_info()

   # Check backend capabilities
   print(f"Backend: {slide.backend_name}")
   print(f"Available backends: {dlup.backends.get_available_backends()}")

Integration with Other Libraries
---------------------------------

PyTorch Integration
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   from torch.utils.data import Dataset

   class CustomSlideDataset(Dataset):
       def __init__(self, slide_path, transform=None):
           self.slide_path = slide_path
           self.transform = transform

       def __len__(self):
           # Return number of tiles
           return 1000

       def __getitem__(self, idx):
           with dlup.SlideImage.from_file_path(self.slide_path) as slide:
               # Generate tile coordinates
               x, y = self.get_tile_coordinates(idx)

               # Read tile
               tile = slide.read_region(
                   location=(x, y),
                   size=(256, 256),
                   resolution=0.5
               )

               if self.transform:
                   tile = self.transform(tile)

               return torch.from_numpy(tile).permute(2, 0, 1)

TensorFlow Integration
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import tensorflow as tf

   def create_tf_dataset(slide_path, batch_size=32):
       # Create Python generator
       def slide_generator():
           with dlup.SlideImage.from_file_path(slide_path) as slide:
               # Generate tiles
               for x, y in tile_coordinates:
                   tile = slide.read_region((x, y), (256, 256), 0.5)
                   yield tile

       # Convert to TF dataset
       dataset = tf.data.Dataset.from_generator(
           slide_generator,
           output_signature=tf.TensorSpec(shape=(256, 256, 3), dtype=tf.uint8)
       )

       return dataset.batch(batch_size)

Best Practices
--------------

General Guidelines
~~~~~~~~~~~~~~~~~~

1. **Use context managers** for proper resource cleanup
2. **Enable caching** for overlapping reads
3. **Choose appropriate backends** for your use case
4. **Handle exceptions** gracefully
5. **Monitor memory usage** for large slides

Performance Tips
~~~~~~~~~~~~~~~~

1. **Batch operations** when possible
2. **Use appropriate tile sizes** for your model
3. **Enable parallel processing** for multiple slides
4. **Monitor cache hit ratios** for optimization
5. **Use memory-efficient data types** when possible

Memory Management
~~~~~~~~~~~~~~~~~

1. **Process slides sequentially** for very large datasets
2. **Use smaller tile sizes** to reduce memory usage
3. **Implement proper cleanup** in long-running processes
4. **Monitor memory usage** and implement limits if needed

Troubleshooting Guide
--------------------

Common Issues and Solutions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Import Errors**
   - Ensure all dependencies are installed
   - Check Python version compatibility
   - Verify backend libraries are available

**Memory Issues**
   - Reduce batch sizes
   - Use smaller tile sizes
   - Process slides sequentially
   - Enable garbage collection

**Performance Issues**
   - Enable caching
   - Use appropriate backends
   - Optimize tile sizes
   - Use parallel processing judiciously

**Annotation Issues**
   - Verify annotation format compatibility
   - Check coordinate system consistency
   - Validate annotation geometry

Getting Help
------------

- **Documentation**: This user guide and API reference
- **Examples**: Check the examples directory
- **GitHub**: Report issues and ask questions
- **Community**: Join discussions for community support

For more detailed information, see:

- :doc:`quickstart` - Quick start guide
- :doc:`../api/index` - Complete API reference
- :doc:`../examples/index` - Practical examples
