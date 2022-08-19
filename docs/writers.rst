
Writers
=======

dlup provides utilities to write pyramidal tiff images and masks using `tifffile`_. Several compressions are supported.


Usage
-----
Example code

.. code-block:: python

    from dlup import SlideImage
    from dlup.data.dataset import TiledROIsSlideImageDataset
    from dlup.tiling import GridOrder, TilingMode
    from dlup.writers import TifffileImageWriter, TiffCompression
    from dlup.experimental_backends import ImageBackends


    path = "image.tif"
    output_file = "output.tif"
    mask = SlideImage.from_file_path(path, backend=ImageBackends.TIFFFILE)
    writer = TifffileImageWriter(
        filename=output_file,
        size=(*mask.size, 3),
        tile_size=(512, 512),
        mpp=mask.mpp,
        compression=TiffCompression.JPEG,
        pyramid=True,
    )

    ds = TiledROIsSlideImageDataset.from_standard_tiling(
        path,
        mask.mpp,
        grid_order=GridOrder.C,
        tile_size=(512, 512),
        tile_overlap=(0, 0),
        tile_mode=TilingMode.overflow,
        backend=ImageBackends.TIFFFILE,
    )

    def iterator():
        for sample in ds:
            image = np.asarray(sample["image"])
            yield image

    writer.from_tiles_iterator(iterator())


Benchmarks
----------

This benchmark determines the compression time and file size of the following file:
`TCGA-A1-A0SK-01Z-00-DX1.A44D70FA-4D96-43F4-9DD7-A61535786297.tif` into a pyramidal tiff.

This file is provided by the `TIGER challenge`_.


.. list-table:: Lossless image
   :widths: 50 50 50 100
   :header-rows: 1

   * - Compressor
     - Time
     - Size
     - Comment
   * - JPEG
     - 174s
     - 1.5G
     -
   * - JP2K
     - 851s
     - 1.5G
     -
   * - WEBP
     - 248s
     - 933M
     -
   * - ZSTD
     - 296s
     - 3.8G
     -
   * - PACKBITS
     - 166s
     - 6.9G
     -
   * - DEFLATE
     - 248s
     - 4.0G
     -
   * - CCITT_T4
     - N/A
     - N/A
     - Not supported
   * - LZW
     - N/A
     - N/A
     - Not implemented
   * - JP2K_LOSSY
     - 848s
     - 1.5G
     - Equivalent to JP2K when quality = 100
   * - PNG
     - 593s
     - 3.3G
     -


.. list-table:: Lossy image (quality 90%)
   :widths: 50 50 50 100
   :header-rows: 1

   * - Compressor
     - Time
     - Size
     - Comment
   * - JPEG
     - 167s
     - 653M
     -
   * - JP2K
     - N/A
     - N/A
     - Not lossy
   * - WEBP
     - 216s
     - 554M
     -
   * - ZSTD
     - N/A
     - N/A
     - Not lossy
   * - PACKBITS
     - N/A
     - N/A
     - Not lossy
   * - DEFLATE
     - N/A
     - N/A
     - Not lossy
   * - CCITT_T4
     - N/A
     - N/A
     - Not supported
   * - LZW
     - N/A
     - N/A
     - Not implemented
   * - JP2K_LOSSY
     - 857s
     - 1.5G
     - Seemingly does not work
   * - PNG
     - 602s
     - 3.3G
     - Seemingly does not work


.. list-table:: Annotations
   :widths: 50 50 50 100
   :header-rows: 1

   * - Compressor
     - Time
     - Size
     - Comment
   * - JPEG
     - 45s
     - 40M
     -
   * - JP2K
     - 118s
     - 3.6M
     -
   * - WEBP
     - N/A
     - N/A
     - Not supported for masks
   * - ZSTD
     - 45s
     - 495k
     -
   * - PACKBITS
     - 100s
     - 48M
     -
   * - DEFLATE
     - 48s
     - 3.5M
     -
   * - CCITT_T4
     - N/A
     - N/A
     - Not supported
   * - LZW
     - N/A
     - N/A
     - Not implemented
   * - JP2K_LOSSY
     - 116
     - 3.6M
     -
   * - PNG
     - 61s
     - 4.2M
     -


.. list-table:: Binary mask
   :widths: 50 50 50 100
   :header-rows: 1

   * - Compressor
     - Time
     - Size
     - Comment
   * - JPEG
     - 46s
     - 41M
     -
   * - JP2K
     - 120s
     - 4.2M
     -
   * - WEBP
     - N/A
     - N/A
     - Not supported for masks
   * - ZSTD
     - 70s
     - 1.3M
     -
   * - PACKBITS
     - 96s
     - 49M
     -
   * - DEFLATE
     - 54s
     - 4.4M
     -
   * - CCITT_T4
     - N/A
     - N/A
     - Not supported
   * - LZW
     - N/A
     - N/A
     - Not implemented
   * - JP2K_LOSSY
     - 119s
     - 4.2M
     -
   * - PNG
     - 66s
     - 8.9M
     -

.. _tifffile: https://github.com/cgohlke/tifffile
.. _TIGER challenge: https://tiger.grand-challenge.org
