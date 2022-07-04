Backends
========

Dlup supports several backends to read whole-slide images. Currently available are:

* `openslide`_
* `pyvips`_
* `tifffile`_

By default *SlideImage*, which represents a single whole slide image will read the image using `openslide`_.
Remember that you can instantiate it by using the path to the WSI file with:

.. code-block:: python

    import dlup
    from dlup.experimental_backends import ImageBackends
    wsi = dlup.SlideImage.from_file_path("checkerboard.svs", backend=ImageBackends.OPENSLIDE)

The other options are `ImageBackends.PYVIPS`, `ImageBackends.TIFFFILE` and `ImageBackends.AUTODETECT`.

In case `AUTODETECT` is selected and in case the image ends with `.tif` or `.tiff` dlup will try to read a region with
`pyvips` and next with `tifffile` in case the previous fails. If not autodetect will continue by trying `openslide`
and `pyvips`. The results of this computation are cached, so next time one tries to read the same WSI file, the same
backend will be selected.

.. warning::
    There are minor differences in how `openslide`_ and `pyvips`_ read the same files, and there might me minor pixel
    misalignments between the two at different levels. In addition, `openslide`_ will always return an RGBA image,
    regardless whether the original image has an alpha channel or not. `pyvips`_ on the other hand will output the
    data type of the underlying WSI. This can lead to differences in the number of channels depending on the backend.

.. _openslide: https://openslide.org/api/python/
.. _pyvips: https://libvips.github.io/pyvips/
.. _tifffile: https://github.com/cgohlke/tifffile
