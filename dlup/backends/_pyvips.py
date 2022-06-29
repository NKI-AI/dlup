# coding=utf-8
# Copyright (c) dlup contributors
import numpy as np
import PIL.Image
import pyvips

from dlup.backends.common import AbstractSlideBackend, check_mpp


def open_slide(filename):
    return PyVipsSlide(filename)


class PyVipsSlide(AbstractSlideBackend):
    def __init__(self, path: str) -> None:
        super().__init__(path)
        # You can have pyvips figure out the reader
        self._images = []
        self._images.append(pyvips.Image.new_from_file(str(path)))

        loader = self._images[0].get("vips-loader")

        if loader == "tiffload":
            self._read_as_tiff(path)
        elif loader == "openslideload":
            self._read_as_openslide(path)
        else:
            raise NotImplementedError(f"Loader {loader} is not implemented.")

        self._regions = [pyvips.Region.new(image) for image in self._images]

    def _read_as_tiff(self, path):
        self._level_count = int(self._images[0].get_value("n-pages"))
        for level in range(1, self._level_count):
            self._images.append(pyvips.Image.tiffload(str(path), page=level))

        # Each tiff page has a resolution
        unit_dict = {"cm": 10000, "centimeter": 10000}
        self._downsamples.append(1.0)
        for idx, image in enumerate(self._images):
            mpp_x = unit_dict[image.get("resolution-unit")] / float(image.get("xres"))
            mpp_y = unit_dict[image.get("resolution-unit")] / float(image.get("yres"))
            check_mpp(mpp_x, mpp_y)

            self._spacings.append(mpp_x)
            if idx >= 1:
                downsample = mpp_x / self._spacings[0]
                self._downsamples.append(downsample)
            self._shapes.append((image.get("width"), image.get("height")))

    def _read_as_openslide(self, path):
        self._level_count = int(self._images[0].get("openslide.level-count"))
        for level in range(1, self._level_count):
            self._images.append(pyvips.Image.openslideload(str(path), level=level))

        for idx, image in enumerate(self._images):
            self._shapes.append(
                (int(image.get(f"openslide.level[{idx}].width")), int(image.get(f"openslide.level[{idx}].height")))
            )

        for idx, image in enumerate(self._images):
            self._downsamples.append(float(image.get(f"openslide.level[{idx}].downsample")))

        mpp_x = float(self._images[0].get("openslide.mpp-x"))
        mpp_y = float(self._images[0].get("openslide.mpp-y"))
        self.__check_mpp(mpp_x, mpp_y)
        self._mpps = [mpp_x * downsample for downsample in self._downsamples]

    @property
    def properties(self):
        """Metadata about the image.
        This is a map: property name -> property value."""
        keys = self._images[0].get_fields()
        return {key: self._images[0].get_value(key) for key in keys}

    @property
    def associated_images(self):
        """Images associated with this whole-slide image.
        This is a map: image name -> PIL.Image."""
        associated_images = (_.strip() for _ in self.properties["slide-associated-images"].split(","))

        raise NotImplementedError

    def set_cache(self, cache):
        """Use the specified cache to store recently decoded slide tiles.
        cache: an OpenSlideCache object."""
        raise NotImplementedError

    def read_region(self, coordinates, level, size):
        image = self._regions[level]
        ratio = self._downsamples[level]
        x, y = coordinates
        height, width = size
        vreg = PIL.Image.fromarray(
            np.asarray(image.fetch(int(x // ratio), int(y // ratio), int(height), int(width))).reshape(
                int(width), int(height), -1
            )
        )

        return vreg

    def close(self):
        del self._regions
        del self._images
