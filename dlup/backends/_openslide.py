# coding=utf-8
# Copyright (c) dlup contributors
import os

import numpy as np

import openslide
from dlup import UnsupportedSlideError
from dlup.backends.common import AbstractSlideBackend, check_mpp


def open_slide(filename: os.PathLike) -> "OpenSlideSlide":
    return OpenSlideSlide(filename)


class OpenSlideSlide(openslide.OpenSlide, AbstractSlideBackend):
    def __init__(self, filename: os.PathLike):
        super().__init__(str(filename))

        try:
            mpp_x = float(self.properties[openslide.PROPERTY_NAME_MPP_X])
            mpp_y = float(self.properties[openslide.PROPERTY_NAME_MPP_Y])

        except KeyError:
            raise UnsupportedSlideError(f"slide property mpp is not available.", str(filename))

        check_mpp(mpp_x, mpp_y)
        mpp = np.array([mpp_y, mpp_x])
        self._spacings = tuple([tuple(mpp * downsample) for downsample in self.level_downsamples])


if __name__ == "__main__":
    path = "/mnt/archive/data/pathology/TCGA/images/gdc_manifest.2021-11-01_tissue_uterus_nos.txt/076fde5e-e081-4316-8279-3de6108dd021/TCGA-N6-A4VD-01A-01-TSA.E009A1D0-3BB5-45A4-BC4A-9C505F71AA22.svs"
    # slide = OpenSlideSlide(path)

    from dlup import SlideImage
    from dlup.backends import ImageBackends

    # Try to autodetect the backend
    slide_autodetect = SlideImage.from_file_path(path, backend=ImageBackends.AUTODETECT)

    print("hi")
