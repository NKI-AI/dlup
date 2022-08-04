# coding=utf-8
# Copyright (c) dlup contributors
import tempfile

import numpy as np
import PIL.Image
import pytest
import pyvips

from dlup import SlideImage
from dlup.data.dataset import TiledROIsSlideImageDataset, TilingMode
from dlup.experimental_backends import ImageBackends
from dlup.tiling import GridOrder, TilingMode
from dlup.utils.pyvips import vips_to_numpy
from dlup.writers import TiffCompression, TifffileImageWriter


def rewrite_image():
    path = "/mnt/archive/data/pathology/TIGER/wsitils/images/104S.tif"
    mask = SlideImage.from_file_path(path, backend=ImageBackends.TIFFFILE)
    #   Image Width: 55552 Image Length: 36352, need the underlying array size!!!
    writer = TifffileImageWriter(
        filename="/home/j.teuwen/104S_tissue_jonas_pyramid.tif",
        size=(*mask.size[::-1], 3),
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
        # crop=False,
    )

    def iterator():
        for sample in ds:
            image = np.asarray(sample["image"])
            yield image

    writer.from_tiles_iterator(iterator())


if __name__ == "__main__":
    rewrite_image()
    # rewrite_image()

#
# def image_benchmark():
#     import time
#     from pathlib import Path
#
#     from dlup import SlideImage
#     from dlup.data.dataset import TiledROIsSlideImageDataset, TilingMode
#     from dlup.experimental_backends import ImageBackends
#     from dlup.tiling import GridOrder
#
#     base_path = Path("/processing/j.teuwen/")
#     image = "TCGA-A1-A0SK-01Z-00-DX1.A44D70FA-4D96-43F4-9DD7-A61535786297.tif"
#     mask = "TCGA-A1-A0SK-01Z-00-DX1.A44D70FA-4D96-43F4-9DD7-A61535786297_tissue.tif"
#     path0 = (base_path / image, True)
#     path1 = (base_path / "annotations" / image, False)
#     path2 = (base_path / "tissue" / mask, False)
#
#     compressions = [
#         TiffCompression.JPEG,
#         TiffCompression.JP2K,
#         TiffCompression.WEBP,
#         TiffCompression.ZSTD,
#         TiffCompression.PACKBITS,
#         TiffCompression.DEFLATE,
#         TiffCompression.CCITTFAX4,
#         TiffCompression.LZW,
#         TiffCompression.JP2K_LOSSY,
#         TiffCompression.PNG,
#     ]
#
#     for idx, (path, is_rgb) in enumerate((path0, path1, path2)):
#         image_object = SlideImage.from_file_path(path, backend=ImageBackends.TIFFFILE)
#         #   Image Width: 55552 Image Length: 36352, need the underlying array size!!!
#         for quality in (90, 100):
#             if not is_rgb and quality == 90:
#                 continue
#
#             size = (*image_object.size[::-1], 3) if is_rgb else image_object.size[::-1]
#
#             for compression in compressions:
#                 start_time = time.time()
#                 print(f"Working on {compression.value} for {path}")
#                 writer = TifffileImageWriter(
#                     size=size,
#                     tile_width=512,
#                     tile_height=512,
#                     mpp=image_object.mpp,
#                     compression=compression,
#                     pyramid=True,
#                     quality=quality,
#                 )
#
#                 ds = TiledROIsSlideImageDataset.from_standard_tiling(
#                     path,
#                     image_object.mpp,
#                     grid_order=GridOrder.C,
#                     tile_size=(512, 512),
#                     tile_overlap=(0, 0),
#                     tile_mode=TilingMode.overflow,
#                     backend=ImageBackends.TIFFFILE,
#                     # crop=False,
#                 )
#                 add = ""
#                 if idx == 0:
#                     add = "image"
#                 if idx == 1:
#                     add = "annotations"
#                 if idx == 2:
#                     add = "tissue"
#                 output_fn = f"/processing/j.teuwen/output/TCGA_{compression.value}_{quality}_{add}.tif"
#
#                 if Path(output_fn).exists():
#                     continue
#
#                 def iterator():
#                     for sample in ds:
#                         image = np.asarray(sample["image"])
#                         yield image
#
#                 try:
#                     writer.from_iterator(iterator(), output_fn)
#                 except Exception as e:
#                     print(f"Failed with {e}.")
#
#                 print(f"Total time to write {output_fn}: {time.time() - start_time}.")
#
#     # print()
