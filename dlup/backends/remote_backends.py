import functools
import io
import json
import math
import os
import tempfile
from abc import abstractmethod
from functools import partial
from pathlib import Path

import boto3
import numpy as np
import openslide
import PIL.Image

from dlup.data.dataset import TiledWsiDataset, TilingMode

from .common import AbstractSlideBackend

METADATA_CACHE = 128


class RemoteSlideBackend(AbstractSlideBackend):
    def __init__(self):
        pass

    @abstractmethod
    def _fetch_metadata(self):
        """Fetch metadata from the remote backend"""

    @abstractmethod
    def read_tile_from_remote(self, level: int, row: int, col: int):
        """Read a tile from the remote backend"""


class TiledSlideImage(RemoteSlideBackend):
    def __init__(self, s3_client: boto3.client, bucket: str, bucket_prefix: str):
        self._s3_client = s3_client
        self._bucket_prefix = bucket_prefix

        self._bucket = bucket

        tile_width = self.metadata["levels"]["0"]["tile_width"]
        tile_height = self.metadata["levels"]["0"]["tile_height"]

        self._stride = (tile_width, tile_height)
        self._tile_size = (tile_width, tile_height)
        self._tile_overlap = (0, 0)

        self._size = self.metadata["full_size"]
        self._mpp = tuple(self.metadata["mpp"])
        self._downsamples = self.metadata["downscale_ratios"]
        self._shapes = [(int(self._size[0] / ds), int(self._size[1] / ds)) for ds in self._downsamples]

        self._total_rows = math.ceil((self._size[1] - self._tile_overlap[1]) / self._stride[1])
        self._total_cols = math.ceil((self._size[0] - self._tile_overlap[0]) / self._stride[0])

        self._num_channels = 3

        self._num_levels = None

    @property
    def level_dimensions(self) -> list[tuple[int, int]]:
        """A list of (width, height) tuples, one for each level of the image.
        This property level_dimensions[n] contains the dimensions of the image at level n.

        Returns
        -------
        list

        """
        return self._shapes

    @property
    def dimensions(self):
        return self._size

    @property
    def spacing(self):
        return self._mpp

    @property
    def num_levels(self):
        if self._num_levels:
            return self._num_levels

        scale_factor = 2
        current_size = self._size
        levels = 1

        # Continue until the current size is smaller than the minimum size in either dimension.
        while current_size[0] >= self._tile_size[0] and current_size[1] >= self._tile_size[1]:
            current_size = (current_size[0] / scale_factor, current_size[1] / scale_factor)
            levels += 1
        self._num_levels = levels

        return levels

    @functools.lru_cache(maxsize=METADATA_CACHE)
    def _fetch_metadata(self, bucket: str, bucket_prefix: str) -> dict[str, any]:
        file_stream = io.BytesIO()
        self._s3_client.download_fileobj(bucket, f"{bucket_prefix}metadata.json", file_stream)
        file_stream.seek(0)
        metadata = json.load(file_stream)
        return metadata

    @property
    def metadata(self):
        if not hasattr(self, "_metadata"):
            self._metadata = self._fetch_metadata(self._bucket, self._bucket_prefix)
        return self._metadata

    def read_tile_from_remote(self, level: int, row: int, col: int):
        try:
            file_stream = io.BytesIO()

            self._s3_client.download_fileobj(
                self._bucket, f"{self._bucket_prefix}tiles/{level}/{row}_{col}.jpg", file_stream
            )
            file_stream.seek(0)
            img = PIL.Image.open(file_stream)
            return img
        except:
            # TODO: The out of bounds read should return an empty tile
            image = PIL.Image.new("RGB", self._tile_size, (255, 255, 255))
            return image

    def read_region(self, location: tuple[int, int], level: int, size: tuple[int, int]):
        downsample = 2**level
        x, y = (location[0] // downsample, location[1] // downsample)
        w, h = size

        start_row = y // self._stride[1]
        end_row = min((y + h - 1) // self._stride[1] + 1, math.ceil(self._total_rows / downsample))
        start_col = x // self._stride[0]
        end_col = min((x + w - 1) // self._stride[0] + 1, math.ceil(self._total_cols / downsample))

        stitched_image = PIL.Image.new("RGB", (w, h))
        for i in range(start_row, end_row):
            for j in range(start_col, end_col):
                tile = self.read_tile_from_remote(level, i, j)
                start_y = i * self._stride[1] - y
                end_y = start_y + self._tile_size[1]
                start_x = j * self._stride[0] - x
                end_x = start_x + self._tile_size[0]

                img_start_y = max(0, start_y)
                img_end_y = min(h, end_y)
                img_start_x = max(0, start_x)
                img_end_x = min(w, end_x)

                crop_start_y = img_start_y - start_y
                crop_end_y = img_end_y - start_y
                crop_start_x = img_start_x - start_x
                crop_end_x = img_end_x - start_x

                cropped_tile = tile.crop((crop_start_x, crop_start_y, crop_end_x, crop_end_y))
                stitched_image.paste(cropped_tile, (img_start_x, img_start_y))

        return stitched_image

    def close(self):
        pass

    @property
    def level_downsamples(self) -> tuple[float, ...]:
        """A tuple of downsampling factors for each level of the image.
        level_downsample[n] contains the downsample factor of level n."""
        return tuple(self._downsamples)

    def get_best_level_for_downsample(self, downsample: float) -> int:
        """
        Compute the best level for displaying the given downsample. Returns the closest better resolution.

        Parameters
        ----------
        downsample : float

        Returns
        -------
        int
        """
        level_downsamples = np.array(self.level_downsamples)
        level = 0 if downsample < 1 else int(np.where(level_downsamples <= downsample)[0][-1])
        return level


def create_dataset(s3_client, bucket, bucket_prefix, region_of_interest, tile_size, tile_overlap, mpp):
    backend = partial(TiledSlideImage, s3_client, bucket)
    # Now it can be loaded using bucket_prefix.

    dataset = TiledWsiDataset.from_standard_tiling(
        path=bucket_prefix,
        backend=backend,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        mpp=mpp,
        tile_mode=TilingMode.overflow,
        rois=(region_of_interest,),
    )

    return dataset


if __name__ == "__main__":
    aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID", None)
    aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY", None)

    s3_client = boto3.client(
        "s3",
        endpoint_url="https://minio.test.ai",
        region_name="nl-ams",
        use_ssl=True,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )

    bucket = "dlu_test"
    region_of_interest = (28000, 100000), (5000, 5000)
    tile_size = (256, 256)
    tile_overlap = (0, 0)
    mpp = 0.5
