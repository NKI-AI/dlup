import abc
import functools
import io
import json
import math
from abc import abstractmethod
from functools import partial
from typing import Any, Callable

import PIL.Image

import dlup.utils.imports
from dlup.types import PathLike

if dlup.utils.imports.BOTO3_AVAILABLE:
    import boto3
    import botocore.exceptions

from dlup.backends.common import AbstractSlideBackend
from dlup.data.dataset import TiledWsiDataset
from dlup.tiling import TilingMode

METADATA_CACHE = 128


class RemoteSlideBackend(AbstractSlideBackend):
    def __init__(self, filename: PathLike) -> None:
        super().__init__(filename)
        self._metadata = None
        self._stride: tuple[int, int] | None
        self._tile_size: tuple[int, int]

        self._total_rows: tuple[int] | None
        self._total_cols: tuple[int] | None

    @abstractmethod
    def _fetch_metadata(self) -> dict[str, Any]:
        pass

    @property
    def metadata(self) -> dict[str, Any]:
        if not hasattr(self, "_remote_metadata"):
            self._remote_metadata = self._fetch_metadata()
        return self._remote_metadata

    @abc.abstractmethod
    def read_tile_from_remote(self, level: int, row: int, col: int) -> PIL.Image.Image:
        pass

    def read_region(
        self, location: tuple[int | float, int | float], level: int, size: tuple[int, int]
    ) -> PIL.Image.Image:
        # TODO: This needs to be fixed
        location = (int(location[0]), int(location[1]))

        if not self._stride or not self._total_rows or not self._total_cols:
            raise ValueError("The stride and total rows and columns must be set before reading a region")

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

    def close(self) -> None:
        pass


class V7RemoteTiledImage(RemoteSlideBackend):
    METADATA_CACHE = 128  # Assuming a constant for cache size

    def __init__(self, s3_client, bucket: str, bucket_prefix: str) -> None:
        # filename is required by the superclass but not used in this context.
        super().__init__(filename="")
        self._s3_client = s3_client
        self._bucket = bucket
        self._bucket_prefix = bucket_prefix

        self._set_metadata()
        self._stride = (self._tile_width, self._tile_height)
        self._tile_size = (self._tile_width, self._tile_height)
        self._tile_overlap = (0, 0)
        self._total_rows = math.ceil((self._size[1] - self._tile_overlap[1]) / self._stride[1])
        self._total_cols = math.ceil((self._size[0] - self._tile_overlap[0]) / self._stride[0])
        self._num_channels = 3

    def _set_metadata(self) -> None:
        self._tile_width = self.metadata["levels"]["0"]["tile_width"]
        self._tile_height = self.metadata["levels"]["0"]["tile_height"]
        self._size = self.metadata["full_size"]
        self._mpp = tuple(self.metadata["mpp"])
        self._downsamples = self.metadata["downscale_ratios"]
        self._shapes = [(int(self._size[0] / ds), int(self._size[1] / ds)) for ds in self._downsamples]
        self._spacings = [(self._mpp[0] * ds, self._mpp[1] * ds) for ds in self._downsamples]

    @functools.lru_cache(maxsize=METADATA_CACHE)
    def _fetch_metadata(self) -> dict[str, Any]:
        file_stream = io.BytesIO()
        try:
            self._s3_client.download_fileobj(self._bucket, f"{self._bucket_prefix}metadata.json", file_stream)
            file_stream.seek(0)
            metadata: dict[str, Any] = json.load(file_stream)
            return metadata

        except botocore.exceptions.ClientError as e:
            raise ValueError(f"Could not get metadata: {e}")

    def read_tile_from_remote(self, level: int, row: int, col: int) -> PIL.Image.Image:
        if level < 0 or level >= self.num_levels:
            raise ValueError(f"Level {level} is out of bounds. The number of levels is {self.num_levels}")

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

    @property
    def properties(self) -> dict[str, Any]:
        """Properties of slide"""
        return {}

    @property
    def magnification(self) -> float | None:
        return None

    @property
    def vendor(self) -> str | None:
        """Returns the scanner vendor."""
        return "Darwin V7"

    @property
    def num_levels(self) -> int:
        return len(self._downsamples)


def create_dataset(s3_client, bucket, bucket_prefix, region_of_interest, tile_size, tile_overlap, mpp):
    backend = partial(V7RemoteTiledImage, s3_client, bucket)
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
    aws_access_key_id = "<snip>"
    aws_secret_access_key = "<snip>"
    endpoint_url = "<snip>"
    region_name = "<snip>"
    bucket_prefix = "<snip>"
    bucket = "<snip>"

    region_of_interest = (28000, 100000), (5000, 5000)
    tile_size = (256, 256)
    tile_overlap = (0, 0)
    mpp = 0.5

    if not dlup.utils.imports.BOTO3_AVAILABLE:
        raise ImportError("The boto3 package is required to use the V7RemoteTiledImage backend.")

    s3_client = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        region_name=region_name,
        use_ssl=True,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )

    remote_image = V7RemoteTiledImage(s3_client, bucket, bucket_prefix)

    region = remote_image.read_region(region_of_interest[0], 0, region_of_interest[1])
    region.save("region.jpg", Q=85)

    pass
