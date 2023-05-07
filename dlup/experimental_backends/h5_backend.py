# encoding: utf-8
import dlup
from dlup.data.dataset import TiledROIsSlideImageDataset
from dlup.writers import TifffileImageWriter, Resampling
from dlup.tiling import TilingMode
from pathlib import Path
import json
import PIL.Image
from enum import Enum

import h5py
import numpy as np
import PIL.Image
import math


class StitchingMode(Enum):
    CROP = 0
    AVERAGE = 1
    MAXIMUM = 2


class _DatasetIterator:
    def __init__(self, dataset):
        self.dataset = dataset
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index < len(self.dataset):
            item = self.dataset[self.index]
            self.index += 1
            return item
        else:
            raise StopIteration


class H5FileImageWriter:
    """Image writer that writes tile-by-tile to tiff."""

    def __init__(
        self,
        filename,
        size,
        mpp,
        tile_size,
        tile_overlap,
        progress=None,
    ):
        self._filename = filename
        self._size = size
        self._mpp = mpp
        self._tile_size = tile_size
        self._tile_overlap = tile_overlap
        self._progress = progress

    def from_iterator(self, iterator):
        _iterator = _DatasetIterator(iterator)
        with h5py.File(self._filename, "w") as h5file:
            # Determine the shape of the images in the dataset
            first_item = next(_iterator)
            sample_shape = np.asarray(first_item["image"]).shape
            num_samples = len(ds)

            # Create an empty dataset in the h5 file to store all images
            image_dtype = np.asarray(first_item["image"]).dtype
            image_dataset = h5file.create_dataset(
                "data",
                shape=(num_samples,) + sample_shape,
                dtype=image_dtype,
                compression="gzip",
                chunks=(1,) + sample_shape,
            )

            # Add the first image to the dataset
            image_dataset[0] = first_item["image"]

            # Optionally wrap the iterator with a progress bar
            if self._progress is not None:
                _iterator = self._progress(_iterator, total=num_samples)
                _iterator.update(1)  # Update the progress bar to start at 1

            # Iterate over the remaining images and add them to the dataset
            for i, item in enumerate(_iterator, start=1):
                image_dataset[i] = np.asarray(item["image"])

            # Write metadata as JSON
            metadata = {
                "mpp": self._mpp,
                "dtype": str(image_dtype),
                "sample_shape": sample_shape,
                "num_samples": num_samples,
                "tile_size": self._tile_size,
                "tile_overlap": self._tile_overlap,
            }
            metadata_json = json.dumps(metadata)
            h5file.attrs["metadata"] = metadata_json


class H5FileImageReader:
    def __init__(self, filename, size, tile_size, tile_overlap, stitching_mode):
        self._filename = filename
        self._tile_overlap = tile_overlap
        self._tile_size = tile_size
        self._size = size
        self._stitching_mode = stitching_mode

    def read_region(self, location, size):
        with h5py.File(self._filename, 'r') as h5file:
            image_dataset = h5file['data']
            num_images, tile_height, tile_width, num_channels = image_dataset.shape

            stride_height = tile_height - self._tile_overlap[1]
            stride_width = tile_width - self._tile_overlap[0]

            total_rows = math.ceil((self._size[1] - self._tile_overlap[1]) / stride_height)
            total_cols = math.ceil((self._size[0] - self._tile_overlap[0]) / stride_width)

            assert total_rows * total_cols == num_images

            x, y = location
            w, h = size
            if x < 0 or y < 0 or x + w > self._size[0] or y + h > self._size[1]:
                raise ValueError("Requested region is out of bounds")

            start_row = y // stride_height
            end_row = min((y + h - 1) // stride_height + 1, total_rows)
            start_col = x // stride_width
            end_col = min((x + w - 1) // stride_width + 1, total_cols)

            if self._stitching_mode == StitchingMode.AVERAGE:
                divisor_array = np.zeros((h, w), dtype=np.uint8)
            stitched_image = np.zeros((h, w, num_channels), dtype=np.uint8)
            for i in range(start_row, end_row):
                for j in range(start_col, end_col):
                    tile_idx = (i * total_cols) + j
                    tile = image_dataset[tile_idx]

                    start_y = i * stride_height - y
                    end_y = start_y + tile_height
                    start_x = j * stride_width - x
                    end_x = start_x + tile_width

                    img_start_y = max(0, start_y)
                    img_end_y = min(h, end_y)
                    img_start_x = max(0, start_x)
                    img_end_x = min(w, end_x)

                    if self._stitching_mode == StitchingMode.CROP:
                        crop_start_y = img_start_y - start_y
                        crop_end_y = img_end_y - start_y
                        crop_start_x = img_start_x - start_x
                        crop_end_x = img_end_x - start_x

                        cropped_tile = tile[crop_start_y:crop_end_y, crop_start_x:crop_end_x]
                        stitched_image[img_start_y:img_end_y, img_start_x:img_end_x] = cropped_tile
                    elif self._stitching_mode == StitchingMode.AVERAGE:
                        stitched_image[img_start_y:img_end_y, img_start_x:img_end_x] += tile[:img_end_y - img_start_y,
                                                                                        :img_end_x - img_start_x]
                        divisor_array[img_start_y:img_end_y, img_start_x:img_end_x] += 1
                    else:
                        raise ValueError("Unsupported stitching mode")

        if self._stitching_mode == StitchingMode.AVERAGE:
            stitched_image = np.round(stitched_image / divisor_array[..., np.newaxis], 0).astype(np.uint8)

        return PIL.Image.fromarray(stitched_image)


if __name__ == "__main__":
    import tqdm
    class RandomColoredTilesDataset:
        def __init__(self, num_tiles=500, tile_width=128, tile_height=128):
            self.num_tiles = num_tiles
            self.tile_width = tile_width
            self.tile_height = tile_height

        def __len__(self):
            return self.num_tiles

        def __getitem__(self, idx):
            if idx >= self.num_tiles or idx < 0:
                raise IndexError("Index out of range")

            color = tuple(np.random.randint(0, 256, size=3))
            image = PIL.Image.new('RGB', (self.tile_width, self.tile_height), color)
            return image


    tile_width, tile_height = 128, 128
    image_width, image_height = 128 * 20, 128 * 25
    tiles_per_row = image_width // tile_width
    tiles_per_column = image_height // tile_height
    total_tiles = tiles_per_row * tiles_per_column

    # Create the dataset with the required number of tiles
    dataset = RandomColoredTilesDataset(num_tiles=total_tiles)

    # Initialize an empty array for the output image
    output_image_array = np.zeros((image_height, image_width, 3), dtype=np.uint8)

    # Fill the output image with the tiles from the dataset
    tile_idx = 0
    for y in range(0, image_height, tile_height):
        for x in range(0, image_width, tile_width):
            tile_image = dataset[tile_idx]
            tile_array = np.array(tile_image)
            output_image_array[y:y + tile_height, x:x + tile_width] = tile_array
            tile_idx += 1

    # Convert the output array to a PIL Image
    output_image = PIL.Image.fromarray(output_image_array)

    writer = TifffileImageWriter("test.tiff",
                                 interpolator=Resampling.NEAREST,
                                 pyramid=False, tile_size=(256, 256), mpp=(1.0, 1.0),
                                 size=(2560, 3200, 3))

    writer.from_pil(output_image)  # will be transposed but doesn't matter

    image_fn = Path("test.tiff")
    tile_size = (128, 128)
    tile_overlap = (10, 10)
    mpp = 1.0

    ds = TiledROIsSlideImageDataset.from_standard_tiling(
        image_fn,
        mpp=mpp,
        tile_mode=TilingMode.overflow,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        crop=False,
    )

    slide = ds.slide_image
    print(slide.mpp)
    scale = slide.get_scaling(mpp)
    size = slide.get_scaled_size(scale)
    print(size)

    writer = H5FileImageWriter(
        "output.h5",
        size=size,
        mpp=mpp,
        tile_size=tile_size,
        tile_overlap=tile_overlap, progress=tqdm.tqdm)
    writer.from_iterator(ds)

    # TRY AVERAGE too, you can inspect the division_array what it does.
    reader = H5FileImageReader("output.h5", tile_size=tile_size, tile_overlap=tile_overlap, size=size,
                               stitching_mode=StitchingMode.CROP)

    a = reader.read_region((0, 0), (2400, 2400))
    b = slide.read_region((0, 0), 1, (2400, 2400))
