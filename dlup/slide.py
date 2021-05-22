# coding=utf-8
# Copyright (c) dlup contributors

import functools
import json
import math
import pathlib
import warnings
from dataclasses import dataclass
from typing import Dict, NamedTuple, Optional, Tuple, Union

import numpy as np  # type: ignore
import openslide  # type: ignore
import PIL.Image  # type: ignore
import tifftools  # type: ignore
from numpy.typing import ArrayLike
from openslide.lowlevel import OpenSlideError  # type: ignore
from openslide.lowlevel import OpenSlideUnsupportedFormatError


class Level(NamedTuple):
    level: int
    downsample: float
    mpp: Union[float, ArrayLike]
    shape: ArrayLike


@dataclass
class TileBox:
    idx: np.ndarray
    coords: tuple
    size: tuple
    post_crop_bbox: tuple
    offset: tuple
    num_tiles: tuple
    border_mode: str


def _ensure_array(obj):
    if not isinstance(obj, (tuple, list)):
        obj = (obj, obj)
    return np.asarray(obj)


def near_power_of_two(val1, val2, tolerance=0.02):
    """Check if two values are different by nearly a power of two.

    Parameters
    ----------
    val1 :
        the first value to check.
    val2 :
        the second value to check.
    tolerance :
        the maximum difference in the log2 ratio's mantissa. (Default value = 0.02)

    Returns
    -------
    type
        True if the values are nearly a power of two different from each
        other; false otherwise.

    """
    # If one or more of the values is zero or they have different signs, then
    # return False
    if val1 * val2 <= 0:
        return False
    log2ratio = math.log(float(val1) / float(val2)) / math.log(2)
    # Compare the mantissa of the ratio's log2 value.
    return abs(log2ratio - round(log2ratio)) < tolerance


class Slide:
    """
    Utility class to handle whole-slide images, which relies on OpenSlide.
    """

    def __init__(self, file_name: pathlib.Path, slide_uid: Optional[str] = None):
        self.file_name = file_name
        self.slide_uid = slide_uid

        self.__read_openslide()

        try:
            magnification = self._openslide.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER]
            magnification = float(magnification) if magnification else None
        except KeyError:
            magnification = None
        except (ValueError, openslide.lowlevel.OpenSlideError) as exception:
            raise RuntimeError(f"Could not extract objective power of {self.file_name} with exception: {exception}.")
        self.magnification = magnification

        try:
            mpp_x = float(self._openslide.properties[openslide.PROPERTY_NAME_MPP_X])
            mpp_y = float(self._openslide.properties[openslide.PROPERTY_NAME_MPP_Y])
            mpp = np.array([mpp_y, mpp_x])
        except KeyError:
            mpp = None
        except (ValueError, openslide.lowlevel.OpenSlideError) as exception:
            raise RuntimeError(f"Could not extract mpp of {self.file_name} with exception: {exception}.")

        if mpp is None:
            raise RuntimeError(f"Could not parse mpp of {self.file_name}.")

        # The logic has to be improved here, in case this actually happens
        if np.abs(mpp[0] - mpp[1]) / mpp.max() > 0.001:
            warnings.warn(
                f"{self.file_name}: mpp_x and mpp_y are expected to be equal. " f"Got {mpp[0]} and {mpp[1]}."
            )
        self.mpp = float(mpp.max())

        self.__compute_available_levels()  # Also sets self.n_channels

    def close(self):
        self._openslide.close()

    def __read_openslide(self):
        try:
            slide = openslide.OpenSlide(str(self.file_name))
        except OpenSlideUnsupportedFormatError as e:
            raise RuntimeError(f"File not supported by OpenSlide: {e}.")
        except OpenSlideError as e:
            raise RuntimeError(f"File cannot be opened via OpenSlide: {e}.")
        # if libtiff_ctypes:
        #     try:
        #         self._tiff_info = tifftools.read_tiff(self.large_image_path)
        #     except (
        #         tifftools.TifftoolsException,
        #         Exception,
        #     ) as e:  # Check what else you can get rather than have such a broad exception
        #         self.logger.info(f"Cannot read tiff info: {e}.")
        #         pass
        self._openslide = slide

    def get_tile(self, start_coords, tile_size, level):
        tile = self.__get_openslide_tile(start_coords, tile_size, level=level)

        return tile

    # This has to be cached.
    def __get_openslide_tile(self, start_coords, tile_size, level: int = 0) -> PIL.Image:
        # If the mode is exact, it is possible we get float coordinates. To do that properly,
        # you need to resample on a new grid. We take the option of having a slightly higher resolution.
        # Start coords needs to be in the level 0 reference frame.
        try:
            tile = self._openslide.read_region(tuple(start_coords), level, tile_size)
        except openslide.lowlevel.OpenSlideError as exc:
            raise Exception(f"Failed to get OpenSlide region ({exc!r}).")

        return tile

    def __compute_available_levels(self):
        """Some SVS files (notably some NDPI variants) have levels that cannot be
        read.  Get a list of levels, check that each is at least potentially
        readable, and return a list of these sorted highest-resolution first.

        Parameters
        ----------
        path :
            the path of the SVS file.  After a failure, the file is
            reopened to reset the error state.

        Returns
        -------
        type
            levels.  A list of valid levels, each of which is a
            dictionary of level (the internal 0-based level number), width, and
            height.

        """
        levels = []
        svs_level_dimensions = self._openslide.level_dimensions
        for svs_level in range(len(svs_level_dimensions)):
            try:
                temp_arr = self._openslide.read_region((0, 0), svs_level, (1, 1))
                downsample_factor = self._openslide.level_downsamples[svs_level]
                level = Level(
                    level=svs_level,
                    downsample=downsample_factor,
                    mpp=self.mpp * downsample_factor,
                    shape=svs_level_dimensions[svs_level],
                )
                if level.shape[0] > 0 and level.shape[1] > 0:
                    # add to the list so that we can sort by resolution and
                    # then by earlier entries
                    levels.append((level.shape[0] * level.shape[1], -len(levels), level))
            except OpenSlideError:
                self.__read_openslide()
        # sort highest resolution first.
        levels = [entry[-1] for entry in sorted(levels, reverse=True, key=lambda x: x[:-1])]
        # Discard levels that are not a power-of-two compared to the highest
        # resolution level.
        levels = [
            entry
            for entry in levels
            if near_power_of_two(levels[0].shape[0], entry.shape[0])
            and near_power_of_two(levels[0].shape[1], entry.shape[1])
        ]
        self.n_channels = np.array(temp_arr).shape[-1]
        self.levels = levels

    def shape_at_mpp(self, mpp: Union[float, ArrayLike], exact: bool = False) -> Tuple[np.ndarray, float, float]:
        scaling = self.mpp / mpp
        dimensions = np.array(self.shape) * scaling
        effective_mpp = mpp

        if exact:
            dimensions = np.ceil(dimensions).astype(int)
            scaling = (np.array(self.shape) * dimensions)[0]  # mpp is assumed to be isotropic
            effective_mpp = self.mpp * scaling

        return dimensions, scaling, effective_mpp

    def shape_at_magnification(self, magnification: float, exact: bool = False) -> Tuple[np.ndarray, float, float]:
        scaling = magnification / self.magnification
        dimensions = np.array(self.shape) * scaling
        effective_magnification = magnification

        if exact:
            dimensions = np.ceil(dimensions).astype(int)
            scaling = (np.array(self.shape) * dimensions)[0]  # mpp is assumed to be isotropic
            effective_magnification = self.magnification * scaling

        return dimensions, scaling, effective_magnification

    @property
    def is_lossy_tiff(self) -> bool:
        """
        Checks if input image is actually a lossy tiff file. Useful if you want to save the output
        as non-jpeg, while the input actually is.
        """
        return is_lossy_tiff(self.file_name)

    def get_best_level_for_downsample(self, downsample_factor: float) -> int:
        return self._openslide.get_best_level_for_downsample(downsample_factor)

    def get_thumbnail(self, size=512) -> PIL.Image:
        """
        Returns an RGB numpy thumbnail for the current slide.
        """
        thumbnail = self._openslide.get_thumbnail([size] * 2)

        return thumbnail

    @functools.cached_property
    def properties(self) -> dict:
        """Return additional known metadata about the tile source.  Data returned
        from this method is not guaranteed to be in any particular format or
        have specific values.

        Returns
        -------
        type
            a dictionary of data or None.

        """
        results: Dict[str, Dict] = {"openslide": {}}
        for key in self._openslide.properties:
            results["openslide"][key] = self._openslide.properties[key]
            if key == "openslide.comment":
                leader = self._openslide.properties[key].split("\n", 1)[0].strip()
                if "aperio" in leader.lower():
                    results["aperio_version"] = leader
        # TODO Mirax

        return results

    @property
    def patient_uid(self) -> str:
        return self.file_name.parent.name

    @property
    def level_dimensions(self) -> dict:
        return self._openslide.level_dimensions

    @functools.cached_property
    def shape(self) -> Tuple[int, int]:
        return self._openslide.dimensions

    @property
    def width(self) -> int:
        return self.shape[0]

    @property
    def height(self) -> int:
        return self.shape[1]

    def __repr__(self) -> str:
        out_str = f"Slide(slide_uid={self.slide_uid}, file_name={self.file_name}, shape={self.shape}"
        for key in self.properties:
            if key == "openslide":
                val = json.dumps(self.properties[key])[:15] + "...}"
            else:
                val = self.properties[key]

            out_str += f", {key}={val}"
        out_str += ")"
        return out_str


# Inspired by
# https://github.com/girder/large_image/blob/af518ba2187e60ccd7bc59f7c6d8e0472ef29ee0/utilities/converter/large_image_converter/__init__.py#L414
# TODO: Cleanup
def is_lossy_tiff(filename):
    try:
        tiffinfo = tifftools.read_tiff(filename)
    except Exception:
        return False

    is_compressed = bool(
        tifftools.constants.Compression[tiffinfo["ifds"][0]["tags"][tifftools.Tag.Compression.value]["data"][0]].lossy
    )
    is_eight_bit = True
    try:
        if not all(
            val == tifftools.constants.SampleFormat.uint
            for val in tiffinfo["ifds"][0]["tags"][tifftools.Tag.SampleFormat.value]["data"]
        ):
            is_eight_bit = False

        if tifftools.Tag.BitsPerSample.value in tiffinfo["ifds"][0]["tags"] and not all(
            val == 8 for val in tiffinfo["ifds"][0]["tags"][tifftools.Tag.BitsPerSample.value]["data"]
        ):
            is_eight_bit = False
    except:
        return False

    return is_compressed and is_eight_bit
