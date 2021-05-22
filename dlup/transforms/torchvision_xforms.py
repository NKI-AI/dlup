# coding=utf-8
# Copyright (c) dlup contributors
import warnings

import packaging.version  # pylint: disable=import-error
import torchvision  # pylint: disable=import-error
import torchvision.transforms as transforms  # pylint: disable=import-error

from dlup.transforms._helpers import _wrap_tv

__all__ = (
    "TvLinearTransformation",
    "TvColorJitter",
    "TvGrayscale",
    "TvNormalize",
    "TvCenterCrop",
    "TvPad",
    "TvFiveCrop",
    "TvTenCrop",
    "TvRandomGrayScale",
    "TvRandomErasing",
    "TvRandomCrop",
    "TvRandomHorizontalFlip",
    "TvRandomVerticalFlip",
    "TvRandomPerspective",
    "TvRandomRotation",
    "TvRandomAffine",
)


newer__all__ = [
    "TvRandomInvert",
    "TvRandomPosterize",
    "TvRandomSolarize",
    "TvRandomAdjustSharpness",
    "TvRandomAutocontrast",
    "TvRandomEqualize",
]


# The transforms which only transform the image can be readily mapped
TvLinearTransformation = _wrap_tv(transforms.LinearTransformation)
TvColorJitter = _wrap_tv(transforms.ColorJitter)
TvGrayscale = _wrap_tv(transforms.Grayscale)
TvNormalize = _wrap_tv(transforms.Normalize)

# Random transforms which keep mask invariant
TvRandomGrayScale = _wrap_tv(transforms.RandomGrayscale)
TvRandomErasing = _wrap_tv(transforms.RandomErasing)

# Only supported in newer versions
if packaging.version.parse(torchvision.__version__) >= packaging.version.parse("0.9.0"):
    TvRandomInvert = _wrap_tv(transforms.RandomInvert)
    TvRandomPosterize = _wrap_tv(transforms.RandomPosterize)
    TvRandomSolarize = _wrap_tv(transforms.RandomSolarize)
    TvRandomAdjustSharpness = _wrap_tv(transforms.RandomAdjustSharpness)
    TvRandomAutocontrast = _wrap_tv(transforms.RandomAutocontrast)
    TvRandomEqualize = _wrap_tv(transforms.RandomEqualize)
    __all__ = tuple(list(__all__) + newer__all__)  # type: ignore
else:
    warnings.warn(
        f"torchvision version {torchvision.__version__} does not support all transforms. Requires at least 0.9.0. "
        f"These transforms are missing: {', '.join(newer__all__)}"
    )

# Wrap transforms which cannot have any mask
TvRandomCrop = _wrap_tv(transforms.RandomCrop, cannot_apply_mask=True)
TvRandomHorizontalFlip = _wrap_tv(transforms.RandomHorizontalFlip, cannot_apply_mask=True)
TvRandomVerticalFlip = _wrap_tv(transforms.RandomVerticalFlip, cannot_apply_mask=True)
TvRandomPerspective = _wrap_tv(transforms.RandomPerspective, cannot_apply_mask=True)
TvRandomRotation = _wrap_tv(transforms.RandomRotation, cannot_apply_mask=True)
TvRandomAffine = _wrap_tv(transforms.RandomAffine, cannot_apply_mask=True)


# TvResize = _wrap_tv(torchvision.transforms.Resize, keys=["image", "mask"])

# Transforms that adjust the mask without interpolation defects
TvCenterCrop = _wrap_tv(transforms.CenterCrop, keys=["image", "mask"])
TvPad = _wrap_tv(transforms.Pad, keys=["image", "mask"])
TvFiveCrop = _wrap_tv(transforms.FiveCrop, keys=["image", "mask"])
TvTenCrop = _wrap_tv(transforms.TenCrop, keys=["image", "mask"])
