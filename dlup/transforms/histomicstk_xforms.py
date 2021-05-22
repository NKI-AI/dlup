# coding=utf-8
# Copyright (c) dlup contributors
import warnings

import torch

from dlup.transforms import DlupTransform

# Do not strictly require availability of histomicstk
from dlup.utils import _HISTOMICSTK_AVAILABLE
from dlup.utils.decorators import apply_as_numpy, output_as_tensor

if _HISTOMICSTK_AVAILABLE:
    from histomicstk.preprocessing.augmentation.color_augmentation import (  # pylint: disable=import-error
        rgb_perturb_stain_concentration,
    )


def apply_with_channels_last(array, func):
    """
    Apply a function to an image which assumes that the channel dimension is at the last axis for an input with
    channels first.

    Parameters
    ----------
    array : np.ndarray
    func : callable

    Returns
    -------
    np.ndarray
    """
    if not array.ndim == 3:
        raise ValueError(f"Can only apply when dimension is 3. Got {array.ndim}.")

    if array.shape[0] != 3:
        raise ValueError(f"Expected first dimension to be the channels. Got {array.shape}.")

    return func(array.transpose(1, 2, 0)).transpose(2, 0, 1)


class HtkRgbPerturbStainConcentration(DlupTransform):
    # TODO: Add parameters and docstring
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not _HISTOMICSTK_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError("You want to use transforms from `histomicstk` which is not installed yet.")
        self.transform = rgb_perturb_stain_concentration

    @output_as_tensor
    @apply_as_numpy
    def __call__(self, sample):
        if self.channels_order == "NCHW":
            sample["image"] = apply_with_channels_last(sample["image"], self.transform)
        else:
            sample["image"] = self.transform(sample["image"])

        return sample


# TODO: Move this to a testing environment
if __name__ == "__main__":
    xform = HtkRgbPerturbStainConcentration()
    sample = {"image": torch.Tensor([1])}

    output = xform(sample)
    assert isinstance(output["image"], torch.Tensor)
