# coding=utf-8
# Copyright (c) dlup contributors
import abc
import collections.abc
import logging
from functools import partial
from typing import Any, Callable, Iterable, Optional, Union

import torch.nn as nn

from dlup.utils import _TORCHVISION_AVAILABLE
from dlup.utils.decorators import convert_numpy_to_tensor
from dlup.utils.types import string_classes


class DlupTransform(abc.ABC):
    def __init__(self, *args, **kwargs):
        self.logger = logging.getLogger(type(self).__name__)
        # Pytorch is channels first
        self.channels_order = "NCWH"

    @abc.abstractmethod
    def __call__(self, sample):
        pass

    def _transforms_get(self):
        raise NotImplementedError("It seems you are not using a " "proper tansformation object!")

    def _transforms_set(self, value: Optional[Callable[[Any, Any], None]]):
        raise NotImplementedError("It seems you are not using a " "proper tansformation object!")

    transforms = property(_transforms_get, _transforms_set)

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string


def _wrap_tv(
    original_class=None,
    keys: Optional[Union[Iterable]] = "image",
    cannot_apply_mask: bool = False,
):
    """
    Decorator to wrap torchvision transforms to dlup transforms.

    Parameters
    ----------
    original_class : class
    keys : Union[List, str]
        Key to wrap the transform in, e.g. if "image" the output will transform the dictionary with key "image".
    cannot_apply_mask : bool
        If true, this means this transform cannot be applied to a mask, point or bounding box due randomness
        in the transform.

    Returns
    -------

    """
    if original_class is None:
        return partial(_wrap_tv, keys=keys, cannot_apply_mask=cannot_apply_mask)

    if isinstance(keys, string_classes):
        keys = [keys]

    class Wrapper(DlupTransform):
        def __init__(self, *args, **kwargs):
            if not _TORCHVISION_AVAILABLE:  # pragma: no cover
                raise ModuleNotFoundError("You want to use transforms from `torchvision` which is not installed yet.")

            super().__init__()
            self.original_instance = original_class(*args, **kwargs)

        def __getattribute__(self, s):
            try:
                attr = super().__getattribute__(s)
            except AttributeError:
                pass
            else:
                return attr
            attr = self.original_instance.__getattribute__(s)

            # The following is called whenever this is an instance method such as get_hyperparams()
            if isinstance(self.__init__, type(attr)):  # it is an instance method
                return attr
            return attr  # This is called when we access a property (either decorated or defined in constructor)

        def __call__(self, sample):
            sample = convert_numpy_to_tensor(sample)

            if cannot_apply_mask:
                forbidden_keys = ["mask", "points", "bbox", "contour"]
                if any(_ in sample.keys() for _ in forbidden_keys):
                    raise ValueError(
                        f"Cannot apply torchvision {original_class.__name__} "
                        f"due to random changes in image input geometry."
                    )

            # Points and bbox might need a custom function to implement, or a 'poor mans' version converting to a mask.
            if "points" in sample:
                raise NotImplementedError

            if "bbox" in sample:
                raise NotImplementedError

            if "contour" in sample:
                raise NotImplementedError

            for key in keys:
                data = sample[key]
                if data.ndim != 3:
                    raise ValueError(
                        f"torchvision transforms can only be applied to 2D data. Got {data.shape} for {key}."
                    )
                sample[key] = self.original_instance.__call__(data)
            return sample

        def __dir__(self) -> Iterable[str]:
            return self.original_instance.__dir__()

        def __repr__(self):
            return f"Tv{self.original_instance.__repr__()}"

    Wrapper.__name__ = original_class.__name__
    original_doc = original_class.__doc__

    Wrapper.__doc__ = f"Wrapped torchvision class to dlup format.\nOriginal documentation:\n{original_doc}"
    return Wrapper
