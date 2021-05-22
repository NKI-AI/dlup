# coding=utf-8
# Copyright (c) dlup contributors
"""Main module to hold the dlup transforms (or wrapped versions from other libraries)."""
import random
import warnings
from typing import Dict, List, Sequence, Union

import numpy as np
import torch

from dlup.transforms._helpers import DlupTransform
from dlup.utils import _HISTOMICSTK_AVAILABLE, _TORCHVISION_AVAILABLE
from dlup.utils.asserts import assert_probability

if _TORCHVISION_AVAILABLE:
    from .torchvision_xforms import *
else:  # pragma: no cover
    warnings.warn("torchvision is not installed, which is required for dlup versions of torchvision transforms.")

if _HISTOMICSTK_AVAILABLE:
    from .histomicstk_xforms import *
else:  # pragma: no cover
    warnings.warn("histomicstk is not installed, which is required for dlup version of histomicstk transforms.")


class Compose(DlupTransform):
    """Compose several transformations together, for instance ClipAndScale and a flip.
    Code based on torchvision: https://github.com/pytorch/vision, but got forked from there as torchvision has some
    additional dependencies.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for transform in self.transforms:
            sample = transform(sample)
        return sample

    def __repr__(self):
        repr_string = self.__class__.__name__ + "("
        for transform in self.transforms:
            repr_string += "\n"
            repr_string += f"    {transform}"
        repr_string += "\n)"
        return repr_string


class RandomChoice(DlupTransform):
    """
    Select a transform randomly from the list
    """

    def __init__(self, transforms: List, weights: Union[np.ndarray, Sequence] = None) -> None:
        """
        Parameters
        ----------
        transforms : List
            List of transforms
        weights : iterable
            List of weights that can be cast to a numpy array. Must all be positive and non-zero.
        """
        super().__init__()
        self.transforms = transforms
        self.weights = weights
        if weights:
            if len(self.transforms) != len(weights):
                raise ValueError(
                    f"Should have same number of transforms and weights. Got {len(self.transforms)} and {len(weights)}."
                )

            weights = np.asarray(weights)
            if np.any(weights <= 0):
                raise ValueError(f"All weights should be positive numbers. Got {weights}.")

            self.weights = weights / weights.sum()

    def __call__(self, sample: Dict) -> Dict:
        transform = np.random.choice(self.transforms, 1, p=self.weights)
        return transform(sample)


class RandomOrder(DlupTransform):
    """
    Apply a list of transforms in a random order.
    """

    def __init__(self, transforms: Sequence):
        super().__init__()
        self.transforms = transforms

    def __call__(self, sample):
        order = list(range(len(self.transforms)))
        random.shuffle(order)
        for idx in order:
            sample = self.transforms[idx](sample)
        return sample


class RandomAxisFlip(DlupTransform):
    def __init__(self, probability: float, axis: str) -> None:
        super().__init__()
        assert_probability(probability)
        if axis not in ["H", "V"]:
            raise ValueError(f"Axis not in H or V. Got {axis}.")
        self.probability = probability
        self.axis = axis

    def __call__(self, sample):
        if np.random.random() > self.probability:
            return sample

        axis = 0 if self.axis == "V" else 1
        if self.channels_order == "NCHW":
            axis += 1

        sample["image"] = self.flip(sample["image"], axis=axis)
        if "mask" in sample:
            sample["mask"] = self.flip(sample["mask"], axis=axis)

        if "bbox" in sample:
            raise NotImplementedError

        if "points" in sample:
            raise NotImplementedError

        return sample

    @staticmethod
    def flip(data, axis):
        if isinstance(data, np.ndarray):
            data = np.flip(data, axis=axis)

        elif isinstance(data, torch.Tensor):
            data = torch.flip(data, axis)

        else:
            raise ValueError(f"Type {type(data)} is not supported.")

        return data

    def __repr__(self):
        return f"{type(self).__name__}(probability={self.probability}, axis={self.axis})"


class RandomDiagonalFlip(DlupTransform):
    def __init__(self, probability: float, diagonal: str) -> None:
        super().__init__()
        assert_probability(probability)
        if diagonal not in ["D0", "D1"]:
            raise ValueError(f"Axis not in D0 or D1. Got {diagonal}.")
        self.probability = probability
        self.diagonal = diagonal

    def __call__(self, sample):
        if np.random.random() > self.probability:
            return sample

        shift = 1 if self.channels_order == "NCHW" else 0
        axis = (0 + shift, 1 + shift)

        sample["image"] = self.transpose(sample["image"])
        if "mask" in sample:
            sample["mask"] = self.transpose(sample["mask"])

        if "bbox" in sample:
            raise NotImplementedError

        if "points" in sample:
            raise NotImplementedError

        return sample

    @staticmethod
    def transpose(data):
        if isinstance(data, np.ndarray):
            raise NotImplementedError

        elif isinstance(data, torch.Tensor):
            raise NotImplementedError

        else:
            raise ValueError(f"Type {type(data)} is not supported.")

        return data

    def __repr__(self):
        return f"{type(self).__name__}(probability={self.probability}, diagonal={self.diagonal})"
