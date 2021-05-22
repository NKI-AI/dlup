# coding=utf-8
# Copyright (c) dlup contributors
"""Decorates which can wrap existing transforms (augmentations) to the dlup format."""
import collections.abc
import functools
import re

import numpy as np
import torch

from dlup.utils.types import string_classes

np_str_obj_array_pattern = re.compile(r"[SaUO]")


# Function taken from:
# https://github.com/pytorch/pytorch/blob/a469336292e411a056fbe9445ed948108acde57d/torch/utils/data/_utils/collate.py#L15
# Under BSD license. Slightly adapted.
def convert_numpy_to_tensor(data):
    """Converts each NumPy array data field into a tensor."""
    elem_type = type(data)
    if isinstance(data, torch.Tensor):
        return data
    if elem_type.__module__ == "numpy" and elem_type.__name__ != "str_" and elem_type.__name__ != "string_":
        # array of string classes and object
        if elem_type.__name__ == "ndarray" and np_str_obj_array_pattern.search(data.dtype.str) is not None:
            return data
        return torch.as_tensor(data)
    if isinstance(data, collections.abc.Mapping):
        return {key: convert_numpy_to_tensor(data[key]) for key in data}
    if isinstance(data, tuple) and hasattr(data, "_fields"):  # namedtuple
        return elem_type(*(convert_numpy_to_tensor(d) for d in data))
    if isinstance(data, collections.abc.Sequence) and not isinstance(data, string_classes):
        return [convert_numpy_to_tensor(d) for d in data]
    return data


def convert_tensor_to_numpy(data):
    """Converts each NumPy array data field into a tensor."""
    elem_type = type(data)
    if isinstance(data, np.ndarray):
        return data
    if isinstance(data, torch.Tensor):
        return data.numpy()
    if isinstance(data, collections.abc.Mapping):
        return {key: convert_tensor_to_numpy(data[key]) for key in data}
    if isinstance(data, tuple) and hasattr(data, "_fields"):  # namedtuple
        return elem_type(*(convert_tensor_to_numpy(d) for d in data))
    if isinstance(data, collections.abc.Sequence) and not isinstance(data, string_classes):
        return [convert_tensor_to_numpy(d) for d in data]
    return data


def apply_as_numpy(func):
    """
    Decorate to ensure that when applying an operation the data is first converted to numpy and subsequently
    mapped back.

    Parameters
    ----------
    func : Callable

    Returns
    -------
    Callable
    """

    @functools.wraps(func)
    def wrapper_apply_as_numpy(*args, **kwargs):
        args = convert_tensor_to_numpy(args)
        kwargs = convert_tensor_to_numpy(kwargs)
        output = func(*args, **kwargs)
        return output

    return wrapper_apply_as_numpy


def output_as_tensor(func):
    """
    Decorate to ensure that when applying an the data in any output mapping is a torch tensor

    Parameters
    ----------
    func : Callable

    Returns
    -------
    Callable
    """

    @functools.wraps(func)
    def wrapper_output_as_tensor(*args, **kwargs):
        output = func(*args, **kwargs)
        return convert_numpy_to_tensor(output)

    return wrapper_output_as_tensor


if __name__ == "__main__":

    @apply_as_numpy
    def _test(arr):
        return arr

    @output_as_tensor
    @apply_as_numpy
    def _test2(arr):
        return arr

    assert isinstance(_test(torch.Tensor(1)), np.ndarray)
    assert isinstance(_test(np.array(1)), np.ndarray)

    assert isinstance(_test2(torch.Tensor(1)), torch.Tensor)
    assert isinstance(_test2(np.array(1)), torch.Tensor)

    assert isinstance(_test2({"image": np.array(1)})["image"], torch.Tensor)
