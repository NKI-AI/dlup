# coding=utf-8
# Copyright (c) dlup contributors
"""General utilities for module imports"""
from importlib.util import find_spec


def _module_available(module_path: str) -> bool:
    r"""
    Check if a path is available in your environment
    >>> _module_available('os')
    True
    >>> _module_available('bla.bla')
    False

    Adapted from: https://github.com/PyTorchLightning/pytorch-lightning/blob/ef7d41692ca04bb9877da5c743f80fceecc6a100/pytorch_lightning/utilities/imports.py#L27
    Under Apache 2.0 license.
    """  # noqa: E501
    try:
        return find_spec(module_path) is not None
    except ModuleNotFoundError:
        return False


PYTORCH_AVAILABLE = _module_available("pytorch")
PYHALOXML_AVAILABLE = _module_available("pyhaloxml")
DARWIN_SDK_AVAILABLE = _module_available("darwin")
