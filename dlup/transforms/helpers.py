# coding=utf-8
# Copyright (c) dlup contributors
import abc
import logging
from functools import partial
from typing import Any, Callable, Iterable, Optional, Union

from dlup.utils import _TORCHVISION_AVAILABLE
from dlup.utils.decorators import convert_numpy_to_tensor
from dlup.utils.types import string_classes


class DlupTransform(abc.ABC):
    """Abstract baseclass for dlup transforms."""

    def __init__(self, *args, **kwargs):
        self.logger = logging.getLogger(type(self).__name__)
        # Pytorch is channels first
        self.channels_order = "NCWH"

    @abc.abstractmethod
    def __call__(self, sample):
        pass
