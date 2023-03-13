# coding=utf-8
# Copyright (c) dlup contributors

"""Utility classes to easily extend sequences logic without creating new classes."""
import bisect
import collections
import itertools
from typing import Any, Callable, Iterable, Sequence

import numpy as np

import dlup.tiling


class MapSequence(collections.abc.Sequence):
    """Wraps the __getitem__ function of a sequence.

    It's similar to the built-in map(), although instead of returning
    an iterator, a sequence object is returned instead.
    """

    def __init__(self, function: Callable[[Any], dlup.tiling.Grid], sequence: dlup.tiling.Grid) -> None:
        self._function = function
        self._sequence = sequence

    def __len__(self) -> int:
        return len(self._sequence)

    def __getitem__(self, key: int) -> "MapSequence":
        return self._function(key, self._sequence[key])

    def __iter__(self) -> Iterable:
        for i in range(len(self)):
            yield self[i]


class ConcatSequences(collections.abc.Sequence):
    """Concatenate two or more sequences."""

    def __init__(self, sequences: Sequence):
        self._sequences = sequences
        cumsum = list(itertools.accumulate([len(s) for s in sequences]))
        self._starting_indices = [0] + cumsum[:-1]
        self._len = cumsum[-1]

    def __getitem__(self, key: int) -> list[np.int_]:
        starting_index = bisect.bisect_right(self._starting_indices, key) - 1
        sequence_index = key - self._starting_indices[starting_index]
        return self._sequences[starting_index][sequence_index]

    def __len__(self) -> int:
        return self._len

    def __iter__(self) -> Iterable[Any]:
        for i in range(len(self)):
            yield self[i]
