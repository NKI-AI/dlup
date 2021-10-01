# coding=utf-8
# Copyright (c) dlup contributors

"""Utility classes to easily extend sequences logic without creating new classes."""

import bisect
import collections
import functools
import itertools


class MapSequence(collections.abc.Sequence):
    """Wraps the __getitem__ function of a sequence.

    It's similar to the built-in map(), although instead of returning
    an iterator, a sequence object is returned instead.
    """

    def __init__(self, function, sequence):
        self._function = function
        self._sequence = sequence

    def __len__(self):
        return len(self._sequence)

    def __getitem__(self, key):
        return self._function(key, self._sequence[key])

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


class IndexSequence(collections.abc.Sequence):
    """Returns a sub-sequence from the provided list of indices."""

    def __init__(self, indices, sequence):
        self._indices = indices
        self._sequence = sequence

    def __getitem__(self, key):
        return self._sequence[self._indices[key]]

    def __len__(self):
        return len(self._indices)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


class ConcatSequences(collections.abc.Sequence):
    """Concatenate two or more sequences."""

    def __init__(self, sequences):
        self._sequences = sequences
        cumsum = list(itertools.accumulate([len(s) for s in sequences]))
        self._starting_indices = [0] + cumsum[:-1]
        self._len = cumsum[-1]

    def __getitem__(self, key):
        starting_index = bisect.bisect_right(self._starting_indices, key) - 1
        sequence_index = key - self._starting_indices[starting_index]
        return self._sequences[starting_index][sequence_index]

    def __len__(self):
        return self._len

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
