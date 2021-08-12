# coding=utf-8
# Copyright (c) dlup contributors

"""Utility classes to easily extend sequences logic without creating new classes."""

import collections


class MapSequence(collections.abc.Sequence):
    """Wraps the __getitem__ function of a sequence.

    It's similar to the built-in map(), although instead of returning
    an iterator, a sequence object is returned instead.
    """

    def __init__(self, function, sequence):
        self._function = function
        self._sequence = sequence

    def __getitem__(self, key):
        return self._function(key, self._sequence[key])

    def __len__(self):
        return len(self._sequence)


class IndexSequence(collections.abc.Sequence):
    """Returns a sub-sequence from the provided list of indices."""

    def __init__(self, indices, sequence):
        self._indices = indices
        self._sequence = sequence

    def __getitem__(self, key):
        return self._sequence[self._indices[key]]

    def __len__(self):
        return len(self._indices)
