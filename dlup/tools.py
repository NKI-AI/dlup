# Copyright (c) dlup contributors

"""Utility classes to easily extend sequences logic without creating new classes."""

import bisect
import collections
import itertools
from typing import Any, Callable, Iterator, List, Sequence, TypeVar, Union, overload

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)


class MapSequence(collections.abc.Sequence[T]):
    """Wraps the __getitem__ function of a sequence.

    It's similar to the built-in map(), although instead of returning
    an iterator, a sequence object is returned instead.
    """

    _function: Callable[[Any, T], T]
    _sequence: Sequence[T]

    def __init__(self, function: Callable[[Any, T], T], sequence: Sequence[T]) -> None:
        self._function = function
        self._sequence = sequence

    def __len__(self) -> int:
        return len(self._sequence)

    @overload
    def __getitem__(self, key: int) -> T:
        ...

    @overload
    def __getitem__(self, key: slice) -> Sequence[T]:
        ...

    def __getitem__(self, key: Union[int, slice]) -> Union[T, Sequence[T]]:
        if isinstance(key, slice):
            # Handle slices by returning a subsequence
            return [self._function(i, item) for i, item in enumerate(self._sequence[key])]
        return self._function(key, self._sequence[key])

    def __iter__(self) -> Iterator[T]:
        for i in range(len(self)):
            yield self[i]


class ConcatSequences(collections.abc.Sequence[T]):
    """Concatenate two or more sequences."""

    _sequences: List[Sequence[T]]
    _starting_indices: List[int]
    _len: int

    def __init__(self, sequences: List[Sequence[T]]) -> None:
        self._sequences = sequences
        cumsum = list(itertools.accumulate([len(s) for s in sequences]))
        self._starting_indices = [0] + cumsum[:-1]
        self._len = cumsum[-1]

    @overload
    def __getitem__(self, key: int) -> T:
        ...

    @overload
    def __getitem__(self, key: slice) -> Sequence[T]:
        ...

    def __getitem__(self, key: Union[int, slice]) -> Union[T, Sequence[T]]:
        if isinstance(key, slice):
            return [self[i] for i in range(*key.indices(len(self)))]
        starting_index = bisect.bisect_right(self._starting_indices, key) - 1
        sequence_index = key - self._starting_indices[starting_index]
        return self._sequences[starting_index][sequence_index]

    def __len__(self) -> int:
        return self._len

    def __iter__(self) -> Iterator[T]:
        for i in range(len(self)):
            yield self[i]
