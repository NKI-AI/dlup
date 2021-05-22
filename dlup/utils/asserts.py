# coding=utf-8
# Copyright (c) DLUP Contributors


def assert_probability(number: float):
    if not 0 <= number <= 1:
        raise ValueError(f"Expected number to be a probability. Got {number}.")
