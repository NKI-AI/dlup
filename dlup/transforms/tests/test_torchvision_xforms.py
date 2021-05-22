# coding=utf-8
# Copyright (c) dlup contributors
import numpy as np

from dlup.transforms import TvGrayscale


def test_tv_grayscale():
    grayscale_transform = TvGrayscale()

    sample = {"image": np.random.random((3, 15, 20))}
    output = grayscale_transform(sample)

    assert list(output["image"].shape) == [1, 15, 20]

    sample = {
        "image": np.random.random((3, 15, 20)),
        "mask": np.random.random((3, 15, 20)),
    }
    output = grayscale_transform(sample)
    assert np.all(output["mask"].numpy() == sample["mask"])


if __name__ == "__main__":
    test_tv_grayscale()
