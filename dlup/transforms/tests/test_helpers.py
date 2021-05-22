# coding=utf-8
# Copyright (c) dlup contributors
import numpy as np
import torch

from dlup.transforms._helpers import _wrap_tv


# TODO: change the wrapper in a pytest.
def test__wrap_tv():
    @_wrap_tv(keys="not_image")
    class Foo:
        def __init__(self, vars):
            self.vars = vars

        def a(self):
            return None

        @staticmethod
        def b(z):
            return 4 * z

        def __call__(self, idx):
            return self.value * idx

        @property
        def value(self):
            return self.vars

    foo = Foo(3)

    # We are checking that our function is being left intact.
    var_a = foo.a()  # pylint:disable=assignment-from-none
    assert not var_a
    var_b = foo.b(3)
    assert var_b == 12
    value = foo.value
    assert value == 3

    arr = np.random.random((3, 12, 10))
    sample = {"not_image": arr, "mask": None}
    output = foo(sample)

    assert output["mask"] is None
    assert list(output.keys()) == ["not_image", "mask"]
    assert torch.allclose(value * torch.from_numpy(arr), output["not_image"])

    # Now wrap without arguments.
    @_wrap_tv
    class Foo2:
        def __init__(self, vars):
            self.vars = vars

        def __call__(self, idx):
            return self.value * idx

        @property
        def value(self):
            return self.vars

    foo2 = Foo2(2)
    value2 = foo2.value

    assert value2 == 2

    sample = {"image": arr, "mask": None}
    output = foo2(sample)

    assert output["mask"] is None
    assert list(output.keys()) == ["image", "mask"]
    assert torch.allclose(value2 * torch.from_numpy(arr), output["image"])


if __name__ == "__main__":
    test__wrap_tv()
