# coding=utf-8
# Copyright (c) dlup contributors
import ast
import functools
import importlib
from typing import Any, Callable, Union


class DlupClass:
    # TODO: Have a better debugging output
    def __repr__(self):
        repr_string = self.__class__.__name__
        return repr_string


# Taken from:
# https://github.com/PyTorchLightning/pytorch-lightning-bolts/blob/41bd36cc5e02a03753776078c3fcfd6226adf78b/pl_bolts/utils/__init__.py
# Under Apache 2.0 license
def _module_available(module_path: str) -> bool:
    """Testing if given module is avalaible in your env

    >>> _module_available("os")
    True
    >>> _module_available("bla.bla")
    False
    """
    # todo: find a better way than try / except
    try:
        mods = module_path.split(".")
        assert mods, "nothing given to test"
        # it has to be tested as per partets
        for i in range(len(mods)):
            module_path = ".".join(mods[: i + 1])
            if importlib.util.find_spec(module_path) is None:  # type: ignore
                return False
        return True
    except AttributeError:
        return False


_HISTOMICSTK_AVAILABLE: bool = _module_available("histomicstk")
_TORCHVISION_AVAILABLE: bool = _module_available("torchvision")


# Also in https://github.com/directgroup/direct/blob/master/direct/utils/__init__.py
def str_to_class(module_name: str, function_name: str) -> Union[object, Any, Callable]:
    """
    Convert a string to a class

    Based on: https://stackoverflow.com/a/1176180/576363
    Also support function arguments, e.g. ifft(dim=2) will be parsed as a partial and return ifft where dim has been
    set to 2.

    Example
    -------
    >>> def mult(f, mul=2):
    >>>    return f*mul
    >>> str_to_class(".", "mult(mul=4)")
    >>> str_to_class(".", "mult(mul=4)")
    will return a function which multiplies the input times 4, while
    >>> str_to_class(".", "mult")
    just returns the function itself.
    Parameters
    ----------
    function_name :
    module_name : str
        e.g. direct.data.transforms
    class_name : str
        e.g. Identity
    Returns
    -------
    object
    """
    tree = ast.parse(function_name)
    func_call = tree.body[0].value  # type: ignore
    args = [ast.literal_eval(arg) for arg in func_call.args] if hasattr(func_call, "args") else []
    kwargs = (
        {arg.arg: ast.literal_eval(arg.value) for arg in func_call.keywords} if hasattr(func_call, "keywords") else {}
    )

    # Load the module, will raise ModuleNotFoundError if module cannot be loaded.
    module = importlib.import_module(module_name)

    if not args and not kwargs:
        return getattr(module, function_name)
    return functools.partial(getattr(module, func_call.func.id), *args, **kwargs)
