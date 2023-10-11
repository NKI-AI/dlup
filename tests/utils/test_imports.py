# Copyright (c) dlup contributors

from dlup.utils.imports import _module_available


def test_module_available_existing_module():
    assert _module_available("os") is True


def test_module_available_non_existing_module():
    assert _module_available("non.existing.module") is False


def test_module_available_submodule():
    assert _module_available("pytest") is True


def test_module_available_non_existing_submodule():
    assert _module_available("os.non_existing_submodule") is False


def test_module_available_raises_module_not_found_error(mocker):
    mocker.patch("importlib.util.find_spec", side_effect=ModuleNotFoundError)
    assert _module_available("bla") is False
