import pathlib
from argparse import ArgumentTypeError
from unittest.mock import patch

import pytest

from dlup.cli import dir_path, file_path, main


def test_dir_path_valid_directory(tmpdir):
    path = tmpdir.mkdir("subdir")
    assert dir_path(str(path)) == pathlib.Path(path)


def test_dir_path_invalid_directory():
    with pytest.raises(ArgumentTypeError):
        dir_path("/path/which/does/not/exist")


def test_file_path_valid_file(tmpdir):
    path = tmpdir.join("test_file.txt")
    path.write("content")
    assert file_path(str(path)) == pathlib.Path(path)


def test_file_path_invalid_file():
    with pytest.raises(ArgumentTypeError):
        file_path("/path/which/does/not/exist.txt")


def test_file_path_no_need_exists():
    _path = "/path/which/does/not/need/to/exist.txt"
    assert file_path(_path, need_exists=False) == pathlib.Path(_path)


def test_main_no_arguments(capsys):
    with patch("sys.argv", ["dlup"]):
        with pytest.raises(SystemExit):
            main()
