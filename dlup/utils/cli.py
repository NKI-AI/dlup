# coding=utf-8
# Copyright (c) dlup contributors
"""Utilities for the CLI interface."""
import argparse
import pathlib


def dir_path(path: str) -> pathlib.Path:
    """Check if the path is a valid directory.
    Parameters
    ----------
    path : str

    Returns
    -------
    pathlib.Path
        The path as a pathlib.Path object.
    """
    _path = pathlib.Path(path)
    if _path.is_dir():
        return _path
    raise argparse.ArgumentTypeError(f"{path} is not a valid directory.")


def file_path(path: str, need_exists: bool = True) -> pathlib.Path:
    """Check if the path is a valid file.
    Parameters
    ----------
    path : str
    need_exists : bool

    Returns
    -------
    pathlib.Path
        The path as a pathlib.Path object.
    """
    _path = pathlib.Path(path)
    if need_exists:
        if _path.is_file():
            return _path
        raise argparse.ArgumentTypeError(f"{path} is not a valid file.")
    return _path
