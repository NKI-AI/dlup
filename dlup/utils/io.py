# coding=utf-8
# Copyright (c) dlup contributors

# From: https://github.com/directgroup/direct/blob/master/direct/utils/io.py

import json
import pathlib
import warnings
from typing import Dict, List, Union

import numpy as np
import torch


def read_json(filename: Union[Dict, str, pathlib.Path]) -> Dict:
    """
    Read file and output dict, or take dict and output dict.


    Parameters
    ----------
    filename : Union[Dict, str, pathlib.Path]


    Returns
    -------
    dict

    """
    if isinstance(filename, dict):
        return filename

    with open(filename, "r") as file:
        data = json.load(file)
    return data


class JsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, pathlib.PosixPath):
            obj = str(obj)

        if isinstance(obj, torch.Tensor):
            obj = obj.numpy()

        if isinstance(obj, np.ndarray):
            if obj.size > 10e4:
                warnings.warn(
                    f"Trying to JSON serialize a very large array of size {obj.size}. "
                    f"Reconsider doing this differently"
                )
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def write_json(filename: Union[str, pathlib.Path], data: Dict, indent=2) -> None:
    """
    Write dict data to filename.

    Parameters
    ----------
    filename : Path or str
    data : dict
    indent: int

    Returns
    -------
    None
    """
    with open(filename, "w") as f:
        json.dump(data, f, indent=indent, cls=JsonEncoder)


def read_list(filename: Union[List, str, pathlib.Path]) -> List:
    """
    Read file and output list, or take list and output list.

    Parameters
    ----------
    filename : Union[[list, str, pathlib.Path]]
        Input text file or list

    Returns
    -------
    list
    """
    if isinstance(filename, (pathlib.Path, str)):
        with open(filename) as f:
            filter_fns = f.readlines()
        return [_.strip() for _ in filter_fns if not _.startswith("#")]
    return filename


def write_list(fn: Union[str, pathlib.Path], data) -> None:
    """
    Write list line by line to file.

    Parameters
    ----------
    fn : Union[[list, str, pathlib.Path]]
        Input text file or list
    data : list or tuple
    Returns
    -------
    None
    """
    with open(fn, "w") as file:
        for line in data:
            file.write(f"{line}\n")
