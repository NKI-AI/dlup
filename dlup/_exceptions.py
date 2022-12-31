# coding=utf-8
# Copyright (c) dlup contributors
from __future__ import annotations


class DlupError(Exception):
    pass


class UnsupportedSlideError(DlupError):
    def __init__(self, msg: str, identifier: str | None = None):
        msg = msg if identifier is None else f"slide '{identifier}': " + msg
        super().__init__(self, msg)


class AnnotationError(DlupError):
    def __init__(self, msg: str):
        super().__init__(self, msg)
