# coding=utf-8
# Copyright (c) dlup contributors
"""
Exceptions for dlup.
"""
from __future__ import annotations


class DlupError(Exception):
    """Base class for dlup exceptions."""



class UnsupportedSlideError(DlupError):
    """Raised when a slide cannot be read by any backend."""

    def __init__(self, msg: str, identifier: str | None = None):
        msg = msg if identifier is None else f"slide '{identifier}': " + msg
        super().__init__(self, msg)


class AnnotationError(DlupError):
    """Raised when an annotation is invalid."""
    def __init__(self, msg: str):
        super().__init__(self, msg)
