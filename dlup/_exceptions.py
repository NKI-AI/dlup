# Copyright (c) dlup contributors
"""Custom exceptions for dlup."""
from __future__ import annotations


class DlupError(Exception):
    """Base class for exceptions in dlup."""

    pass


class UnsupportedSlideError(DlupError):
    """Raised when a slide is not supported."""

    def __init__(self, msg: str, identifier: str | None = None):
        msg = msg if identifier is None else f"slide '{identifier}': " + msg
        super().__init__(self, msg)


class AnnotationError(DlupError):
    """Raised when an annotation is invalid."""

    def __init__(self, msg: str):
        super().__init__(self, msg)
