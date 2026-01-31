# Copyright 2024 AI for Oncology Research Group. All Rights Reserved.
# Copyright 2024 Jonas Teuwen. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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


class SlideError(RuntimeError):
    """Raised when an error occurs with a slide."""

    def __init__(self, msg: str, identifier: str | None = None):
        msg = msg if identifier is None else f"slide '{identifier}': " + msg
        super().__init__(self, msg)


class AnnotationError(DlupError):
    """Raised when an annotation is invalid."""

    def __init__(self, msg: str):
        super().__init__(self, msg)
