# coding=utf-8


from typing import Optional


class DlupError(Exception):
    pass


class DlupUnsupportedSlideError(DlupError):
    def __init__(self, msg: str, identifier: Optional[str] = None):
        msg = msg if identifier is None else f"slide '{identifier}': " + msg
        super().__init__(self, msg)
