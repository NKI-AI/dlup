# coding=utf-8


class DlupError(Exception):
    pass


class DlupUnsupportedSlideError(DlupError):
    def __init__(self, msg, *args, identifier=None, **kwargs):
        msg = f"slide '{identifier}': " + msg if identifier is None else msg
        super().__init__(self, msg, *args, **kwargs)
