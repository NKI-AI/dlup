from typing import NamedTuple, Optional


class TagAttribute(NamedTuple):
    label: str
    color: Optional[tuple[int, int, int]]


class SlideTag(NamedTuple):
    attributes: Optional[list[TagAttribute]]
    label: str
    color: Optional[tuple[int, int, int]]
