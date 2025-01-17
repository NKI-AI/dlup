from dataclasses import dataclass, field
from typing import List, Optional

from xsdata.models.datatype import XmlDate


@dataclass
class BasePolygonType:
    exterior: Optional["BasePolygonType.Exterior"] = field(
        default=None,
        metadata={
            "name": "Exterior",
            "type": "Element",
            "required": True,
        },
    )
    interiors: Optional["BasePolygonType.Interiors"] = field(
        default=None,
        metadata={
            "name": "Interiors",
            "type": "Element",
        },
    )

    @dataclass
    class Exterior:
        point: List["BasePolygonType.Exterior.Point"] = field(
            default_factory=list,
            metadata={
                "name": "Point",
                "type": "Element",
                "min_occurs": 1,
            },
        )

        @dataclass
        class Point:
            x: Optional[float] = field(
                default=None,
                metadata={
                    "type": "Attribute",
                    "required": True,
                },
            )
            y: Optional[float] = field(
                default=None,
                metadata={
                    "type": "Attribute",
                    "required": True,
                },
            )

    @dataclass
    class Interiors:
        interior: List["BasePolygonType.Interiors.Interior"] = field(
            default_factory=list,
            metadata={
                "name": "Interior",
                "type": "Element",
                "min_occurs": 1,
            },
        )

        @dataclass
        class Interior:
            point: List["BasePolygonType.Interiors.Interior.Point"] = field(
                default_factory=list,
                metadata={
                    "name": "Point",
                    "type": "Element",
                    "min_occurs": 1,
                },
            )

            @dataclass
            class Point:
                x: Optional[float] = field(
                    default=None,
                    metadata={
                        "type": "Attribute",
                        "required": True,
                    },
                )
                y: Optional[float] = field(
                    default=None,
                    metadata={
                        "type": "Attribute",
                        "required": True,
                    },
                )


@dataclass
class Metadata:
    image_id: Optional[str] = field(
        default=None,
        metadata={
            "name": "ImageID",
            "type": "Element",
            "required": True,
        },
    )
    description: Optional[str] = field(
        default=None,
        metadata={
            "name": "Description",
            "type": "Element",
        },
    )
    version: Optional[str] = field(
        default=None,
        metadata={
            "name": "Version",
            "type": "Element",
            "required": True,
        },
    )
    authors: Optional["Metadata.Authors"] = field(
        default=None,
        metadata={
            "name": "Authors",
            "type": "Element",
        },
    )
    date_created: Optional[XmlDate] = field(
        default=None,
        metadata={
            "name": "DateCreated",
            "type": "Element",
            "required": True,
        },
    )
    software: Optional[str] = field(
        default=None,
        metadata={
            "name": "Software",
            "type": "Element",
            "required": True,
        },
    )

    @dataclass
    class Authors:
        author: List[str] = field(
            default_factory=list,
            metadata={
                "name": "Author",
                "type": "Element",
                "min_occurs": 1,
            },
        )


@dataclass
class RectangleType:
    x_min: Optional[float] = field(
        default=None,
        metadata={
            "name": "xMin",
            "type": "Attribute",
            "required": True,
        },
    )
    y_min: Optional[float] = field(
        default=None,
        metadata={
            "name": "yMin",
            "type": "Attribute",
            "required": True,
        },
    )
    x_max: Optional[float] = field(
        default=None,
        metadata={
            "name": "xMax",
            "type": "Attribute",
            "required": True,
        },
    )
    y_max: Optional[float] = field(
        default=None,
        metadata={
            "name": "yMax",
            "type": "Attribute",
            "required": True,
        },
    )


@dataclass
class Tag:
    attribute: List["Tag.Attribute"] = field(
        default_factory=list,
        metadata={
            "name": "Attribute",
            "type": "Element",
        },
    )
    text: Optional[str] = field(
        default=None,
        metadata={
            "name": "Text",
            "type": "Element",
        },
    )
    label: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        },
    )
    color: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"#[0-9a-fA-F]{6}",
        },
    )

    @dataclass
    class Attribute:
        value: str = field(
            default="",
            metadata={
                "required": True,
            },
        )
        color: Optional[str] = field(
            default=None,
            metadata={
                "type": "Attribute",
                "pattern": r"#[0-9a-fA-F]{6}",
            },
        )


@dataclass
class BoundingBoxType(RectangleType):
    label: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )
    color: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"#[0-9a-fA-F]{6}",
        },
    )
    index: Optional[int] = field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )


@dataclass
class BoxType(RectangleType):
    label: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        },
    )
    color: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"#[0-9a-fA-F]{6}",
        },
    )


@dataclass
class MultiPolygonType:
    polygon: List[BasePolygonType] = field(
        default_factory=list,
        metadata={
            "name": "Polygon",
            "type": "Element",
            "min_occurs": 1,
        },
    )
    label: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        },
    )
    color: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"#[0-9a-fA-F]{6}",
        },
    )
    index: Optional[int] = field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )
    order: Optional[int] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        },
    )


@dataclass
class RegionBoxType(RectangleType):
    label: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        },
    )
    index: Optional[int] = field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )
    order: Optional[int] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        },
    )


@dataclass
class RegionMultiPolygonType:
    polygon: List[BasePolygonType] = field(
        default_factory=list,
        metadata={
            "name": "Polygon",
            "type": "Element",
            "min_occurs": 1,
        },
    )
    label: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        },
    )
    index: Optional[int] = field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )
    order: Optional[int] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        },
    )


@dataclass
class RegionPolygonType(BasePolygonType):
    label: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        },
    )
    index: Optional[int] = field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )
    order: Optional[int] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        },
    )


@dataclass
class StandalonePolygonType(BasePolygonType):
    label: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        },
    )
    color: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"#[0-9a-fA-F]{6}",
        },
    )
    index: Optional[int] = field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )
    order: Optional[int] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        },
    )


@dataclass
class Tags:
    tag: List[Tag] = field(
        default_factory=list,
        metadata={
            "name": "Tag",
            "type": "Element",
            "min_occurs": 1,
        },
    )


@dataclass
class Geometries:
    polygon: List[StandalonePolygonType] = field(
        default_factory=list,
        metadata={
            "name": "Polygon",
            "type": "Element",
        },
    )
    multi_polygon: List[MultiPolygonType] = field(
        default_factory=list,
        metadata={
            "name": "MultiPolygon",
            "type": "Element",
        },
    )
    box: List[BoxType] = field(
        default_factory=list,
        metadata={
            "name": "Box",
            "type": "Element",
        },
    )
    bounding_box: Optional[BoundingBoxType] = field(
        default=None,
        metadata={
            "name": "BoundingBox",
            "type": "Element",
        },
    )
    point: List["Geometries.Point"] = field(
        default_factory=list,
        metadata={
            "name": "Point",
            "type": "Element",
        },
    )
    multi_point: List["Geometries.MultiPoint"] = field(
        default_factory=list,
        metadata={
            "name": "MultiPoint",
            "type": "Element",
        },
    )

    @dataclass
    class Point:
        x: Optional[float] = field(
            default=None,
            metadata={
                "type": "Attribute",
                "required": True,
            },
        )
        y: Optional[float] = field(
            default=None,
            metadata={
                "type": "Attribute",
                "required": True,
            },
        )
        label: Optional[str] = field(
            default=None,
            metadata={
                "type": "Attribute",
                "required": True,
            },
        )
        color: Optional[str] = field(
            default=None,
            metadata={
                "type": "Attribute",
                "pattern": r"#[0-9a-fA-F]{6}",
            },
        )

    @dataclass
    class MultiPoint:
        point: List["Geometries.MultiPoint.Point"] = field(
            default_factory=list,
            metadata={
                "name": "Point",
                "type": "Element",
                "min_occurs": 1,
            },
        )
        label: Optional[str] = field(
            default=None,
            metadata={
                "type": "Attribute",
                "required": True,
            },
        )
        color: Optional[str] = field(
            default=None,
            metadata={
                "type": "Attribute",
                "pattern": r"#[0-9a-fA-F]{6}",
            },
        )
        index: Optional[int] = field(
            default=None,
            metadata={
                "type": "Attribute",
            },
        )

        @dataclass
        class Point:
            x: Optional[float] = field(
                default=None,
                metadata={
                    "type": "Attribute",
                    "required": True,
                },
            )
            y: Optional[float] = field(
                default=None,
                metadata={
                    "type": "Attribute",
                    "required": True,
                },
            )


@dataclass
class RegionsOfInterest:
    polygon: List[RegionPolygonType] = field(
        default_factory=list,
        metadata={
            "name": "Polygon",
            "type": "Element",
        },
    )
    multi_polygon: List[RegionMultiPolygonType] = field(
        default_factory=list,
        metadata={
            "name": "MultiPolygon",
            "type": "Element",
        },
    )
    box: List[RegionBoxType] = field(
        default_factory=list,
        metadata={
            "name": "Box",
            "type": "Element",
        },
    )


@dataclass
class DlupAnnotations:
    metadata: Optional[Metadata] = field(
        default=None,
        metadata={
            "name": "Metadata",
            "type": "Element",
            "required": True,
        },
    )
    tags: Optional[Tags] = field(
        default=None,
        metadata={
            "name": "Tags",
            "type": "Element",
        },
    )
    geometries: Optional[Geometries] = field(
        default=None,
        metadata={
            "name": "Geometries",
            "type": "Element",
            "required": True,
        },
    )
    regions_of_interest: Optional[RegionsOfInterest] = field(
        default=None,
        metadata={
            "name": "RegionsOfInterest",
            "type": "Element",
        },
    )
    version: str = field(
        init=False,
        default="1.0",
        metadata={
            "type": "Attribute",
        },
    )
