# pylint: disable=missing-function-docstring,missing-module-docstring

from dataclasses import dataclass
from typing import Tuple, Union

HexLike = Union[str, int]
RGB = Tuple[int, int, int]
RGBA = Tuple[int, int, int, int]
NormalRGB = Tuple[float, float, float]
NormalRGBA = Tuple[float, float, float, float]


@dataclass
class Color:
    """Color class"""

    r: int = 255
    g: int = 255
    b: int = 255
    a: int = 255

    def as_tuple(self) -> RGBA:
        return (self.r, self.g, self.b, self.a)

    def as_rgb_tuple(self) -> RGB:
        return (self.r, self.g, self.b)

    def as_normalized(self) -> NormalRGBA:
        return (self.r / 255, self.g / 255, self.b / 255, self.a / 255)

    def as_rgb_normalized(self) -> NormalRGB:
        return (self.r / 255, self.g / 255, self.b / 255)

    def as_hex(self) -> HexLike:
        r, g, b = (
            self.r & 0xFF,
            self.g & 0xFF,
            self.b & 0xFF,
        )
        return f"#{r:02x}{g:02x}{b:02x}"

    def as_hex8(self) -> HexLike:
        r, g, b, a = self.r & 0xFF, self.g & 0xFF, self.b & 0xFF, self.a & 0xFF
        return f"#{r:02x}{g:02x}{b:02x}{a:02x}"

    def with_alpha(self, alpha: int) -> "Color":
        return Color(self.r, self.g, self.b, alpha & 0xFF)

    @staticmethod
    def from_rgb(r: int, g: int, b: int) -> "Color":
        return Color(r & 0xFF, g & 0xFF, b & 0xFF, 255)

    @staticmethod
    def from_rgba(r: int, g: int, b: int, a: int) -> "Color":
        return Color(r & 0xFF, g & 0xFF, b & 0xFF, a & 0xFF)

    @staticmethod
    def from_hex(value: HexLike) -> "Color":
        if not isinstance(value, int):
            value = int(value.replace("#", ""), 16)
        return Color((value >> 16) & 0xFF, (value >> 8) & 0xFF, value & 0xFF, 255)

    @staticmethod
    def from_hex8(value: HexLike) -> "Color":
        if not isinstance(value, int):
            value = int(value.replace("#", ""), 16)
        return Color(
            (value >> 24), (value >> 16) & 0xFF, (value >> 8) & 0xFF, value & 0xFF
        )
