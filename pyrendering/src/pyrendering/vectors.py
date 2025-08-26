# pylint: disable=missing-function-docstring,missing-module-docstring

from dataclasses import dataclass


@dataclass
class Vec2:
    """2D Vector"""

    x: float
    y: float

    def __add__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar) -> "Vec2":
        return Vec2(self.x * scalar, self.y * scalar)

    def length(self) -> float:
        return (self.x**2 + self.y**2) ** 0.5

    def normalize(self) -> "Vec2":
        l = self.length()  # noqa: E741
        return Vec2(self.x / l, self.y / l) if l > 0 else Vec2(0, 0)
