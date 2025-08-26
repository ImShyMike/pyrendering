# pylint: disable=missing-function-docstring,missing-module-docstring

from dataclasses import dataclass, field

import numpy as np

from pyrendering.color import Color


@dataclass
class Vec2:
    """2D Vector"""

    data: np.ndarray

    def __init__(self, x: float, y: float):
        self.data = np.array([x, y], dtype=float)

    @property
    def x(self) -> float:
        return self.data[0]

    @property
    def y(self) -> float:
        return self.data[1]

    def __add__(self, other: "Vec2") -> "Vec2":
        return Vec2(*(self.data + other.data))

    def __sub__(self, other: "Vec2") -> "Vec2":
        return Vec2(*(self.data - other.data))

    def __mul__(self, scalar) -> "Vec2":
        return Vec2(*(self.data * scalar))

    def length(self) -> float:
        return float(np.linalg.norm(self.data))

    def normalize(self) -> "Vec2":
        length = self.length()
        return Vec2(*(self.data / length)) if length > 0 else Vec2(0, 0)


@dataclass
class Point:
    """Point class"""

    position: Vec2
    color: Color = field(default_factory=Color)

    @property
    def x(self) -> float:
        return self.position.x

    @property
    def y(self) -> float:
        return self.position.y

    def unpack(self):
        return self.position.x, self.position.y, self.color
