# pylint: disable=missing-function-docstring,missing-module-docstring

from dataclasses import dataclass, field

import numpy as np

from pyrendering.color import Color
from pyrendering.vectors import Vec2


class Shape:
    """Base shape class"""


@dataclass
class Rect(Shape):
    """Rectangle class"""

    x: float
    y: float
    width: float
    height: float
    color: Color = field(default_factory=Color)
    filled: bool = True

    def __contains__(self, point: Vec2) -> bool:
        return self.contains_point(point)

    def to_rounded(self, angle: float = 45.0) -> "RoundedRect":
        return RoundedRect(self.x, self.y, self.width, self.height, radius=angle)

    @property
    def center(self) -> Vec2:
        return Vec2(self.x + self.width / 2, self.y + self.height / 2)

    @property
    def area(self) -> float:
        return self.width * self.height

    def contains_point(self, point: Vec2) -> bool:
        return bool(
            np.all(
                [
                    self.x <= point.x <= self.x + self.width,
                    self.y <= point.y <= self.y + self.height,
                ]
            )
        )


@dataclass
class RoundedRect(Rect):
    """Rounded Rectangle class"""

    radius: float = 0.0

    @property
    def bounding_rect(self) -> Rect:
        return Rect(self.x, self.y, self.width, self.height)

    @property
    def area(self) -> float:
        corner_area = (1 - np.pi / 4) * (self.radius**2)
        return self.width * self.height - 4 * corner_area

    def contains_point(self, point: Vec2) -> bool:
        if super().contains_point(point):
            corners = np.array(
                [
                    [self.x + self.radius, self.y + self.radius],
                    [self.x + self.width - self.radius, self.y + self.radius],
                    [self.x + self.radius, self.y + self.height - self.radius],
                    [
                        self.x + self.width - self.radius,
                        self.y + self.height - self.radius,
                    ],
                ]
            )
            distances = np.linalg.norm(corners - point.data, axis=1)
            if np.any(distances <= self.radius):
                return True

            if (
                self.x + self.radius <= point.x <= self.x + self.width - self.radius
            ) or (
                self.y + self.radius <= point.y <= self.y + self.height - self.radius
            ):
                return True
        return False


@dataclass
class Circle(Shape):
    """Circle class"""

    center: Vec2
    radius: float
    color: Color = field(default_factory=Color)
    segments: int = 32

    def __contains__(self, point: Vec2) -> bool:
        return self.contains_point(point)

    @property
    def diameter(self) -> float:
        return self.radius * 2

    @property
    def circumference(self) -> float:
        return 2 * np.pi * self.radius

    @property
    def area(self) -> float:
        return np.pi * (self.radius**2)

    @property
    def bounding_rect(self) -> Rect:
        return Rect(
            self.center.x - self.radius,
            self.center.y - self.radius,
            self.diameter,
            self.diameter,
        )

    def contains_point(self, point: Vec2) -> bool:
        return bool(np.linalg.norm(point.data - self.center.data) <= self.radius)

    def intersects_rect(self, rect: Rect) -> bool:
        closest = np.array(
            [
                np.clip(self.center.x, rect.x, rect.x + rect.width),
                np.clip(self.center.y, rect.y, rect.y + rect.height),
            ]
        )
        return bool(np.linalg.norm(closest - self.center.data) <= self.radius)
