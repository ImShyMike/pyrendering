# pylint: disable=missing-function-docstring,missing-module-docstring

from dataclasses import dataclass

import numpy as np

from pyrendering.vectors import Vec2


@dataclass
class Rect:
    """Rectangle class"""

    x: float
    y: float
    width: float
    height: float

    def __contains__(self, point: Vec2) -> bool:
        return self.contains_point(point)

    @property
    def center(self) -> Vec2:
        return Vec2(self.x + self.width / 2, self.y + self.height / 2)

    @property
    def area(self) -> float:
        return self.width * self.height

    def contains_point(self, point: Vec2) -> bool:
        return (
            self.x <= point.x <= self.x + self.width
            and self.y <= point.y <= self.y + self.height
        )


@dataclass
class Circle:
    """Circle class"""

    center: Vec2
    radius: float

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
        return (point - self.center).length() <= self.radius

    def intersects_rect(self, rect: Rect) -> bool:
        closest_x = max(rect.x, min(self.center.x, rect.x + rect.width))
        closest_y = max(rect.y, min(self.center.y, rect.y + rect.height))
        distance = Vec2(closest_x, closest_y) - self.center
        return distance.length() <= self.radius
