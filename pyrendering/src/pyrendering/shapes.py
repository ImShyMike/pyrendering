# pylint: disable=missing-function-docstring,missing-module-docstring

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from pyrendering.color import Color
from pyrendering.vectors import Point, Vec2


class Shape:
    """Base shape class"""


@dataclass
class Rect(Shape):
    """Rectangle class"""

    p1: Point
    p2: Point
    p3: Point
    p4: Point

    def __contains__(self, point: Vec2) -> bool:
        return self.contains_point(point)

    @staticmethod
    def from_dimensions(
        x: float, y: float, width: float, height: float, color: Color = Color()
    ):
        return Rect(
            Point(Vec2(x, y), color),
            Point(Vec2(x + width, y), color),
            Point(Vec2(x + width, y + height), color),
            Point(Vec2(x, y + height), color),
        )

    @property
    def center(self) -> Vec2:
        return Vec2(
            (self.p1.x + self.p2.x + self.p3.x + self.p4.x) / 4,
            (self.p1.y + self.p2.y + self.p3.y + self.p4.y) / 4,
        )

    @property
    def area(self) -> float:
        # Using the Shoelace formula for quadrilateral
        return (
            abs(
                (self.p1.x * self.p2.y - self.p2.x * self.p1.y)
                + (self.p2.x * self.p3.y - self.p3.x * self.p2.y)
                + (self.p3.x * self.p4.y - self.p4.x * self.p3.y)
                + (self.p4.x * self.p1.y - self.p1.x * self.p4.y)
            )
            / 2
        )

    def contains_point(self, point: Vec2) -> bool:
        # Check if the point is inside the quadrilateral using triangles
        def triangle_area(p1, p2, p3):
            return abs(
                (p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y))
                / 2.0
            )

        total_area = self.area
        area1 = triangle_area(point, self.p1, self.p2)
        area2 = triangle_area(point, self.p2, self.p3)
        area3 = triangle_area(point, self.p3, self.p4)
        area4 = triangle_area(point, self.p4, self.p1)

        return abs(total_area - (area1 + area2 + area3 + area4)) < 1e-6


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
        return Rect.from_dimensions(
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
                np.clip(
                    self.center.x,
                    min(rect.p1.x, rect.p2.x, rect.p3.x, rect.p4.x),
                    max(rect.p1.x, rect.p2.x, rect.p3.x, rect.p4.x),
                ),
                np.clip(
                    self.center.y,
                    min(rect.p1.y, rect.p2.y, rect.p3.y, rect.p4.y),
                    max(rect.p1.y, rect.p2.y, rect.p3.y, rect.p4.y),
                ),
            ]
        )
        return bool(np.linalg.norm(closest - self.center.data) <= self.radius)


@dataclass
class Triangle(Shape):
    """Triangle class"""

    p1: Point
    p2: Point
    p3: Point

    def __contains__(self, point: Vec2) -> bool:
        return self.contains_point(point)

    @property
    def area(self) -> float:
        # Using the first three points for the triangle area
        return abs(
            (
                self.p1.x * (self.p2.y - self.p3.y)
                + self.p2.x * (self.p3.y - self.p1.y)
                + self.p3.x * (self.p1.y - self.p2.y)
            )
            / 2.0
        )

    @property
    def bounding_rect(self) -> Rect:
        min_x = min(self.p1.x, self.p2.x, self.p3.x)
        max_x = max(self.p1.x, self.p2.x, self.p3.x)
        min_y = min(self.p1.y, self.p2.y, self.p3.y)
        max_y = max(self.p1.y, self.p2.y, self.p3.y)
        return Rect(
            Point(Vec2(min_x, min_y)),
            Point(Vec2(max_x, min_y)),
            Point(Vec2(max_x, max_y)),
            Point(Vec2(min_x, max_y)),
        )

    def contains_point(self, point: Vec2) -> bool:
        # Barycentric technique for the first three points
        def sign(p1, p2, p3):
            return (p1.x - p3.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p3.y)

        b1 = sign(point, self.p1, self.p2) < 0.0
        b2 = sign(point, self.p2, self.p3) < 0.0
        b3 = sign(point, self.p3, self.p1) < 0.0

        return b1 == b2 == b3


@dataclass
class Texture(Shape):
    """Texture class"""

    v1: Vec2
    v2: Vec2
    v3: Vec2
    v4: Vec2
    path: Path
    color: Color = field(default_factory=Color)
    _texture: Optional[object] = field(default=None, init=False)

    def __post_init__(self):
        if not self.path.exists() or not self.path.is_file():
            raise FileNotFoundError(f"Texture file not found: {self.path}")

    def load_texture(self, ctx):
        """Load the texture if not already loaded"""
        if not self._texture:
            try:
                img = Image.open(self.path)
                if img.mode != "RGBA":
                    img = img.convert("RGBA")

                # Create texture
                self._texture = ctx.texture(img.size, 4)
                self._texture.write(img.tobytes())  # type: ignore
                img.close()
            except Exception as e:
                print(f"Error loading texture {self.path}: {e}")
                raise RuntimeError(f"Failed to load texture {self.path}: {e}") from e

        return self._texture

    @staticmethod
    def from_path(
        x: float,
        y: float,
        width: float,
        height: float,
        path: Path,
        color: Color = Color(),
    ) -> "Texture":
        """Create a texture rectangle from dimensions and path"""
        return Texture(
            Vec2(x, y),
            Vec2(x + width, y),
            Vec2(x + width, y + height),
            Vec2(x, y + height),
            color=color,
            path=path,
        )
