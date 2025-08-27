# pylint: disable=missing-function-docstring,missing-module-docstring

from typing import Optional

import numpy as np

from pyrendering.color import Color
from pyrendering.graphics import Graphics, GraphicsContext
from pyrendering.shapes import Circle, Rect, Shape, Triangle
from pyrendering.vectors import Vec2, Vec3


class Engine:
    """Rendering Engine"""

    def __init__(self, gfx: Graphics):
        self.gfx = gfx
        self.shapes = {}
        self.shape_id_counter = 0

    def add_shape(self, shape: Shape, shape_id: Optional[str] = None) -> str:
        if shape_id is None:
            shape_id = str(self.shape_id_counter)
        self.shapes[shape_id] = shape
        self.shape_id_counter += 1
        return shape_id

    def get_shape(self, shape_id: str) -> Optional[Shape]:
        return self.shapes.get(shape_id, None)

    def remove_shape(self, shape_id: str):
        if shape_id in self.shapes:
            del self.shapes[shape_id]

    def render(self):
        for shape in self.shapes.values():
            self.gfx.draw(shape)

    def clear(self):
        self.shapes.clear()

    @staticmethod
    def rotate_point(point: Vec2, center: Vec2, cos_a: float, sin_a: float) -> Vec2:
        translated = point - center
        rotated = Vec2(
            translated.x * cos_a - translated.y * sin_a,
            translated.x * sin_a + translated.y * cos_a,
        )
        return rotated + center

    def rotate_shape(self, shape_id: str, angle: float, center: Optional[Vec2] = None) -> bool:
        shape = self.get_shape(shape_id)
        if shape is None:
            return False

        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        if isinstance(shape, Shape):
            if isinstance(shape, Rect):
                if not center:
                    center = shape.center
                shape.p1.position = Engine.rotate_point(
                    shape.p1.position, center, cos_a, sin_a
                )
                shape.p2.position = Engine.rotate_point(
                    shape.p2.position, center, cos_a, sin_a
                )
                shape.p3.position = Engine.rotate_point(
                    shape.p3.position, center, cos_a, sin_a
                )
                shape.p4.position = Engine.rotate_point(
                    shape.p4.position, center, cos_a, sin_a
                )
                return True

            if isinstance(shape, Triangle):
                if not center:
                    center = Vec2(
                        (shape.p1.x + shape.p2.x + shape.p3.x) / 3,
                        (shape.p1.y + shape.p2.y + shape.p3.y) / 3,
                    )
                shape.p1.position = Engine.rotate_point(
                    shape.p1.position, center, cos_a, sin_a
                )
                shape.p2.position = Engine.rotate_point(
                    shape.p2.position, center, cos_a, sin_a
                )
                shape.p3.position = Engine.rotate_point(
                    shape.p3.position, center, cos_a, sin_a
                )
                return True
        return False
