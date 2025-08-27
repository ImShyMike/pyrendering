# pylint: disable=missing-function-docstring,missing-module-docstring

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from pyrendering.graphics import DrawModes, Graphics
from pyrendering.shapes import Circle, Rect, Shape, Triangle
from pyrendering.vectors import Vec2


class StatusCode(Enum):
    """Status codes for engine operations"""

    SUCCESS = 0
    FAILURE = 1
    NOT_FOUND = 2
    INVALID_OPERATION = 3

    def __bool__(self):
        return self == StatusCode.SUCCESS


@dataclass
class ModeShape:
    """Helper dataclass to store mode and shape together"""

    mode: DrawModes
    shape: Shape

    def __iter__(self):
        return iter((self.mode, self.shape))


class Engine:
    """Rendering Engine"""

    def __init__(self, gfx: Graphics):
        self.gfx = gfx
        self.shapes = {}
        self.shape_id_counter = 0

    def add_shape(
        self,
        shape: Shape,
        draw_mode: DrawModes = "fill",
        shape_id: Optional[str] = None,
    ) -> str:
        """Add a shape to the engine

        Args:
            shape (Shape): Shape object.
            draw_mode (DrawModes, optional): Draw mode. Defaults to "fill".
            shape_id (Optional[str], optional): Overwrite the shape ID. Defaults to None.

        Returns:
            str: Shape ID.
        """
        if shape_id is None:
            shape_id = str(self.shape_id_counter)
        self.shapes[shape_id] = (draw_mode, shape)
        self.shape_id_counter += 1
        return shape_id

    def get_shape(self, shape_id: str) -> Optional[ModeShape]:
        """Get a shape by its ID

        Args:
            shape_id (str): Shape ID.

        Returns:
            Optional[ModeShape]: Returns ModeShape or None if not found
        """
        return self.shapes.get(shape_id, None)

    def remove_shape(self, shape_id: str):
        """Remove a shape by its ID

        Args:
            shape_id (str): Shape ID.
        """
        if shape_id in self.shapes:
            del self.shapes[shape_id]

    def render(self):
        """Render all shapes"""
        for draw_mode, shape in self.shapes.values():
            self.gfx.draw(shape, draw_mode=draw_mode)

    def clear(self):
        """Clear all shapes"""
        self.shapes.clear()

    @staticmethod
    def lerp(a: float, b: float, t: float) -> float:
        """Linear interpolation between a and b by t

        Args:
            a (float): Start value
            b (float): End value
            t (float): Time factor (0.0 to 1.0)

        Returns:
            float: Value between a and b based on t
        """
        return a + (b - a) * t

    @staticmethod
    def rotate_point(point: Vec2, center: Vec2, cos_a: float, sin_a: float) -> Vec2:
        """Rotate a point around a center by given cosine and sine of the angle

        Args:
            point (Vec2): Point to rotate
            center (Vec2): Center of rotation
            cos_a (float): Precomputed cosine of the angle
            sin_a (float): Precomputed sine of the angle

        Returns:
            Vec2: Rotated point
        """
        translated = point - center
        rotated = Vec2(
            translated.x * cos_a - translated.y * sin_a,
            translated.x * sin_a + translated.y * cos_a,
        )
        return rotated + center

    def rotate_shape(
        self, shape_id: str, angle: float, center: Optional[Vec2] = None
    ) -> StatusCode:
        """Rotate a shape around a center by a given angle in radians

        Args:
            shape_id (str): Shape ID.
            angle (float): Angle in radians.
            center (Optional[Vec2], optional): Center of rotation. \
                If None, uses the shape's center. Defaults to None.

        Returns:
            StatusCode: Success status
        """
        mode_shape = self.get_shape(shape_id)
        if mode_shape is None:
            return StatusCode.NOT_FOUND

        _, shape = mode_shape

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
                return StatusCode.SUCCESS

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
                return StatusCode.SUCCESS

        return StatusCode.INVALID_OPERATION

    def translate_shape(self, shape_id: str, offset: Vec2) -> StatusCode:
        """Translate a shape by a given offset

        Args:
            shape_id (str): Shape ID.
            offset (Vec2): Offset vector.

        Returns:
            StatusCode: Success status
        """
        mode_shape = self.get_shape(shape_id)
        if mode_shape is None:
            return StatusCode.NOT_FOUND

        _, shape = mode_shape

        if isinstance(shape, Shape):
            if isinstance(shape, Rect):
                shape.p1.position += offset
                shape.p2.position += offset
                shape.p3.position += offset
                shape.p4.position += offset
                return StatusCode.SUCCESS

            if isinstance(shape, Triangle):
                shape.p1.position += offset
                shape.p2.position += offset
                shape.p3.position += offset
                return StatusCode.SUCCESS

            if isinstance(shape, Circle):
                shape.center += offset
                return StatusCode.SUCCESS

        return StatusCode.INVALID_OPERATION

    def move_shape_to(self, shape_id: str, position: Vec2) -> StatusCode:
        """Move a shape to a specific position by translating its center to that position

        Args:
            shape_id (str): Shape ID.
            position (Vec2): Position to move the shape's center to.

        Returns:
            StatusCode: Success status
        """
        mode_shape = self.get_shape(shape_id)
        if mode_shape is None:
            return StatusCode.NOT_FOUND

        _, shape = mode_shape

        if isinstance(shape, Shape):
            if isinstance(shape, Rect):
                center = shape.center
                offset = position - center
                return self.translate_shape(shape_id, offset)

            if isinstance(shape, Triangle):
                center = Vec2(
                    (shape.p1.x + shape.p2.x + shape.p3.x) / 3,
                    (shape.p1.y + shape.p2.y + shape.p3.y) / 3,
                )
                offset = position - center
                return self.translate_shape(shape_id, offset)

            if isinstance(shape, Circle):
                offset = position - shape.center
                shape.center += offset
                return StatusCode.SUCCESS

        return StatusCode.INVALID_OPERATION

    def update_shape(self, shape_id: str, new_shape: Shape) -> StatusCode:
        """Update an existing shape with a new shape

        Args:
            shape_id (str): Old shape ID.
            new_shape (Shape): New shape to replace the old one.

        Returns:
            StatusCode: Success status
        """
        mode_shape = self.get_shape(shape_id)
        if mode_shape is None:
            return StatusCode.NOT_FOUND

        draw_mode, _ = mode_shape
        self.shapes[shape_id] = (draw_mode, new_shape)
        return StatusCode.SUCCESS

    def update_shape_color(self, shape_id: str, new_color) -> StatusCode:
        """Update the color of an existing shape

        Args:
            shape_id (str): Shape ID.
            new_color (Color): New color.

        Returns:
            StatusCode: Success status
        """
        mode_shape = self.get_shape(shape_id)
        if mode_shape is None:
            return StatusCode.NOT_FOUND

        _, shape = mode_shape

        if isinstance(shape, Shape):
            if isinstance(shape, Rect):
                shape.p1.color = new_color
                shape.p2.color = new_color
                shape.p3.color = new_color
                shape.p4.color = new_color
                return StatusCode.SUCCESS

            if isinstance(shape, Triangle):
                shape.p1.color = new_color
                shape.p2.color = new_color
                shape.p3.color = new_color
                return StatusCode.SUCCESS

            if isinstance(shape, Circle):
                shape.color = new_color
                return StatusCode.SUCCESS

        return StatusCode.INVALID_OPERATION

    def update_shape_mode(self, shape_id: str, new_mode: DrawModes) -> StatusCode:
        """Update the draw mode of an existing shape

        Args:
            shape_id (str): Shape ID.
            new_mode (DrawModes): New draw mode.

        Returns:
            StatusCode: Success status
        """
        mode_shape = self.get_shape(shape_id)
        if mode_shape is None:
            return StatusCode.NOT_FOUND

        _, shape = mode_shape
        self.shapes[shape_id] = (new_mode, shape)
        return StatusCode.SUCCESS

    def shape_count(self) -> int:
        """Get the number of shapes managed by the engine

        Returns:
            int: Number of shapes
        """
        return len(self.shapes)

    def list_shape_ids(self):
        """List all shape IDs managed by the engine

        Returns:
            List[str]: List of shape IDs
        """
        return list(self.shapes.keys())

    def shape_exists(self, shape_id: str) -> bool:
        """Check if a shape exists by its ID

        Args:
            shape_id (str): Shape ID.

        Returns:
            bool: True if the shape exists, False otherwise
        """
        return shape_id in self.shapes
