# pylint: disable=missing-function-docstring,missing-module-docstring

from dataclasses import dataclass, field

from pyrendering.vectors import Vec2


@dataclass
class Camera:
    """Camera class"""

    position: Vec2 = field(default_factory=lambda: Vec2(0.0, 0.0))
    zoom: float = 1.0

    def move(self, delta: Vec2):
        self.position += delta

    def set_zoom(self, zoom: float):
        self.zoom = max(0.1, zoom)

    def reset(self):
        self.position = Vec2(0.0, 0.0)
        self.zoom = 1.0
