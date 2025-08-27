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

    def unpack(self):
        return self.data[0], self.data[1]

    def to_point(self, color: Color = Color()) -> "Point":
        return Point(self, color)


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


@dataclass
class Vec3:
    """3D Vector"""

    data: np.ndarray

    def __init__(self, x: float, y: float, z: float):
        self.data = np.array([x, y, z], dtype=float)

    @property
    def x(self) -> float:
        return self.data[0]

    @property
    def y(self) -> float:
        return self.data[1]

    @property
    def z(self) -> float:
        return self.data[2]

    def __add__(self, other: "Vec3") -> "Vec3":
        return Vec3(*(self.data + other.data))

    def __sub__(self, other: "Vec3") -> "Vec3":
        return Vec3(*(self.data - other.data))

    def __mul__(self, scalar) -> "Vec3":
        return Vec3(*(self.data * scalar))

    def length(self) -> float:
        return float(np.linalg.norm(self.data))

    def normalize(self) -> "Vec3":
        length = self.length()
        return Vec3(*(self.data / length)) if length > 0 else Vec3(0, 0, 0)

    def cross(self, other: "Vec3") -> "Vec3":
        cross_product = np.cross(self.data, other.data)
        return Vec3(*cross_product)

    def dot(self, other: "Vec3") -> float:
        return float(np.dot(self.data, other.data))

    def unpack(self):
        return self.data[0], self.data[1], self.data[2]


@dataclass
class Point3D:
    """3D Point class"""

    position: Vec3
    color: Color = field(default_factory=Color)

    @property
    def x(self) -> float:
        return self.position.x

    @property
    def y(self) -> float:
        return self.position.y

    @property
    def z(self) -> float:
        return self.position.z

    def unpack(self):
        return self.position.x, self.position.y, self.position.z, self.color


@dataclass
class Matrix:
    """Matrix class"""

    data: np.ndarray

    def __init__(self, data: np.ndarray):
        self.data = data

    def __matmul__(self, other: "Matrix") -> "Matrix":
        return Matrix(np.matmul(self.data, other.data))

    def transpose(self) -> "Matrix":
        return Matrix(self.data.T)

    def inverse(self) -> "Matrix":
        return Matrix(np.linalg.inv(self.data))

    @staticmethod
    def rotation_x(angle: float) -> "Matrix":
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        return Matrix(np.array([[1, 0, 0], [0, cos_a, -sin_a], [0, sin_a, cos_a]]))

    @staticmethod
    def rotation_y(angle: float) -> "Matrix":
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        return Matrix(np.array([[cos_a, 0, sin_a], [0, 1, 0], [-sin_a, 0, cos_a]]))

    @staticmethod
    def rotation_z(angle: float) -> "Matrix":
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        return Matrix(np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]]))

    def unpack(self):
        return self.data
