# pylint: disable=missing-function-docstring,missing-module-docstring

# this is very broken and im not sure im going to implement it

import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import List

import numpy as np

from pyrendering.color import Color
from pyrendering.graphics import Graphics
from pyrendering.shapes import Shape, Triangle
from pyrendering.vectors import Matrix, Point, Point3D, Vec2, Vec3


@dataclass
class Camera:
    """Camera class for 3D rendering"""

    position: Vec3 = field(default_factory=lambda: Vec3(0, 0, 0))
    look_at: Vec3 = field(
        default_factory=lambda: Vec3(0, 0, -1)
    )  # Looking down -Z by default
    up: Vec3 = field(default_factory=lambda: Vec3(0, 1, 0))  # Y is up by default
    fov: float = 90


@dataclass
class Mesh(Shape):
    """3D Mesh class"""

    vertices: List[Point3D]
    faces: List[Vec3]

    def __post_init__(self):
        if not self.vertices or not self.faces:
            raise ValueError("Mesh must have vertices and faces defined.")


class Renderer3D:
    """3D Renderer class"""

    def __init__(self, graphics: Graphics, cam: Camera):
        self.gfx = graphics
        self.camera = cam
        self.meshes = []

    def add_mesh(self, mesh: Mesh):
        self.meshes.append(mesh)

    def project_vertices(self, vertices: List[Point3D]) -> List[Point]:
        projected = []
        aspect_ratio = (
            self.gfx.graphics_context.width / self.gfx.graphics_context.height
        )
        fov_rad = 1 / np.tan(np.radians(self.camera.fov) / 2)

        for vertex in vertices:
            # Simple perspective projection
            x = vertex.position.x - self.camera.position.x
            y = vertex.position.y - self.camera.position.y
            z = vertex.position.z - self.camera.position.z

            if z == 0:
                z = 0.0001

            px = (x * fov_rad * aspect_ratio) / z
            py = (y * fov_rad) / z

            # Convert to screen space
            screen_x = (px + 1) * 0.5 * self.gfx.graphics_context.width
            screen_y = (1 - (py + 1) * 0.5) * self.gfx.graphics_context.height
            projected.append(Point(Vec2(screen_x, screen_y), vertex.color))

        return projected

    def render(self):
        for mesh in self.meshes:
            # Render each mesh to the screen with the current camera settings
            vertices_2d = self.project_vertices(mesh.vertices)

            for face in mesh.faces:
                # Use original 3D vertices for normal calculation
                p1_3d = mesh.vertices[int(face.x)].position
                p2_3d = mesh.vertices[int(face.y)].position
                p3_3d = mesh.vertices[int(face.z)].position

                # Backface culling: calculate the normal and check if it's facing the camera
                v1 = p2_3d - p1_3d
                v2 = p3_3d - p1_3d
                normal = v1.cross(v2)

                # Adjust backface culling logic based on coordinate system
                if normal.dot(self.camera.look_at - self.camera.position) >= 0:
                    continue

                # Use 2D projected vertices for rendering
                p1 = vertices_2d[int(face.x)]
                p2 = vertices_2d[int(face.y)]
                p3 = vertices_2d[int(face.z)]

                triangle = Triangle(p1, p2, p3)
                self.gfx.draw(triangle, draw_mode="fill")

    def clear_meshes(self):
        self.meshes = []

    def set_camera(self, cam: Camera):
        self.camera = cam

    def move_camera(self, new_position: Vec3):
        self.camera.position = new_position

    def look_at(self, target: Vec3):
        self.camera.look_at = target

    def set_fov(self, fov: float):
        self.camera.fov = fov


def rotate_vertices(vertices, angle_x_local, angle_y_local, angle_z_local):
    rotation_x = Matrix.rotation_x(angle_x_local)
    rotation_y = Matrix.rotation_y(angle_y_local)
    rotation_z = Matrix.rotation_z(angle_z_local)

    rotated_vertices = []
    for vertex in deepcopy(vertices):
        rotated_position_array = (
            rotation_x @ rotation_y @ rotation_z @ vertex.position
        ).data
        rotated_position = Vec3(
            rotated_position_array[0],
            rotated_position_array[1],
            rotated_position_array[2],
        )
        rotated_vertices.append(Point3D(rotated_position, vertex.color))

    return rotated_vertices


if __name__ == "__main__":
    # Example usage
    gfx = Graphics(800, 600, "3D Renderer Example")
    camera = Camera(position=Vec3(0, 0, 5))
    renderer = Renderer3D(gfx, camera)

    # Define a simple cube mesh
    cube_vertices = [
        Point3D(Vec3(-1, -1, -1), Color.from_rgb(255, 0, 0)),
        Point3D(Vec3(1, -1, -1), Color.from_rgb(0, 255, 0)),
        Point3D(Vec3(1, 1, -1), Color.from_rgb(0, 0, 255)),
        Point3D(Vec3(-1, 1, -1), Color.from_rgb(255, 255, 0)),
        Point3D(Vec3(-1, -1, 1), Color.from_rgb(0, 255, 255)),
        Point3D(Vec3(1, -1, 1), Color.from_rgb(255, 0, 255)),
        Point3D(Vec3(1, 1, 1), Color.from_rgb(192, 192, 192)),
        Point3D(Vec3(-1, 1, 1), Color.from_rgb(128, 0, 128)),
    ]

    cube_faces = [
        Vec3(0, 1, 2),
        Vec3(0, 2, 3),
        Vec3(4, 5, 6),
        Vec3(4, 6, 7),
        Vec3(0, 1, 5),
        Vec3(0, 5, 4),
        Vec3(2, 3, 7),
        Vec3(2, 7, 6),
        Vec3(0, 3, 7),
        Vec3(0, 7, 4),
        Vec3(1, 2, 6),
        Vec3(1, 6, 5),
    ]

    cube_mesh = Mesh(vertices=cube_vertices, faces=cube_faces)
    renderer.add_mesh(cube_mesh)

    angle_x_main, angle_y_main, angle_z_main = 0, 0, 0

    while not gfx.should_close():
        gfx.poll_events()
        gfx.begin_frame()
        gfx.clear(Color.from_hex("#000000"))  # Clear to black

        # Rotate the cube
        angle_x_main += 0.01
        angle_y_main += 0.01
        angle_z_main += 0.01
        cube_mesh.vertices = rotate_vertices(
            cube_vertices, angle_x_main, angle_y_main, angle_z_main
        )

        renderer.render()

        gfx.display()
        time.sleep(0.016)  # Limit frame rate to ~60 FPS

    gfx.cleanup()
