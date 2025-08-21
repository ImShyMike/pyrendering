# pylint: disable=missing-function-docstring,missing-module-docstring

import math
import traceback
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import time

import numpy as np
import wgpu
from PIL import Image
from wgpu.gui.auto import select_backend

HexLike = Union[str, int]
RGB = Tuple[int, int, int]
RGBA = Tuple[int, int, int, int]
NormalRGB = Tuple[float, float, float]
NormalRGBA = Tuple[float, float, float, float]


@dataclass
class Color:
    """Color class"""

    r: int
    g: int
    b: int
    a: int = 255

    def as_tuple(self) -> RGBA:
        return (self.r, self.g, self.b, self.a)

    def as_rgb_tuple(self) -> RGB:
        return (self.r, self.g, self.b)

    def as_normalized(self) -> NormalRGBA:
        return (self.r / 255, self.g / 255, self.b / 255, self.a / 255)

    def as_rgb_normalized(self) -> NormalRGB:
        return (self.r / 255, self.g / 255, self.b / 255)

    def as_hex(self) -> HexLike:
        r, g, b = (
            self.r & 0xFF,
            self.g & 0xFF,
            self.b & 0xFF,
        )
        return f"#{r:02x}{g:02x}{b:02x}"

    def as_hex8(self) -> HexLike:
        r, g, b, a = self.r & 0xFF, self.g & 0xFF, self.b & 0xFF, self.a & 0xFF
        return f"#{r:02x}{g:02x}{b:02x}{a:02x}"

    @staticmethod
    def from_rgb(r: int, g: int, b: int) -> "Color":
        return Color(r & 0xFF, g & 0xFF, b & 0xFF, 255)

    @staticmethod
    def from_rgba(r: int, g: int, b: int, a: int) -> "Color":
        return Color(r & 0xFF, g & 0xFF, b & 0xFF, a & 0xFF)

    @staticmethod
    def from_hex(value: HexLike) -> "Color":
        if not isinstance(value, int):
            value = int(value.replace("#", ""), 16)
        return Color((value >> 16) & 0xFF, (value >> 8) & 0xFF, value & 0xFF, 255)

    @staticmethod
    def from_hex8(value: HexLike) -> "Color":
        if not isinstance(value, int):
            value = int(value.replace("#", ""), 16)
        return Color(
            (value >> 24), (value >> 16) & 0xFF, (value >> 8) & 0xFF, value & 0xFF
        )


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


class GraphicsContext:
    """Graphics context handler"""

    def __init__(self, width: int, height: int, title: str = "pyrendering"):
        # Create the WGPU canvas
        self.canvas = select_backend().WgpuCanvas(size=(width, height), title=title)

        # Init WGPU
        self.adapter = wgpu.gpu.request_adapter_sync()
        self.device = self.adapter.request_device_sync()

        # Create surface and configure
        self.context = self.canvas.get_context("wgpu")
        self.surface_format = self.context.get_preferred_format(self.adapter)
        self.context.configure(
            device=self.device,
            format=self.surface_format,
        )

        self.width = width
        self.height = height

        # Create command encoder
        self.encoder = None
        self.render_pass = None

        # Batch system
        self.sprite_batch = SpriteBatch(self.device)
        self.shape_batch = ShapeBatch(
            self.device, self.surface_format, self.width, self.height
        )

    def begin_frame(self, clear_color: Color = Color(0, 0, 0, 255)):
        """Start a new frame"""
        self.encoder = self.device.create_command_encoder()

        # Get current surface texture
        self.current_texture = self.context.get_current_texture()  # pylint: disable=attribute-defined-outside-init

        # Create render pass
        self.render_pass = self.encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": self.current_texture.create_view(),
                    "resolve_target": None,
                    "clear_value": clear_color.as_normalized(),
                    "load_op": wgpu.LoadOp.clear,
                    "store_op": wgpu.StoreOp.store,
                }
            ]
        )

    def end_frame(self):
        """Finish and display the frame"""
        if self.render_pass:
            # Flush all batches
            self.sprite_batch.flush(self.render_pass)
            self.shape_batch.flush(self.render_pass)

            self.render_pass.end()
            self.render_pass = None

        if self.encoder:
            command_buffer = self.encoder.finish()
            self.device.queue.submit([command_buffer])
            self.encoder = None

        try:
            self.context.present()
        except Exception:
            pass

    def is_closed(self) -> bool:
        return self.canvas.is_closed()

    def cleanup(self):
        """Cleanup resources"""
        if self.render_pass:
            self.render_pass.end()
            self.render_pass = None

        if self.encoder:
            self.encoder = None

        if hasattr(self, "canvas") and self.canvas:
            self.canvas.close()


class Texture:
    """Texture wrapper"""

    def __init__(self, device: wgpu.GPUDevice, image_path: str):
        # Load the image
        img = Image.open(image_path).convert("RGBA")
        self.width, self.height = img.size

        # Create GPU texture
        self.gpu_texture = device.create_texture(
            size=(self.width, self.height, 1),
            format=wgpu.TextureFormat.rgba8unorm,  # type: ignore
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,  # type: ignore
        )

        # Convert image to bytes
        img_data = np.array(img, dtype=np.uint8)

        # Upload image data
        device.queue.write_texture(
            {"texture": self.gpu_texture, "mip_level": 0, "origin": (0, 0, 0)},
            memoryview(img_data.tobytes()),
            {
                "offset": 0,
                "bytes_per_row": self.width * 4,
                "rows_per_image": self.height,
            },
            (self.width, self.height, 1),
        )

        # Create texture view for sampling
        self.view = self.gpu_texture.create_view()


class SpriteBatch:
    """Sprite batching system"""

    def __init__(self, device: wgpu.GPUDevice, max_sprites: int = 1000):
        self.device = device
        self.sprites = []
        self.max_sprites = max_sprites

        # Each sprite = position + size + texture coords + color
        self.vertex_buffer = None
        self.setup_pipeline()

    def setup_pipeline(self):
        """Create rendering pipeline for sprites"""
        # TODO: actually render stuff :sob:

    def add_sprite(
        self,
        texture: Texture,
        position: Vec2,
        size: Optional[Vec2] = None,
        color: Color = Color(1, 1, 1, 1),
        rotation: float = 0,
    ):
        """Add a sprite to the batch"""

        if size is None:
            size = Vec2(texture.width, texture.height)

        self.sprites.append(
            {
                "texture": texture,
                "position": position,
                "size": size,
                "color": color,
                "rotation": rotation,
            }
        )

        # Auto flush if batch is full
        if len(self.sprites) >= self.max_sprites:
            self.flush()

    def flush(self, render_pass=None):
        """Render all batched sprites"""
        if not self.sprites or not render_pass:
            return

        # TODO: update vertex buffer with sprite data
        # TODO: make draw calls
        # TODO: clear sprite list

        self.sprites.clear()


class ShapeBatch:
    """Shape drawing system"""

    def __init__(self, device: wgpu.GPUDevice, surface_format, width: int, height: int):
        self.device = device
        self.vertices = []
        self.indices = []
        self.line_vertices = []
        self.line_indices = []

    def add_rectangle(self, rect: Rect, color: Color, filled: bool = True):
        """Add a rectangle to the batch"""

        if filled:
            # Add 4 vertices for filled quad
            base_idx = len(self.vertices)

            self.vertices.extend(
                [
                    [rect.x, rect.y, *color.as_normalized()],
                    [rect.x + rect.width, rect.y, *color.as_normalized()],
                    [rect.x + rect.width, rect.y + rect.height, *color.as_normalized()],
                    [rect.x, rect.y + rect.height, *color.as_normalized()],
                ]
            )

            # Add indices for 2 triangles
            self.indices.extend(
                [
                    base_idx,
                    base_idx + 1,
                    base_idx + 2,
                    base_idx,
                    base_idx + 2,
                    base_idx + 3,
                ]
            )
        else:
            # Add vertices for rectangle outline
            base_idx = len(self.line_vertices)
            self.line_vertices.extend(
                [
                    [rect.x, rect.y, *color.as_normalized()],
                    [rect.x + rect.width, rect.y, *color.as_normalized()],
                    [rect.x + rect.width, rect.y + rect.height, *color.as_normalized()],
                    [rect.x, rect.y + rect.height, *color.as_normalized()],
                ]
            )

            # Add indices for line loop
            self.line_indices.extend(
                [
                    base_idx, base_idx + 1,
                    base_idx + 1, base_idx + 2,
                    base_idx + 2, base_idx + 3,
                    base_idx + 3, base_idx
                ]
            )

    def add_circle(self, circle: Circle, color: Color, segments: int = 32):
        base_idx = len(self.vertices)

        center = circle.center
        radius = circle.radius

        # Add center vertex
        self.vertices.append([center.x, center.y, *color.as_normalized()])

        # Add perimeter vertices
        for i in range(segments):
            angle = 2 * math.pi * i / segments
            x = center.x + radius * math.cos(angle)
            y = center.y + radius * math.sin(angle)
            self.vertices.append([x, y, *color.as_normalized()])

        # Add triangles from center to perimeter
        for i in range(segments):
            next_i = (i + 1) % segments
            self.indices.extend([base_idx, base_idx + 1 + i, base_idx + 1 + next_i])

    def flush(self, render_pass=None):
        """Render all batched shapes"""
        if not render_pass:
            return

        # TODO: upload vertices/indices to GPU and draw

        self.vertices.clear()
        self.indices.clear()
        self.line_vertices.clear()
        self.line_indices.clear()


class Graphics:
    """Graphics handler"""

    def __init__(self, width: int = 800, height: int = 600, title: str = "Graphics"):
        self.context = GraphicsContext(width, height, title)
        self.textures = {}  # Texture cache
        self.last_frame = time.perf_counter()

    def tick(self, fps: int):
        if fps > 0:
            now = time.perf_counter()
            elapsed = now - self.last_frame
            delay = max(0, (1.0 / fps) - elapsed)
            if delay > 0:
                time.sleep(delay)
            self.last_frame = time.perf_counter()

    def load_texture(self, path: str, name: Optional[str] = None) -> Texture:
        if name is None:
            name = path

        if name not in self.textures:
            self.textures[name] = Texture(self.context.device, path)

        return self.textures[name]

    def clear(self, color: Color = Color(0, 0, 0, 255)):
        """Clear the screen with a color"""
        self.context.begin_frame(color)
        self.context.end_frame()

    def draw_sprite(
        self,
        texture_name: str,
        position: Vec2,
        size: Optional[Vec2] = None,
        color: Color = Color(255, 255, 255, 255),
        rotation: float = 0,
    ):
        """Draw a sprite"""
        texture = self.textures[texture_name]
        self.context.sprite_batch.add_sprite(texture, position, size, color, rotation)

    def draw_rect(self, rect: Rect, color: Color, filled: bool = True):
        """Draw a rectangle"""
        self.context.shape_batch.add_rectangle(rect, color, filled)

    def draw_circle(self, circle: Circle, color: Color):
        """Draw a circle"""
        self.context.shape_batch.add_circle(circle, color)

    def display(self):
        """Display the frame (call once per frame)"""
        self.last_frame = time.perf_counter()
        self.context.end_frame()

    def begin_frame(self):
        """Start a new frame"""
        self.context.begin_frame(getattr(self, "clear_color", Color(0, 0, 0, 255)))

    def is_closed(self) -> bool:
        """Check if the graphics context is closed"""
        return self.context.is_closed()

    def cleanup(self):
        """Cleanup resources"""
        self.context.cleanup()
        self.textures.clear()


def main():
    """Testest"""

    # Initialize graphics
    gfx = Graphics(800, 600, "My Game")
    frame_count = 0

    try:
        while not gfx.is_closed():
            frame_count += 1

            gfx.begin_frame()
            gfx.clear(Color.from_hex("#1a1a2e"))  # Dark blue background

            # Draw some shapes
            gfx.draw_rect(
                Rect(100, 100, 200, 150), Color.from_hex("#e94560")
            )  # Red rectangle
            gfx.draw_circle(
                Circle(Vec2(400, 300), 50), Color.from_hex("#0f3460")
            )  # Blue circle

            # Draw outline rectangle
            gfx.draw_rect(
                Rect(350, 100, 150, 100), Color.from_hex("#ffffff"), filled=False
            )

            gfx.tick(1)
            gfx.display()
            print(f"Frame {frame_count} rendered")

    except Exception as e: # pylint: disable=broad-exception-caught
        print(f"Error during rendering: {e}")
        traceback.print_exc()
    finally:
        print("Cleaning up...")
        gfx.cleanup()
        print("Cleanup completed")


if __name__ == "__main__":
    main()
