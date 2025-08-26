# pylint: disable=missing-function-docstring,missing-module-docstring

import math
import time
import traceback
from dataclasses import dataclass
from typing import Tuple, Union, cast

import glfw
import moderngl
import numpy as np

HexLike = Union[str, int]
RGB = Tuple[int, int, int]
RGBA = Tuple[int, int, int, int]
NormalRGB = Tuple[float, float, float]
NormalRGBA = Tuple[float, float, float, float]

NUM_VERTICES = 10000


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

    def __init__(
        self,
        width: int,
        height: int,
        title: str = "pyrendering",
        standalone: bool = False,
        vsync: bool = True,
    ):
        self.width = width
        self.height = height
        self.title = title
        self.window = None
        self.last_time = time.time()

        if standalone:
            # Create headless context
            self.ctx = moderngl.create_context(standalone=True)
        else:
            # Create windowed context with GLFW
            if not glfw.init():
                raise RuntimeError("Failed to initialize GLFW")

            # Create window
            self.window = glfw.create_window(width, height, title, None, None)
            if not self.window:
                glfw.terminate()
                raise RuntimeError("Failed to create GLFW window")

            # Make context current
            glfw.make_context_current(self.window)

            # Set the Vsync mode
            glfw.swap_interval(1 if vsync else 0)

            # Create ModernGL context from current OpenGL context
            self.ctx = moderngl.create_context()

        # Enable blending for transparency
        self.ctx.enable(moderngl.BLEND)  # pylint: disable=no-member
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA  # pylint: disable=no-member

        # Create shader program
        self.program = self.ctx.program(
            vertex_shader="""
#version 330

in vec2 in_vert;
in vec3 in_color;

out vec3 v_color;

uniform vec2 u_resolution;

void main() {
    // Convert from pixel coordinates to NDC (-1 to 1)
    vec2 position = ((in_vert / u_resolution) * 2.0) - 1.0;
    position.y = -position.y; // Flip Y axis
    
    v_color = in_color;
    gl_Position = vec4(position, 0.0, 1.0);
}
""",
            fragment_shader="""
#version 330

in vec3 v_color;
out vec4 fragColor;

void main() {
    fragColor = vec4(v_color, 1.0);
}
""",
        )

        # Set resolution uniform
        u_resolution = cast(moderngl.Uniform, self.program["u_resolution"])
        u_resolution.value = (float(width), float(height))

        # Create vertex buffer for batched rendering
        self.vertex_buffer = self.ctx.buffer(
            reserve=NUM_VERTICES * 5 * 4
        )  # 5 floats per vertex, 4 bytes each

        # Create vertex array object
        self.vao = self.ctx.vertex_array(
            self.program, [(self.vertex_buffer, "2f 3f", "in_vert", "in_color")]
        )

        # Vertex data for current frame
        self.vertices = []

        # Create framebuffer for offscreen rendering if needed
        if standalone:
            self.fbo = self.ctx.framebuffer(
                color_attachments=[self.ctx.texture((width, height), 4)]
            )
        else:
            self.fbo = None

    def clear(self, color: Color):
        """Clear the screen with a color"""
        normalized = color.as_normalized()
        self.ctx.clear(normalized[0], normalized[1], normalized[2], normalized[3])

    def screen_to_ndc(self, x: float, y: float) -> Tuple[float, float]:
        """Convert screen coordinates to normalized device coordinates"""
        ndc_x = (x / self.width) * 2.0 - 1.0
        ndc_y = -((y / self.height) * 2.0 - 1.0)  # flip Y
        return ndc_x, ndc_y

    def add_vertex(self, x: float, y: float, color: Color):
        """Add a vertex to the batch"""
        r, g, b = color.as_rgb_normalized()
        self.vertices.extend([x, y, r, g, b])

    def draw_rect(self, rect: Rect, color: Color, filled: bool = True):
        """Draw a rectangle"""
        if filled:
            # Add vertices for filled rectangle (2 triangles)
            # Triangle 1
            self.add_vertex(rect.x, rect.y, color)
            self.add_vertex(rect.x + rect.width, rect.y, color)
            self.add_vertex(rect.x, rect.y + rect.height, color)

            # Triangle 2
            self.add_vertex(rect.x + rect.width, rect.y, color)
            self.add_vertex(rect.x + rect.width, rect.y + rect.height, color)
            self.add_vertex(rect.x, rect.y + rect.height, color)
        else:
            # Add vertices for rectangle outline (lines)
            # Top line
            self.add_vertex(rect.x, rect.y, color)
            self.add_vertex(rect.x + rect.width, rect.y, color)

            # Right line
            self.add_vertex(rect.x + rect.width, rect.y, color)
            self.add_vertex(rect.x + rect.width, rect.y + rect.height, color)

            # Bottom line
            self.add_vertex(rect.x + rect.width, rect.y + rect.height, color)
            self.add_vertex(rect.x, rect.y + rect.height, color)

            # Left line
            self.add_vertex(rect.x, rect.y + rect.height, color)
            self.add_vertex(rect.x, rect.y, color)

    def draw_circle(self, circle: Circle, color: Color, segments: int = 32):
        """Draw a circle"""
        center_x, center_y = circle.center.x, circle.center.y
        radius = circle.radius

        # Generate triangles from center to perimeter
        for i in range(segments):
            angle1 = 2 * math.pi * i / segments
            angle2 = 2 * math.pi * (i + 1) / segments

            # Center vertex
            self.add_vertex(center_x, center_y, color)

            # First perimeter vertex
            x1 = center_x + radius * math.cos(angle1)
            y1 = center_y + radius * math.sin(angle1)
            self.add_vertex(x1, y1, color)

            # Second perimeter vertex
            x2 = center_x + radius * math.cos(angle2)
            y2 = center_y + radius * math.sin(angle2)
            self.add_vertex(x2, y2, color)

    def begin_frame(self):
        """Begin a new frame"""
        self.vertices.clear()
        if self.fbo:
            self.fbo.use()

    def flush(self):
        """Render all batched geometry"""
        if not self.vertices:
            return

        # Convert to numpy array and upload to GPU
        vertex_data = np.array(self.vertices, dtype=np.float32)
        self.vertex_buffer.write(vertex_data.tobytes())

        # Draw triangles for filled shapes and lines for wireframes
        num_vertices = len(self.vertices) // 5

        # Render the vertices
        if num_vertices >= 3:
            self.ctx.viewport = (0, 0, self.width, self.height)
            self.vao.render(vertices=num_vertices)

        # Clear vertex buffer for next frame
        self.vertices.clear()

    def display(self):
        """Present the rendered frame"""
        self.flush()
        # Swap buffers if using windowed mode
        if self.window:
            glfw.swap_buffers(self.window)

    def tick(self, target_fps: float) -> float:
        """Cap frame rate and return normalized delta time."""
        now = time.time()
        elapsed = now - self.last_time
        target_frame_time = 1.0 / target_fps if target_fps > 0 else 0.0

        if target_frame_time > 0.0 and elapsed < target_frame_time:
            time.sleep(target_frame_time - elapsed)
            now = time.time()
            elapsed = now - self.last_time

        self.last_time = now

        if target_fps > 0 and elapsed > 0:
            actual_fps = 1.0 / elapsed
            normalized = actual_fps / target_fps
            return max(0.0, min(normalized, 1.0))
        return 0.0

    def should_close(self) -> bool:
        """Check if the window should close"""
        if self.window:
            return glfw.window_should_close(self.window)
        return False

    def poll_events(self):
        """Poll for window events"""
        if self.window:
            glfw.poll_events()

    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, "vertex_buffer"):
            self.vertex_buffer.release()
        if hasattr(self, "vao"):
            self.vao.release()
        if hasattr(self, "program"):
            self.program.release()
        if hasattr(self, "fbo") and self.fbo:
            self.fbo.release()

        # Clean up GLFW
        if self.window:
            glfw.destroy_window(self.window)
            glfw.terminate()


class Graphics:
    """Graphics handler"""

    def __init__(
        self,
        width: int = 800,
        height: int = 600,
        title: str = "Graphics",
        standalone: bool = False,
        vsync: bool = True,
    ):
        self.graphics_context = GraphicsContext(width, height, title, standalone, vsync)

    def should_close(self) -> bool:
        """Check if the window should close"""
        return self.graphics_context.should_close()

    def poll_events(self):
        """Poll for window events"""
        self.graphics_context.poll_events()

    def begin_frame(self):
        """Begin a new frame"""
        self.graphics_context.begin_frame()

    def clear(self, color: Color):
        """Clear the screen"""
        self.graphics_context.clear(color)

    def draw_rect(self, rect: Rect, color: Color, filled: bool = True):
        """Draw a rectangle"""
        self.graphics_context.draw_rect(rect, color, filled)

    def draw_circle(self, circle: Circle, color: Color, segments: int = 32):
        """Draw a circle"""
        self.graphics_context.draw_circle(circle, color, segments)

    def display(self):
        """Display the frame"""
        self.graphics_context.display()

    def tick(self, target_fps: float) -> float:
        """Update timing"""
        return self.graphics_context.tick(target_fps)

    def cleanup(self):
        """Clean up resources"""
        self.graphics_context.cleanup()


def main():
    """Testest"""

    # Initialize graphics
    gfx = Graphics(800, 600, "My Game", standalone=False, vsync=True)
    frame_count = 0
    target_fps = 60

    try:
        # Run until window closes
        while not gfx.should_close():
            delta_time = gfx.tick(target_fps)

            frame_count += 1

            gfx.poll_events()
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

            gfx.display()

            if frame_count % target_fps == 0:  # Print data every second
                print(
                    f"Frame {frame_count} rendered. FPS: {(target_fps * delta_time):.2f}"
                )

    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"Error during rendering: {e}")
        traceback.print_exc()
    finally:
        print("Cleaning up...")
        gfx.cleanup()
        print("Cleanup completed")


if __name__ == "__main__":
    main()
