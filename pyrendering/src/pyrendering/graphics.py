# pylint: disable=missing-function-docstring,missing-module-docstring

import time
from typing import Tuple, cast

import glfw
import moderngl
import numpy as np

from pyrendering.color import Color
from pyrendering.shapes import Shape, Circle, Rect

NUM_VERTICES = 10000


class DrawMode:
    """Vertex types"""

    TRIANGLE = 0
    LINE = 1


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

            self.monitor = glfw.get_primary_monitor()
            self.mode = glfw.get_video_mode(self.monitor)

            # Create ModernGL context from current OpenGL context
            self.ctx = moderngl.create_context()

        # Enable blending for transparency
        self.ctx.enable(moderngl.BLEND)  # pylint: disable=no-member
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA  # pylint: disable=no-member

        # Set the viewport
        self.ctx.viewport = (0, 0, width, height)

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
        self.draw_mode = DrawMode.TRIANGLE

        self.triangle_vertices = np.empty(
            (0, 5), dtype=np.float32
        )  # 5 floats per vertex
        self.line_vertices = np.empty((0, 5), dtype=np.float32)  # 5 floats per vertex

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

    def get_monitor_mode(self) -> Tuple[int, int, int]:
        """Get the current monitor mode (width, height, refresh rate)"""
        if self.monitor and self.mode:
            return self.mode.size.width, self.mode.size.height, self.mode.refresh_rate
        return 0, 0, 0

    def add_vertex(self, x: float, y: float, color: Color):
        """Add a vertex to the batch"""
        r, g, b = color.as_rgb_normalized()
        vertex = np.array([x, y, r, g, b], dtype=np.float32)
        if self.draw_mode == DrawMode.TRIANGLE:
            self.triangle_vertices = np.append(self.triangle_vertices, [vertex], axis=0)
        elif self.draw_mode == DrawMode.LINE:
            self.line_vertices = np.append(self.line_vertices, [vertex], axis=0)

    def draw_rect(self, rect: Rect):
        """Draw a rectangle"""
        x, y = rect.x, rect.y
        width, height = rect.width, rect.height
        color = rect.color

        if rect.filled:
            # Add vertices for filled rectangle (2 triangles)
            self.draw_mode = DrawMode.TRIANGLE

            # Triangle 1
            self.add_vertex(x, y, color)
            self.add_vertex(x + width, y, color)
            self.add_vertex(x, y + height, color)

            # Triangle 2
            self.add_vertex(x + width, y, color)
            self.add_vertex(x + width, y + height, color)
            self.add_vertex(x, y + height, color)
        else:
            # Add vertices for rectangle outline (lines)
            self.draw_mode = DrawMode.LINE

            # Top line
            self.add_vertex(x, y, color)
            self.add_vertex(x + width, y, color)

            # Right line
            self.add_vertex(x + width, y, color)
            self.add_vertex(x + width, y + height, color)

            # Bottom line
            self.add_vertex(x + width, y + height, color)
            self.add_vertex(x, y + height, color)

            # Left line
            self.add_vertex(x, y + height, color)
            self.add_vertex(x, y, color)

    def draw_circle(self, circle: Circle):
        """Draw a circle"""
        center = circle.center.data
        radius = circle.radius
        color = circle.color
        segments = circle.segments

        self.draw_mode = DrawMode.TRIANGLE

        # Generate angles for the circle
        angles = np.linspace(0, 2 * np.pi, segments, endpoint=False)
        offsets = np.stack((np.cos(angles), np.sin(angles)), axis=1) * radius

        # Create vertices for the circle
        center_vertex = np.array(
            [*center, *color.as_rgb_normalized()], dtype=np.float32
        )
        vertices = np.empty((0, 5), dtype=np.float32)

        for i in range(segments):
            v1 = center + offsets[i]
            v2 = center + offsets[(i + 1) % segments]
            triangle = np.array(
                [
                    center_vertex,
                    [*v1, *color.as_rgb_normalized()],
                    [*v2, *color.as_rgb_normalized()],
                ],
                dtype=np.float32,
            )
            vertices = np.append(vertices, triangle, axis=0)

        self.triangle_vertices = np.append(self.triangle_vertices, vertices, axis=0)

    def begin_frame(self):
        """Begin a new frame"""
        self.triangle_vertices = self.triangle_vertices[:0]
        self.line_vertices = self.line_vertices[:0]
        if self.fbo:
            self.fbo.use()

    def flush(self):
        """Render all batched geometry"""
        if self.triangle_vertices.size == 0 and self.line_vertices.size == 0:
            return

        # Upload triangle vertices to GPU
        if self.triangle_vertices.size > 0:
            self.vertex_buffer.write(self.triangle_vertices.tobytes())

            # Calculate the number of triangle vertices
            triangle_num_vertices = len(self.triangle_vertices)

            # Render triangles
            if triangle_num_vertices >= 3:  # At least 3 vertices, render as triangles
                self.vao.render(moderngl.TRIANGLES, vertices=triangle_num_vertices)  # pylint: disable=no-member

        # Upload line vertices to GPU
        if self.line_vertices.size > 0:
            self.vertex_buffer.write(self.line_vertices.tobytes())

            # Calculate the number of line vertices
            line_num_vertices = len(self.line_vertices)

            # Render lines
            if line_num_vertices >= 2:
                self.vao.render(moderngl.LINES, vertices=line_num_vertices)  # pylint: disable=no-member

        # Clear vertex buffers for next frame
        self.triangle_vertices = self.triangle_vertices[:0]
        self.line_vertices = self.line_vertices[:0]

    def display(self):
        """Present the rendered frame"""
        self.flush()
        # Swap buffers if using windowed mode
        if self.window:
            glfw.swap_buffers(self.window)

    def tick(self, target_fps: float = 0) -> float:
        """Cap frame rate and return delta time in seconds."""
        now = time.time()
        elapsed = now - self.last_time
        target_frame_time = 1.0 / target_fps if target_fps > 0 else 0.0

        if target_frame_time > 0.0 and elapsed < target_frame_time:
            time.sleep(target_frame_time - elapsed)
            now = time.time()
            elapsed = now - self.last_time

        self.last_time = now
        return elapsed

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

    def get_monitor_mode(self) -> Tuple[int, int, int]:
        """Get the current monitor mode (width, height, refresh rate)"""
        return self.graphics_context.get_monitor_mode()

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

    def draw(self, shape: Shape):
        """Draw a shape"""
        if isinstance(shape, Rect):
            self.graphics_context.draw_rect(shape)
        elif isinstance(shape, Circle):
            self.graphics_context.draw_circle(shape)
        else:
            raise ValueError("Unsupported shape type")

    def display(self):
        """Display the frame"""
        self.graphics_context.display()

    def tick(self, target_fps: float) -> float:
        """Update timing"""
        return self.graphics_context.tick(target_fps)

    def cleanup(self):
        """Clean up resources"""
        self.graphics_context.cleanup()
