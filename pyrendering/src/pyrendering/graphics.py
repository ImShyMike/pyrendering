# pylint: disable=missing-function-docstring,missing-module-docstring

import math
import time
from typing import Tuple, cast

import glfw
import moderngl
import numpy as np

from pyrendering.color import Color
from pyrendering.shapes import Circle, Rect

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
        self.triangle_vertices = []
        self.line_vertices = []

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

    def add_vertex(
        self, x: float, y: float, color: Color
    ):
        """Add a vertex to the batch"""
        r, g, b = color.as_rgb_normalized()
        if self.draw_mode == DrawMode.TRIANGLE:
            self.triangle_vertices.extend([x, y, r, g, b])
        elif self.draw_mode == DrawMode.LINE:
            self.line_vertices.extend([x, y, r, g, b])

    def draw_rect(self, rect: Rect, color: Color, filled: bool = True):
        """Draw a rectangle"""
        if filled:
            # Add vertices for filled rectangle (2 triangles)
            self.draw_mode = DrawMode.TRIANGLE

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
            self.draw_mode = DrawMode.LINE

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

        self.draw_mode = DrawMode.TRIANGLE

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
        self.triangle_vertices.clear()
        self.line_vertices.clear()
        if self.fbo:
            self.fbo.use()

    def flush(self):
        """Render all batched geometry"""
        if not self.triangle_vertices and not self.line_vertices:
            return

        # Convert triangle vertices to numpy array and upload to GPU
        if self.triangle_vertices:
            triangle_vertex_data = np.array(self.triangle_vertices, dtype=np.float32)
            self.vertex_buffer.write(triangle_vertex_data.tobytes())

            # Calculate the number of triangle vertices
            triangle_num_vertices = len(self.triangle_vertices) // 5

            # Render triangles
            if triangle_num_vertices >= 3:  # At least 3 vertices, render as triangles
                self.vao.render(moderngl.TRIANGLES, vertices=triangle_num_vertices)

        # Convert line vertices to numpy array and upload to GPU
        if self.line_vertices:
            line_vertex_data = np.array(self.line_vertices, dtype=np.float32)
            self.vertex_buffer.write(line_vertex_data.tobytes())

            # Calculate the number of line vertices
            line_num_vertices = len(self.line_vertices) // 5

            # Render lines
            if line_num_vertices >= 2:
                self.vao.render(moderngl.LINES, vertices=line_num_vertices)  # pylint: disable=no-member

        # Clear vertex buffers for next frame
        self.triangle_vertices.clear()
        self.line_vertices.clear()

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
