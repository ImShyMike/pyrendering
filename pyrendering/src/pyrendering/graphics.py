# pylint: disable=missing-function-docstring,missing-module-docstring

import time
from typing import Optional, Tuple, cast

import glfw
import moderngl
import numpy as np

from pyrendering.color import Color
from pyrendering.font import FontManager, FontRenderer
from pyrendering.shapes import Circle, Rect, Shape, Triangle, Vec2

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

            # Set the version of OpenGL
            glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
            glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
            glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

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

            # Enable multisampling
            glfw.window_hint(glfw.SAMPLES, 4)  # Request 4x multisampling

            # Create ModernGL context from current OpenGL context
            self.ctx = moderngl.create_context()

        # Enable blending for transparency
        self.ctx.enable(moderngl.BLEND)  # pylint: disable=no-member
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA  # pylint: disable=no-member

        # Enable multisampling in the OpenGL context
        self.ctx.enable_direct(0x809D)  # GL_MULTISAMPLE

        # Set the viewport
        self.ctx.viewport = (0, 0, width, height)

        # Create the font renderer
        self.font_renderer = FontRenderer(self.ctx)

        # Create shader program
        self.program = self.ctx.program(
            vertex_shader="""
#version 330

in vec2 in_vert;
in vec4 in_color;

out vec4 v_color;

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

in vec4 v_color;
out vec4 fragColor;

void main() {
    fragColor = v_color;
}
""",
        )

        # Create text shader program
        self.text_program = self.ctx.program(
            vertex_shader="""
#version 330

in vec2 in_vert;
in vec2 in_texcoord;
in vec4 in_color;

out vec2 v_texcoord;
out vec4 v_color;

uniform vec2 u_resolution;

void main() {
    vec2 position = ((in_vert / u_resolution) * 2.0) - 1.0;
    position.y = -position.y;
    
    v_texcoord = in_texcoord;
    v_color = in_color;
    gl_Position = vec4(position, 0.0, 1.0);
}
""",
            fragment_shader="""
#version 330

in vec2 v_texcoord;
in vec4 v_color;
out vec4 fragColor;

uniform sampler2D u_texture;

void main() {
    vec4 texColor = texture(u_texture, v_texcoord);
    fragColor = vec4(v_color.rgb, v_color.a * texColor.a);
}
""",
        )

        # Set resolution uniform
        u_resolution = cast(moderngl.Uniform, self.program["u_resolution"])
        u_resolution.value = (float(width), float(height))

        # Set resolution uniform for text shader
        text_u_resolution = cast(moderngl.Uniform, self.text_program["u_resolution"])
        text_u_resolution.value = (float(width), float(height))

        # Create vertex buffer for batched rendering
        self.vertex_buffer = self.ctx.buffer(
            reserve=NUM_VERTICES * 6 * 4
        )  # 6 floats per vertex, 4 bytes each

        # Create index buffer for indexed rendering
        self.index_buffer_gl = self.ctx.buffer(
            reserve=NUM_VERTICES * 6 * 4  # Reserve space for indices
        )

        # Create text vertex buffer
        self.text_vertex_buffer = self.ctx.buffer(
            reserve=NUM_VERTICES * 8 * 4
        )  # 8 floats per vertex

        # Create vertex array objects for both indexed and non-indexed rendering
        self.vao_indexed = self.ctx.vertex_array(
            self.program,
            [(self.vertex_buffer, "2f 4f", "in_vert", "in_color")],
            self.index_buffer_gl,
        )

        self.vao_simple = self.ctx.vertex_array(
            self.program, [(self.vertex_buffer, "2f 4f", "in_vert", "in_color")]
        )

        # Create text VAO
        self.text_vao = self.ctx.vertex_array(
            self.text_program,
            [
                (
                    self.text_vertex_buffer,
                    "2f 2f 4f",
                    "in_vert",
                    "in_texcoord",
                    "in_color",
                )
            ],
        )

        # Batching data structures
        self.triangle_vertices = np.empty(
            (0, 6), dtype=np.float32
        )  # Non-indexed triangles
        self.line_vertices = np.empty((0, 6), dtype=np.float32)  # Non-indexed lines

        # Indexed rendering data structures
        self.indexed_vertices = np.empty(
            (0, 6), dtype=np.float32
        )  # Vertices for indexed rendering
        self.triangle_indices = np.empty(0, dtype=np.uint32)  # Triangle indices
        self.line_indices = np.empty(0, dtype=np.uint32)  # Line indices

        # Text rendering data
        self.text_render_queue = []  # List of (vertices, texture) pairs

        self.font_manager = FontManager()

        self.vertex_count = 0  # Track current vertex count for indexing

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

    def add_vertex_simple(self, x: float, y: float, color: Color, draw_mode: int):
        """Add a vertex to simple (non-indexed) rendering"""
        r, g, b, a = color.as_normalized()
        vertex = np.array([x, y, r, g, b, a], dtype=np.float32)

        if draw_mode == DrawMode.TRIANGLE:
            self.triangle_vertices = np.append(self.triangle_vertices, [vertex], axis=0)
        elif draw_mode == DrawMode.LINE:
            self.line_vertices = np.append(self.line_vertices, [vertex], axis=0)

    def add_indexed_vertex(self, x: float, y: float, color: Color) -> int:
        """Add a vertex for indexed rendering and return its index"""
        r, g, b, a = color.as_normalized()
        vertex = np.array([x, y, r, g, b, a], dtype=np.float32)

        self.indexed_vertices = np.append(self.indexed_vertices, [vertex], axis=0)
        current_index = self.vertex_count
        self.vertex_count += 1
        return current_index

    def add_triangle_indices(self, indices: np.ndarray):
        """Add triangle indices for indexed rendering"""
        self.triangle_indices = np.append(
            self.triangle_indices, indices.astype(np.uint32)
        )

    def add_line_indices(self, indices: np.ndarray):
        """Add line indices for indexed rendering"""
        self.line_indices = np.append(self.line_indices, indices.astype(np.uint32))

    def draw_triangle(self, triangle: Triangle):
        """Draw a triangle using simple rendering"""
        p1, p2, p3 = triangle.p1, triangle.p2, triangle.p3

        self.add_vertex_simple(p1.x, p1.y, p1.color, DrawMode.TRIANGLE)
        self.add_vertex_simple(p2.x, p2.y, p2.color, DrawMode.TRIANGLE)
        self.add_vertex_simple(p3.x, p3.y, p3.color, DrawMode.TRIANGLE)

    def draw_rect(self, rect: Rect):
        """Draw a rectangle using efficient indexed rendering"""
        # Add the 4 vertices for the rectangle
        v0 = self.add_indexed_vertex(*rect.p1.unpack())
        v1 = self.add_indexed_vertex(*rect.p2.unpack())
        v2 = self.add_indexed_vertex(*rect.p3.unpack())
        v3 = self.add_indexed_vertex(*rect.p4.unpack())

        if rect.filled:
            # Two triangles for filled rectangle: (v0,v1,v2) and (v0,v2,v3)
            triangle_indices = np.array([v0, v1, v2, v0, v2, v3], dtype=np.uint32)
            self.add_triangle_indices(triangle_indices)
        else:
            # Four lines for rectangle outline: v0->v1, v1->v2, v2->v3, v3->v0
            line_indices = np.array([v0, v1, v1, v2, v2, v3, v3, v0], dtype=np.uint32)
            self.add_line_indices(line_indices)

    def draw_circle(self, circle: Circle):
        """Draw a circle using simple rendering"""
        center = circle.center.data
        radius = circle.radius
        color = circle.color
        segments = circle.segments

        # Generate angles for the circle
        angles = np.linspace(0, 2 * np.pi, segments, endpoint=False)
        offsets = np.stack((np.cos(angles), np.sin(angles)), axis=1) * radius

        # Create vertices for the circle using triangle fan approach
        for i in range(segments):
            v1 = center + offsets[i]
            v2 = center + offsets[(i + 1) % segments]

            # Add triangle: center -> v1 -> v2
            self.add_vertex_simple(center[0], center[1], color, DrawMode.TRIANGLE)
            self.add_vertex_simple(v1[0], v1[1], color, DrawMode.TRIANGLE)
            self.add_vertex_simple(v2[0], v2[1], color, DrawMode.TRIANGLE)

    def begin_frame(self):
        """Begin a new frame"""
        # Clear all vertex data
        self.triangle_vertices = self.triangle_vertices[:0]
        self.line_vertices = self.line_vertices[:0]
        self.indexed_vertices = self.indexed_vertices[:0]
        self.triangle_indices = self.triangle_indices[:0]
        self.line_indices = self.line_indices[:0]
        self.vertex_count = 0

        # Clear text render queue
        self.text_render_queue.clear()

        if self.fbo:
            self.fbo.use()

    def flush(self):
        """Render all batched geometry"""
        # Render indexed geometry first (rectangles)
        if self.indexed_vertices.size > 0:
            # Upload vertex data
            self.vertex_buffer.write(self.indexed_vertices.tobytes())

            # Combine triangle and line indices into a single buffer
            all_indices = (
                np.concatenate([self.triangle_indices, self.line_indices])
                if self.triangle_indices.size > 0 and self.line_indices.size > 0
                else (
                    self.triangle_indices
                    if self.triangle_indices.size > 0
                    else self.line_indices
                )
            )

            if all_indices.size > 0:
                self.index_buffer_gl.write(all_indices.tobytes())

                # Render triangles with indices
                if self.triangle_indices.size > 0:
                    triangle_count = len(self.triangle_indices)
                    self.vao_indexed.render(
                        moderngl.TRIANGLES,  # pylint: disable=no-member
                        vertices=triangle_count,
                        first=0,
                        instances=1,
                    )

                # Render lines with indices (offset by triangle count)
                if self.line_indices.size > 0:
                    line_count = len(self.line_indices)
                    line_offset = len(self.triangle_indices)
                    self.vao_indexed.render(
                        moderngl.LINES,  # pylint: disable=no-member
                        vertices=line_count,
                        first=line_offset,
                        instances=1,
                    )

        # Render simple geometry (triangles and lines without indices)
        total_simple_vertices = len(self.triangle_vertices) + len(self.line_vertices)
        if total_simple_vertices > 0:
            # Combine all simple vertices
            all_simple_vertices = np.concatenate(
                [self.triangle_vertices, self.line_vertices]
            )
            self.vertex_buffer.write(all_simple_vertices.tobytes())

            # Render triangles
            if len(self.triangle_vertices) > 0:
                triangle_count = len(self.triangle_vertices)
                self.vao_simple.render(
                    moderngl.TRIANGLES,  # pylint: disable=no-member
                    vertices=triangle_count,
                    first=0,
                )

            # Render lines
            if len(self.line_vertices) > 0:
                line_count = len(self.line_vertices)
                line_start = len(self.triangle_vertices)
                self.vao_simple.render(
                    moderngl.LINES,  # pylint: disable=no-member
                    vertices=line_count,
                    first=line_start,
                )

        # Render text
        self.flush_text()

    def flush_text(self):
        """Render all batched text"""
        if not self.text_render_queue:
            return

        # Render each character separately with its own texture
        for vertices, texture in self.text_render_queue:
            # Upload vertex data for this character
            self.text_vertex_buffer.write(vertices.tobytes())

            # Bind the character's texture
            texture.use(0)

            # Render the 6 vertices (2 triangles) for this character
            self.text_vao.render(
                moderngl.TRIANGLES,  # pylint: disable=no-member
                vertices=6,
            )

    def add_text_quad(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        color: Color,
        texture,
    ):
        """Add a textured quad for text rendering"""
        r, g, b, a = color.as_normalized()

        # UV coordinates are (0,0) to (1,1) for the full texture
        vertices = np.array(
            [
                # First triangle
                [x, y, 0.0, 0.0, r, g, b, a],  # Top-left
                [x + width, y, 1.0, 0.0, r, g, b, a],  # Top-right
                [x, y + height, 0.0, 1.0, r, g, b, a],  # Bottom-left
                # Second triangle
                [x + width, y, 1.0, 0.0, r, g, b, a],  # Top-right
                [x + width, y + height, 1.0, 1.0, r, g, b, a],  # Bottom-right
                [x, y + height, 0.0, 1.0, r, g, b, a],  # Bottom-left
            ],
            dtype=np.float32,
        )

        # Add to render queue with its texture
        self.text_render_queue.append((vertices, texture))

    def draw_text(
        self,
        text: str,
        position: Vec2,
        color: Color,
        font_size: int = 16,
        font_path: Optional[str] = None,
    ):
        """Draw text at the specified position"""
        self.font_renderer = self.font_manager.get_font_renderer(
            self.ctx, font_size, font_path
        )

        x, y = position.x, position.y

        for char in text:
            if char == " ":
                # Handle space character
                x += font_size * 0.5
                continue

            if char == "\n":
                # Handle newline
                y += font_size * 1.2
                x = position.x
                continue

            texture, metrics = self.font_renderer.get_char_texture(char)

            char_x = x + metrics["offset_x"]
            char_y = y + metrics["offset_y"]
            char_width = metrics["width"]
            char_height = metrics["height"]

            # Add textured quad for this character
            self.add_text_quad(char_x, char_y, char_width, char_height, color, texture)

            # Advance cursor
            x += metrics["advance"]

    def display(self):
        """Present the rendered frame"""
        self.flush()
        # Swap buffers if using windowed mode
        if self.window:
            glfw.swap_buffers(self.window)

    def tick(self, target_fps: float = 0) -> float:
        """Cap frame rate and return delta time in seconds"""
        now = time.time()
        elapsed = now - self.last_time
        target_frame_time = 1.0 / target_fps if target_fps > 0 else 0.0

        if target_frame_time > 0.0 and elapsed < target_frame_time:
            time.sleep(target_frame_time - elapsed)
            now = time.time()
            elapsed = now - self.last_time

        self.last_time = now
        return elapsed

    def vsync_tick(self) -> float:
        """Update timing with vsync enabled"""
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
        if hasattr(self, "index_buffer_gl"):
            self.index_buffer_gl.release()
        if hasattr(self, "vao_indexed"):
            self.vao_indexed.release()
        if hasattr(self, "vao_simple"):
            self.vao_simple.release()
        if hasattr(self, "program"):
            self.program.release()
        if hasattr(self, "fbo") and self.fbo:
            self.fbo.release()

        # Clean up text resources
        if hasattr(self, "text_vertex_buffer"):
            self.text_vertex_buffer.release()
        if hasattr(self, "text_vao"):
            self.text_vao.release()
        if hasattr(self, "text_program"):
            self.text_program.release()
        if hasattr(self, "font_renderer"):
            for texture in self.font_renderer.char_textures.values():
                texture.release()

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
        if isinstance(shape, Triangle):
            self.graphics_context.draw_triangle(shape)
        elif isinstance(shape, Rect):
            self.graphics_context.draw_rect(shape)
        elif isinstance(shape, Circle):
            self.graphics_context.draw_circle(shape)
        else:
            raise ValueError("Unsupported shape type")

    def draw_text(
        self,
        text: str,
        position: Vec2,
        color: Color,
        font_size: int = 16,
        font_path: Optional[str] = None,
    ):
        """Draw text at the given position"""
        self.graphics_context.draw_text(text, position, color, font_size, font_path)

    def display(self):
        """Display the frame"""
        self.graphics_context.display()

    def tick(self, target_fps: float) -> float:
        """Cap frame rate and return delta time in seconds"""
        return self.graphics_context.tick(target_fps)

    def vsync_tick(self) -> float:
        """Return delta time without a frame cap"""
        return self.graphics_context.vsync_tick()

    def cleanup(self):
        """Clean up resources"""
        self.graphics_context.cleanup()
