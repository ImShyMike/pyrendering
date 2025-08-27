# pylint: disable=missing-function-docstring,missing-module-docstring

import time
from typing import Callable, Literal, Optional, Tuple, cast

import glfw
import moderngl
import numpy as np

from pyrendering.color import Color
from pyrendering.font import FontManager, FontRenderer
from pyrendering.shapes import Circle, Rect, Shape, Texture, Triangle, Vec2

NUM_VERTICES = 10000

ResizeModes = Literal["stretch", "letterbox", "ignore"]
DrawModes = Literal["fill", "wireframe", "points"]


def framebuffer_size_callback(window, width, height):
    graphics_ctx = glfw.get_window_user_pointer(window)
    if graphics_ctx:
        graphics_ctx.on_resize(width, height)


def key_callback(window, key, scancode, action, mods):
    graphics_ctx = glfw.get_window_user_pointer(window)
    if graphics_ctx:
        graphics_ctx.on_key(key, scancode, action, mods)


def mouse_button_callback(window, button, action, mods):
    graphics_ctx = glfw.get_window_user_pointer(window)
    if graphics_ctx:
        graphics_ctx.on_mouse_button(button, action, mods)


def cursor_position_callback(window, xpos, ypos):
    graphics_ctx = glfw.get_window_user_pointer(window)
    if graphics_ctx:
        graphics_ctx.on_mouse_move(xpos, ypos)


def scroll_callback(window, xoffset, yoffset):
    graphics_ctx = glfw.get_window_user_pointer(window)
    if graphics_ctx:
        graphics_ctx.on_scroll(xoffset, yoffset)


class DrawMode:
    """Vertex types"""

    TRIANGLE = 0
    LINE = 1
    POINT = 2


class GraphicsContext:
    """Graphics context handler"""

    def __init__(
        self,
        width: int,
        height: int,
        title: str = "pyrendering",
        standalone: bool = False,
        vsync: bool = True,
        resize_mode: ResizeModes = "stretch",
    ):
        self.width = width
        self.height = height
        self.original_width = width
        self.original_height = height
        self.title = title
        self.window = None
        self.resize_mode = resize_mode
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

            # Set this instance as the window's user pointer so the callback can access it
            glfw.set_window_user_pointer(self.window, self)

            # Register resize callback
            glfw.set_framebuffer_size_callback(self.window, framebuffer_size_callback)

            # Register input callbacks
            glfw.set_key_callback(self.window, key_callback)
            glfw.set_mouse_button_callback(self.window, mouse_button_callback)
            glfw.set_cursor_pos_callback(self.window, cursor_position_callback)
            glfw.set_scroll_callback(self.window, scroll_callback)

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

        # Create texture shader program for textures
        self.texture_program = self.ctx.program(
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
    fragColor = texColor * v_color;
}
""",
        )

        # Set resolution uniform
        u_resolution = cast(moderngl.Uniform, self.program["u_resolution"])
        u_resolution.value = (float(width), float(height))

        # Set resolution uniform for text shader
        text_u_resolution = cast(moderngl.Uniform, self.text_program["u_resolution"])
        text_u_resolution.value = (float(width), float(height))

        # Set texture uniform for text shader
        text_u_texture = cast(moderngl.Uniform, self.text_program["u_texture"])
        text_u_texture.value = 0

        # Set resolution uniform for texture shader
        texture_u_resolution = cast(
            moderngl.Uniform, self.texture_program["u_resolution"]
        )
        texture_u_resolution.value = (float(width), float(height))

        # Set texture uniform for texture shader
        texture_u_texture = cast(moderngl.Uniform, self.texture_program["u_texture"])
        texture_u_texture.value = 0

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

        # Create texture VAO
        self.texture_vao = self.ctx.vertex_array(
            self.texture_program,
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
        self.point_vertices = np.empty((0, 6), dtype=np.float32)  # Non-indexed points

        # Indexed rendering data structures
        self.indexed_vertices = np.empty(
            (0, 6), dtype=np.float32
        )  # Vertices for indexed rendering
        self.triangle_indices = np.empty(0, dtype=np.uint32)  # Triangle indices
        self.line_indices = np.empty(0, dtype=np.uint32)  # Line indices
        self.point_indices = np.empty(0, dtype=np.uint32)  # Point indices

        # Text rendering data
        self.text_render_queue = []  # List of (vertices, texture) pairs

        # Texture rendering data
        self.texture_render_queue = []  # List of (vertices, texture) pairs

        self.font_manager = FontManager()

        self.vertex_count = 0  # Track current vertex count for indexing

        # Create framebuffer for offscreen rendering if needed
        if standalone:
            self.fbo = self.ctx.framebuffer(
                color_attachments=[self.ctx.texture((width, height), 4)]
            )
        else:
            self.fbo = None

        # Callbacks
        self.key_callback = None
        self.mouse_button_callback = None
        self.mouse_move_callback = None
        self.scroll_callback = None

    def set_key_callback(self, callback: Callable):
        """Set key event callback"""
        self.key_callback = callback

    def set_mouse_button_callback(self, callback: Callable):
        """Set mouse button event callback"""
        self.mouse_button_callback = callback

    def set_mouse_move_callback(self, callback: Callable):
        """Set mouse move event callback"""
        self.mouse_move_callback = callback

    def set_scroll_callback(self, callback: Callable):
        """Set scroll event callback"""
        self.scroll_callback = callback

    def update_shader_uniforms(self, width: int, height: int):
        """Update shader uniforms with new resolution"""
        u_resolution = cast(moderngl.Uniform, self.program["u_resolution"])
        u_resolution.value = (float(width), float(height))

        text_u_resolution = cast(moderngl.Uniform, self.text_program["u_resolution"])
        text_u_resolution.value = (float(width), float(height))

        texture_u_resolution = cast(
            moderngl.Uniform, self.texture_program["u_resolution"]
        )
        texture_u_resolution.value = (float(width), float(height))

    def on_resize(self, width: int, height: int):
        """Handle window resize based on resize_mode"""

        if self.resize_mode == "ignore":
            return

        if self.resize_mode == "letterbox":
            original_aspect = self.original_width / self.original_height
            new_aspect = width / height

            if new_aspect > original_aspect:
                viewport_height = height
                viewport_width = int(height * original_aspect)
                viewport_x = (width - viewport_width) // 2
                viewport_y = 0
            else:
                viewport_width = width
                viewport_height = int(width / original_aspect)
                viewport_x = 0
                viewport_y = (height - viewport_height) // 2

            self.ctx.viewport = (
                viewport_x,
                viewport_y,
                viewport_width,
                viewport_height,
            )

        else:  # "stretch" mode (default)
            self.width = width
            self.height = height
            self.ctx.viewport = (0, 0, width, height)
            self.update_shader_uniforms(width, height)

        if self.fbo:
            self.fbo.release()
            self.fbo = self.ctx.framebuffer(
                color_attachments=[self.ctx.texture((width, height), 4)]
            )

    def on_key(self, key, scancode, action, mods):
        """Handle key events"""
        if self.key_callback and callable(self.key_callback):
            self.key_callback(key, scancode, action, mods)

    def on_mouse_button(self, button, action, mods):
        """Handle mouse button events"""
        # Make sure the mouse is actually inside the window
        xpos, ypos = glfw.get_cursor_pos(self.window)
        if self.window and self.resize_mode == "letterbox":
            viewport = self.ctx.viewport
            if not (
                viewport[0] <= xpos <= viewport[0] + viewport[2]
                and viewport[1] <= ypos <= viewport[1] + viewport[3]
            ):
                return

        if self.resize_mode == "letterbox":
            viewport = self.ctx.viewport

            # Adjust for aspect ratio difference
            original_aspect = self.original_width / self.original_height
            current_aspect = viewport[2] / viewport[3]
            if current_aspect > original_aspect:
                scale = viewport[3] / self.original_height
            else:
                scale = viewport[2] / self.original_width

            xpos = int((xpos - viewport[0]) / scale)
            ypos = int((ypos - viewport[1]) / scale)

        if self.mouse_button_callback and callable(self.mouse_button_callback):
            self.mouse_button_callback(button, xpos, ypos, action, mods)

    def on_mouse_move(self, xpos: int, ypos: int):
        """Handle mouse movement events"""
        # Make sure the mouse is actually inside the window
        if self.window and self.resize_mode == "letterbox":
            viewport = self.ctx.viewport
            if not (
                viewport[0] <= xpos <= viewport[0] + viewport[2]
                and viewport[1] <= ypos <= viewport[1] + viewport[3]
            ):
                return

        if self.resize_mode == "letterbox":
            viewport = self.ctx.viewport

            # Adjust for aspect ratio difference
            original_aspect = self.original_width / self.original_height
            current_aspect = viewport[2] / viewport[3]
            if current_aspect > original_aspect:
                scale = viewport[3] / self.original_height
            else:
                scale = viewport[2] / self.original_width

            xpos = int((xpos - viewport[0]) / scale)
            ypos = int((ypos - viewport[1]) / scale)

        if self.mouse_move_callback and callable(self.mouse_move_callback):
            self.mouse_move_callback(xpos, ypos)

    def on_scroll(self, xoffset: float, yoffset: float):
        """Handle scroll events"""
        # Make sure the mouse is actually inside the window
        xpos, ypos = glfw.get_cursor_pos(self.window)
        if self.window and self.resize_mode == "letterbox":
            viewport = self.ctx.viewport
            if not (
                viewport[0] <= xpos <= viewport[0] + viewport[2]
                and viewport[1] <= ypos <= viewport[1] + viewport[3]
            ):
                return

        if self.scroll_callback and callable(self.scroll_callback):
            self.scroll_callback(xoffset, yoffset, xpos, ypos)

    def clear(self, color: Color):
        """Clear the screen with a color"""
        normalized = color.as_normalized()
        self.ctx.clear(
            normalized[0],
            normalized[1],
            normalized[2],
            normalized[3],
            viewport=self.ctx.viewport,
        )

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
        elif draw_mode == DrawMode.POINT:
            self.point_vertices = np.append(self.point_vertices, [vertex], axis=0)

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

    def add_point_indices(self, indices: np.ndarray):
        """Add point indices for indexed rendering"""
        self.point_indices = np.append(self.point_indices, indices.astype(np.uint32))

    def draw_triangle(self, triangle: Triangle):
        """Draw a triangle using simple rendering"""
        p1, p2, p3 = triangle.p1, triangle.p2, triangle.p3

        self.add_vertex_simple(p1.x, p1.y, p1.color, DrawMode.TRIANGLE)
        self.add_vertex_simple(p2.x, p2.y, p2.color, DrawMode.TRIANGLE)
        self.add_vertex_simple(p3.x, p3.y, p3.color, DrawMode.TRIANGLE)

    def draw_triangle_wireframe(self, triangle: Triangle):
        """Draw a triangle as wireframe using line mode"""
        p1, p2, p3 = triangle.p1, triangle.p2, triangle.p3

        # Draw three lines forming the triangle
        self.draw_line(p1.x, p1.y, p2.x, p2.y, p1.color)
        self.draw_line(p2.x, p2.y, p3.x, p3.y, p2.color)
        self.draw_line(p3.x, p3.y, p1.x, p1.y, p3.color)

    def draw_triangle_points(self, triangle: Triangle):
        """Draw a triangle as points using point mode"""
        p1, p2, p3 = triangle.p1, triangle.p2, triangle.p3

        self.draw_point(p1.x, p1.y, p1.color)
        self.draw_point(p2.x, p2.y, p2.color)
        self.draw_point(p3.x, p3.y, p3.color)

    def draw_rect(self, rect: Rect):
        """Draw a rectangle using efficient indexed rendering"""
        # Add the 4 vertices for the rectangle
        v0 = self.add_indexed_vertex(*rect.p1.unpack())
        v1 = self.add_indexed_vertex(*rect.p2.unpack())
        v2 = self.add_indexed_vertex(*rect.p3.unpack())
        v3 = self.add_indexed_vertex(*rect.p4.unpack())

        # Two triangles for filled rectangle: (v0,v1,v2) and (v0,v2,v3)
        triangle_indices = np.array([v0, v1, v2, v0, v2, v3], dtype=np.uint32)
        self.add_triangle_indices(triangle_indices)

    def draw_rect_wireframe(self, rect: Rect):
        """Draw a rectangle as wireframe using line mode"""
        v0 = self.add_indexed_vertex(*rect.p1.unpack())
        v1 = self.add_indexed_vertex(*rect.p2.unpack())
        v2 = self.add_indexed_vertex(*rect.p3.unpack())
        v3 = self.add_indexed_vertex(*rect.p4.unpack())

        # Four lines for rectangle outline: v0->v1, v1->v2, v2->v3, v3->v0
        line_indices = np.array([v0, v1, v1, v2, v2, v3, v3, v0], dtype=np.uint32)
        self.add_line_indices(line_indices)

    def draw_rect_points(self, rect: Rect):
        """Draw a rectangle as points using point mode"""
        self.draw_point(rect.p1.x, rect.p1.y, rect.p1.color)
        self.draw_point(rect.p2.x, rect.p2.y, rect.p2.color)
        self.draw_point(rect.p3.x, rect.p3.y, rect.p3.color)
        self.draw_point(rect.p4.x, rect.p4.y, rect.p4.color)

    def draw_line(self, x1: float, y1: float, x2: float, y2: float, color: Color):
        """Draw a line between two points"""
        self.add_vertex_simple(x1, y1, color, DrawMode.LINE)
        self.add_vertex_simple(x2, y2, color, DrawMode.LINE)

    def draw_point(self, x: float, y: float, color: Color):
        """Draw a single point"""
        self.add_vertex_simple(x, y, color, DrawMode.POINT)

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

    def draw_circle_wireframe(self, circle: Circle):
        """Draw a circle as wireframe using line mode"""
        center = circle.center.data
        radius = circle.radius
        color = circle.color
        segments = circle.segments

        # Generate angles for the circle
        angles = np.linspace(0, 2 * np.pi, segments, endpoint=False)
        offsets = np.stack((np.cos(angles), np.sin(angles)), axis=1) * radius

        # Create line segments for the circle outline
        for i in range(segments):
            v1 = center + offsets[i]
            v2 = center + offsets[(i + 1) % segments]

            self.add_vertex_simple(v1[0], v1[1], color, DrawMode.LINE)
            self.add_vertex_simple(v2[0], v2[1], color, DrawMode.LINE)

    def draw_circle_points(self, circle: Circle):
        """Draw a circle as points using point mode"""
        center = circle.center.data
        radius = circle.radius
        color = circle.color
        segments = circle.segments

        # Generate angles for the circle
        angles = np.linspace(0, 2 * np.pi, segments, endpoint=False)
        offsets = np.stack((np.cos(angles), np.sin(angles)), axis=1) * radius

        # Add circumference points
        for offset in offsets:
            point = center + offset
            self.add_vertex_simple(point[0], point[1], color, DrawMode.POINT)

    def draw_texture(self, texture: Texture):
        """Draw a texture mapped to an arbitrary quadrilateral"""
        # Load the texture if not already loaded
        gl_texture = texture.load_texture(self.ctx)

        # Add textured quadrilateral for this texture
        self.add_texture_quadrilateral(
            texture.v1,
            texture.v2,
            texture.v3,
            texture.v4,
            gl_texture,
            color=texture.color,
        )

    def begin_frame(self):
        """Begin a new frame"""
        # Clear all vertex data
        self.triangle_vertices = self.triangle_vertices[:0]
        self.line_vertices = self.line_vertices[:0]
        self.point_vertices = self.point_vertices[:0]
        self.indexed_vertices = self.indexed_vertices[:0]
        self.triangle_indices = self.triangle_indices[:0]
        self.line_indices = self.line_indices[:0]
        self.point_indices = self.point_indices[:0]
        self.vertex_count = 0

        # Clear text render queue
        self.text_render_queue.clear()

        # Clear texture render queue
        self.texture_render_queue.clear()

        if self.fbo:
            self.fbo.use()

    def flush(self):
        """Render all batched geometry"""
        # Render indexed geometry first (rectangles)
        if self.indexed_vertices.size > 0:
            # Upload vertex data
            self.vertex_buffer.write(self.indexed_vertices.tobytes())

            # Combine all indices into a single buffer
            all_indices = []
            if self.triangle_indices.size > 0:
                all_indices.append(self.triangle_indices)
            if self.line_indices.size > 0:
                all_indices.append(self.line_indices)
            if self.point_indices.size > 0:
                all_indices.append(self.point_indices)

            if all_indices:
                combined_indices = np.concatenate(all_indices)
                self.index_buffer_gl.write(combined_indices.tobytes())

                current_offset = 0

                # Render triangles with indices
                if self.triangle_indices.size > 0:
                    triangle_count = len(self.triangle_indices)
                    self.vao_indexed.render(
                        moderngl.TRIANGLES,  # pylint: disable=no-member
                        vertices=triangle_count,
                        first=current_offset,
                        instances=1,
                    )
                    current_offset += triangle_count

                # Render lines with indices
                if self.line_indices.size > 0:
                    line_count = len(self.line_indices)
                    self.vao_indexed.render(
                        moderngl.LINES,  # pylint: disable=no-member
                        vertices=line_count,
                        first=current_offset,
                        instances=1,
                    )
                    current_offset += line_count

                # Render points with indices
                if self.point_indices.size > 0:
                    point_count = len(self.point_indices)
                    self.vao_indexed.render(
                        moderngl.POINTS,  # pylint: disable=no-member
                        vertices=point_count,
                        first=current_offset,
                        instances=1,
                    )

        # Render simple geometry (triangles, lines, and points without indices)
        total_simple_vertices = (
            len(self.triangle_vertices)
            + len(self.line_vertices)
            + len(self.point_vertices)
        )
        if total_simple_vertices > 0:
            # Combine all simple vertices
            all_simple_vertices = []
            if len(self.triangle_vertices) > 0:
                all_simple_vertices.append(self.triangle_vertices)
            if len(self.line_vertices) > 0:
                all_simple_vertices.append(self.line_vertices)
            if len(self.point_vertices) > 0:
                all_simple_vertices.append(self.point_vertices)

            combined_vertices = np.concatenate(all_simple_vertices)
            self.vertex_buffer.write(combined_vertices.tobytes())

            current_start = 0

            # Render triangles
            if len(self.triangle_vertices) > 0:
                triangle_count = len(self.triangle_vertices)
                self.vao_simple.render(
                    moderngl.TRIANGLES,  # pylint: disable=no-member
                    vertices=triangle_count,
                    first=current_start,
                )
                current_start += triangle_count

            # Render lines
            if len(self.line_vertices) > 0:
                line_count = len(self.line_vertices)
                self.vao_simple.render(
                    moderngl.LINES,  # pylint: disable=no-member
                    vertices=line_count,
                    first=current_start,
                )
                current_start += line_count

            # Render points
            if len(self.point_vertices) > 0:
                point_count = len(self.point_vertices)
                self.vao_simple.render(
                    moderngl.POINTS,  # pylint: disable=no-member
                    vertices=point_count,
                    first=current_start,
                )

        # Render text
        self.flush_text()

        # Render textures
        self.flush_textures()

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

    def flush_textures(self):
        """Render all batched textures"""
        if not self.texture_render_queue:
            return

        # Render each texture quad separately with its own texture
        for vertices, texture in self.texture_render_queue:
            # Upload vertex data for this quad
            self.text_vertex_buffer.write(vertices.tobytes())

            # Bind the texture
            texture.use(0)

            # Render the 6 vertices (2 triangles) for this quad
            self.texture_vao.render(
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

    def add_texture_quadrilateral(
        self,
        p1: Vec2,
        p2: Vec2,
        p3: Vec2,
        p4: Vec2,
        texture,
        color: Color,
    ):
        """Add a textured quadrilateral for texture rendering on arbitrary shapes"""
        r, g, b, a = color.as_normalized()

        # Triangle 1: p1, p2, p3
        vertices_triangle1 = np.array(
            [
                [p1.x, p1.y, 0.0, 0.0, r, g, b, a],  # p1 -> (0,0)
                [p2.x, p2.y, 1.0, 0.0, r, g, b, a],  # p2 -> (1,0)
                [p3.x, p3.y, 1.0, 1.0, r, g, b, a],  # p3 -> (1,1)
            ],
            dtype=np.float32,
        )

        # Triangle 2: p1, p3, p4
        vertices_triangle2 = np.array(
            [
                [p1.x, p1.y, 0.0, 0.0, r, g, b, a],  # p1 -> (0,0)
                [p3.x, p3.y, 1.0, 1.0, r, g, b, a],  # p3 -> (1,1)
                [p4.x, p4.y, 0.0, 1.0, r, g, b, a],  # p4 -> (0,1)
            ],
            dtype=np.float32,
        )

        # Combine both triangles into one vertex array
        vertices = np.concatenate([vertices_triangle1, vertices_triangle2])

        # Add to render queue with its texture
        self.texture_render_queue.append((vertices, texture))

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

        if target_fps != 0:
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
        if hasattr(self, "texture_vao"):
            self.texture_vao.release()
        if hasattr(self, "texture_program"):
            self.texture_program.release()
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
        resize_mode: ResizeModes = "stretch",
    ):
        """Graphics handler

        Args:
            width (int, optional): Window width. Defaults to 800.
            height (int, optional): Window height. Defaults to 600.
            title (str, optional): Window title. Defaults to "Graphics".
            standalone (bool, optional): Standalone mode (no window). Defaults to False.
            vsync (bool, optional): Use vsync. Defaults to True.
            resize_mode (ResizeModes, optional): Resize mode. \
                Can be "stretch", "letterbox", or "ignore". Defaults to "stretch".
        """
        self.graphics_context = GraphicsContext(
            width, height, title, standalone, vsync, resize_mode
        )

    def set_key_callback(self, callback: Callable):
        """Set key event callback

        Args:
            callback (Callable): Callback function with signature \
                (key: int, scancode: int, action: int, mods: int) -> None
        """
        self.graphics_context.set_key_callback(callback)

    def set_mouse_button_callback(self, callback: Callable):
        """Set mouse button event callback
        
        Args:
            callback (Callable): Callback function with signature \
                (button: int, xpos: float, ypos: float, action: int, mods: int) -> None
        """
        self.graphics_context.set_mouse_button_callback(callback)

    def set_mouse_move_callback(self, callback: Callable):
        """Set mouse move event callback
        
        Args:
            callback (Callable): Callback function with signature \
                (xpos: float, ypos: float) -> None
        """
        self.graphics_context.set_mouse_move_callback(callback)

    def set_scroll_callback(self, callback: Callable):
        """Set scroll event callback
        
        Args:
            callback (Callable): Callback function with signature \
                (xoffset: float, yoffset: float, xpos: float, ypos: float) -> None
        """
        self.graphics_context.set_scroll_callback(callback)

    def get_monitor_mode(self) -> Tuple[int, int, int]:
        """Get the current monitor mode

        Returns:
            Tuple[int, int, int]: Monitor mode (width, height, refresh rate)
        """
        return self.graphics_context.get_monitor_mode()

    def should_close(self) -> bool:
        """Check if the window should close

        Returns:
            bool: Should close
        """
        return self.graphics_context.should_close()

    def poll_events(self):
        """Poll for window events"""
        self.graphics_context.poll_events()

    def begin_frame(self):
        """Begin a new frame"""
        self.graphics_context.begin_frame()

    def clear(self, color: Color):
        """Clear the screen with a color

        Args:
            color (Color): Color to clear the screen with
        """
        self.graphics_context.clear(color)

    def draw(self, shape: Shape, draw_mode: DrawModes = "fill"):
        """Draw a shape with the specified draw mode

        Args:
            shape (Shape): Shape to draw
            draw_mode (DrawModes, optional): Draw mode. \
                Can be "fill", "wireframe", or "points". Defaults to "fill".

        Raises:
            ValueError: Unsupported shape type
            ValueError: Unsupported draw mode
            ValueError: Texture only supports 'fill' draw mode
        """
        if isinstance(shape, Triangle):
            if draw_mode == "fill":
                self.graphics_context.draw_triangle(shape)
            elif draw_mode == "wireframe":
                self.graphics_context.draw_triangle_wireframe(shape)
            elif draw_mode == "points":
                self.graphics_context.draw_triangle_points(shape)
            else:
                raise ValueError(f"Unsupported draw mode: {draw_mode}")
        elif isinstance(shape, Texture):
            if draw_mode == "fill":
                self.graphics_context.draw_texture(shape)
            else:
                raise ValueError(
                    f"Texture only supports 'fill' draw mode, got: {draw_mode}"
                )
        elif isinstance(shape, Rect):
            if draw_mode == "fill":
                self.graphics_context.draw_rect(shape)
            elif draw_mode == "wireframe":
                self.graphics_context.draw_rect_wireframe(shape)
            elif draw_mode == "points":
                self.graphics_context.draw_rect_points(shape)
            else:
                raise ValueError(f"Unsupported draw mode: {draw_mode}")
        elif isinstance(shape, Circle):
            if draw_mode == "fill":
                self.graphics_context.draw_circle(shape)
            elif draw_mode == "wireframe":
                self.graphics_context.draw_circle_wireframe(shape)
            elif draw_mode == "points":
                self.graphics_context.draw_circle_points(shape)
            else:
                raise ValueError(f"Unsupported draw mode: {draw_mode}")
        else:
            raise ValueError(f"Unsupported shape type: {shape}")

    def draw_text(
        self,
        text: str,
        position: Vec2,
        color: Color,
        font_size: int = 16,
        font_path: Optional[str] = None,
    ):
        """Draw text at the specified position

        Args:
            text (str): The text to draw
            position (Vec2): Position to draw the text
            color (Color): Color of the text
            font_size (int, optional): Font size. Defaults to 16.
            font_path (Optional[str], optional): Font path. None for default font. Defaults to None.
        """
        self.graphics_context.draw_text(text, position, color, font_size, font_path)

    def display(self):
        """Display the frame"""
        self.graphics_context.display()

    def tick(self, target_fps: float = 0) -> float:
        """Cap frame rate and return delta time in seconds

        Args:
            target_fps (float, optional): Target frames per second. 0 for uncapped. Defaults to 0.

        Returns:
            float: Delta time in seconds
        """
        return self.graphics_context.tick(target_fps)

    def cleanup(self):
        """Clean up resources"""
        self.graphics_context.cleanup()
