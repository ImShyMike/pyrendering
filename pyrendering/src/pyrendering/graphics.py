# pylint: disable=missing-function-docstring,missing-module-docstring

from typing import Callable, Optional, Tuple

from pyrendering.color import Color
from pyrendering.context import DrawModes, GraphicsContext, ResizeModes
from pyrendering.shapes import Circle, Rect, Shape, Texture, Triangle, Vec2


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

    def get_camera_position(self) -> Vec2:
        """Get the current camera position

        Returns:
            Vec2: Camera position
        """
        return self.graphics_context.camera.position

    def get_camera_zoom(self) -> float:
        """Get the current camera zoom level

        Returns:
            float: Camera zoom level
        """
        return self.graphics_context.camera.zoom

    def move_camera(self, delta: Vec2):
        """Move the camera by the given delta

        Args:
            delta (Vec2): Amount to move the camera
        """
        self.graphics_context.move_camera(delta)

    def set_camera_position(self, position: Vec2):
        """Set the camera position

        Args:
            position (Vec2): New camera position
        """
        self.graphics_context.set_camera_position(position)

    def set_camera_zoom(self, zoom: float):
        """Set the camera zoom level

        Args:
            zoom (float): New zoom level (1.0 = normal, >1.0 = zoomed in, <1.0 = zoomed out)
        """
        self.graphics_context.set_camera_zoom(zoom)

    def reset_camera(self):
        """Reset camera to default position and zoom"""
        self.graphics_context.reset_camera()

    def cleanup(self):
        """Clean up resources"""
        self.graphics_context.cleanup()
