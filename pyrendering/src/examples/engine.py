"""Simple example on how to use pyrendering."""

import traceback

from pyrendering import Circle, Color, Engine, Graphics, Point, Rect, Triangle, Vec2


def main():
    """Engine example"""

    gfx = Graphics(
        800, 600, "Engine demo", standalone=False, vsync=True, resize_mode="letterbox"
    )
    engine = Engine(gfx)

    current_fps = 0
    time_since_last_fps_update = 0.5

    rect_id = engine.add_shape(
        Rect.from_dimensions(350, 100, 150, 100, Color.from_hex("#ca2353")),
        draw_mode="wireframe",
    )

    triangle_id = engine.add_shape(
        Triangle(
            Point(Vec2(400, 400), Color.from_rgb(255, 0, 0)),
            Point(Vec2(500, 300), Color.from_rgb(0, 255, 0)),
            Point(Vec2(300, 300), Color.from_rgb(0, 0, 255)),
        ),
        draw_mode="fill",
    )

    cursor_circle_id = engine.add_shape(
        Circle(Vec2(0, 0), 10, Color.from_hex("#ffffff")), draw_mode="fill"
    )

    rect2_id = engine.add_shape(
        Rect.from_dimensions(100, 100, 50, 50, Color.from_hex("#00ff00")),
        draw_mode="fill",
    )

    triangle_position = 400
    direction = 1
    current_mouse_status = 0

    gfx.set_key_callback(
        lambda key, scancode, action, mods: print(
            f"Key event - key: {key}, scancode: {scancode}, action: {action}, mods: {mods}"
        )
    )

    def _on_mouse_button(button, _xpos, _ypos, action, _mods):
        nonlocal current_mouse_status
        if button == 0:
            current_mouse_status = action

    gfx.set_mouse_button_callback(_on_mouse_button)

    gfx.set_mouse_move_callback(
        lambda xpos, ypos: engine.move_shape_to(cursor_circle_id, Vec2(xpos, ypos))
    )

    def _on_scroll(xoffset, yoffset, _xpos, _ypos):
        engine.translate_shape(rect2_id, Vec2(-xoffset * 10, -yoffset * 10))

    gfx.set_scroll_callback(_on_scroll)

    try:
        # Run until window closes
        while not gfx.should_close():
            delta_time = gfx.tick()
            time_since_last_fps_update += delta_time

            gfx.poll_events()
            gfx.begin_frame()

            gfx.clear(Color.from_hex("#1a1a2e"))  # Dark blue background

            engine.rotate_shape(rect_id, delta_time)

            # Update triangle position
            triangle_position = engine.lerp(
                triangle_position, triangle_position + direction * 100, delta_time
            )

            # Reverse direction if limits are reached
            if triangle_position >= 500 or triangle_position <= 300:
                direction *= -1

            # Move triangle to new position
            engine.move_shape_to(triangle_id, Vec2(triangle_position, 400))

            # Change color based on mouse click status
            if current_mouse_status == 1:
                engine.update_shape_color(cursor_circle_id, Color.from_hex("#ff0000"))
            else:
                engine.update_shape_color(cursor_circle_id, Color.from_hex("#ffffff"))

            engine.render()

            if time_since_last_fps_update >= 1.0:
                current_fps = int(1.0 / delta_time)
                time_since_last_fps_update = 0.0

            gfx.draw_text(
                "FPS: " + str(current_fps),
                Vec2(0, 0),
                Color.from_rgb(0, 255, 0),
                16,
            )

            gfx.display()

    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"Error during rendering: {e}")
        traceback.print_exc()
    finally:
        print("Cleaning up...")
        gfx.cleanup()
        print("Cleanup completed")


if __name__ == "__main__":
    main()
