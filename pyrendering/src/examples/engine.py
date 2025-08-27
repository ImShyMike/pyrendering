"""Simple example on how to use pyrendering."""

import traceback

from pyrendering import Color, Engine, Graphics, Point, Rect, Triangle, Vec2


def main():
    """Testest"""

    gfx = Graphics(
        800, 600, "My Game", standalone=False, vsync=True, resize_mode="letterbox"
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

    triangle_position = 400
    direction = 1

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
