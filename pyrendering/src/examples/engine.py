"""Simple example on how to use pyrendering."""

import traceback

from pyrendering import Circle, Color, Engine, Graphics, Point, Rect, Triangle, Vec2


def main():
    """Testest"""

    gfx = Graphics(800, 600, "My Game", standalone=False, vsync=True, resize_mode="letterbox")
    engine = Engine(gfx)

    current_fps = 0
    time_since_last_fps_update = 0.5

    rect_id = engine.add_shape(
        Rect.from_dimensions(350, 100, 150, 100, Color.from_hex("#ca2353"), True)
    )

    try:
        # Run until window closes
        while not gfx.should_close():
            delta_time = gfx.vsync_tick()
            time_since_last_fps_update += delta_time

            gfx.poll_events()
            gfx.begin_frame()

            gfx.clear(Color.from_hex("#1a1a2e"))  # Dark blue background

            engine.rotate_shape(rect_id, delta_time)

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
