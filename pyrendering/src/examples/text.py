"""Simple example on how to use pyrendering."""

import traceback

from pyrendering import Color, Graphics, Vec2


def main():
    """Testest"""

    # Initialize graphics
    gfx = Graphics(800, 600, "My Game", standalone=False, vsync=False)
    frame_count = 0
    current_fps = 1

    try:
        # Run until window closes
        while not gfx.should_close():
            # vsync_tick() does not limit fps, it just returns the delta time
            delta_time = gfx.vsync_tick()

            frame_count += 1

            gfx.poll_events()
            gfx.begin_frame()

            gfx.clear(Color.from_hex("#1a1a2e"))  # Dark blue background

            gfx.draw_text(
                "abc defg hijklmnopqrs tuvwxyz 0123456789 !@#$%^&*()",
                Vec2(50, 50),
                Color.from_rgb(255, 255, 255),
                24,
            )
            gfx.draw_text(
                "FPS: " + str(current_fps),
                Vec2(0, 0),
                Color.from_rgb(0, 255, 0),
                16,
            )

            if frame_count % current_fps == 0:
                current_fps = int(1 / delta_time)

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
