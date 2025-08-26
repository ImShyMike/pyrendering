"""Simple example on how to use pyrendering."""

import traceback

from pyrendering import Circle, Color, Graphics, Rect, RoundedRect, Triangle, Vec2


def main():
    """Testest"""

    # Initialize graphics
    gfx = Graphics(800, 600, "My Game", standalone=False, vsync=True)
    frame_count = 0
    target_fps = gfx.get_monitor_mode()[2] or 60  # Vscync refresh rate or 60

    try:
        # Run until window closes
        while not gfx.should_close():
            # vsync_tick() does not limit fps, it just returns the delta time
            delta_time = gfx.vsync_tick()

            frame_count += 1

            gfx.poll_events()
            gfx.begin_frame()

            gfx.clear(Color.from_hex("#1a1a2e"))  # Dark blue background

            # Draw some shapes
            gfx.draw(
                Rect(100, 100, 200, 150, Color.from_hex("#e94560"))
            )  # Red rectangle
            gfx.draw(
                Circle(Vec2(400, 300), 50, Color.from_hex("#0f3460"))
            )  # Blue circle

            # Draw outline rectangle
            gfx.draw(Rect(350, 100, 150, 100, Color.from_hex("#ffffff"), False))

            # Draw a semi-transparent triangle
            gfx.draw(
                Triangle(
                    Vec2(400, 400),
                    Vec2(500, 300),
                    Vec2(300, 300),
                    Color.from_hex8("#00ff0077"),
                )
            )

            # Draw a rounded rectangle
            gfx.draw(RoundedRect(550, 100, 200, 100, radius=20, color=Color.from_hex("#ff9900")))

            gfx.display()

            if frame_count % target_fps == 0:  # Print data every second
                print(f"Frame {frame_count} rendered. FPS: {(1 / delta_time):.2f}")

    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"Error during rendering: {e}")
        traceback.print_exc()
    finally:
        print("Cleaning up...")
        gfx.cleanup()
        print("Cleanup completed")


if __name__ == "__main__":
    main()
