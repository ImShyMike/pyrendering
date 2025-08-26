"""Simple example on how to use pyrendering."""

import traceback

from pyrendering import Circle, Color, Graphics, Rect, Vec2


def main():
    """Testest"""

    # Initialize graphics
    gfx = Graphics(800, 600, "My Game", standalone=False, vsync=True)
    frame_count = 0
    target_fps = 60

    try:
        # Run until window closes
        while not gfx.should_close():
            delta_time = gfx.tick(target_fps)

            frame_count += 1

            gfx.poll_events()
            gfx.begin_frame()

            gfx.clear(Color.from_hex("#1a1a2e"))  # Dark blue background

            # Draw some shapes
            gfx.draw_rect(
                Rect(100, 100, 200, 150), Color.from_hex("#e94560")
            )  # Red rectangle
            gfx.draw_circle(
                Circle(Vec2(400, 300), 50), Color.from_hex("#0f3460")
            )  # Blue circle

            # Draw outline rectangle
            gfx.draw_rect(
                Rect(350, 100, 150, 100), Color.from_hex("#ffffff"), filled=False
            )

            gfx.display()

            if frame_count % target_fps == 0:  # Print data every second
                print(
                    f"Frame {frame_count} rendered. FPS: {(target_fps * delta_time):.2f}"
                )

    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"Error during rendering: {e}")
        traceback.print_exc()
    finally:
        print("Cleaning up...")
        gfx.cleanup()
        print("Cleanup completed")


if __name__ == "__main__":
    main()
