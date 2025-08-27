"""Simple example on how to use pyrendering."""

import traceback

from pyrendering import Circle, Color, Graphics, Point, Rect, Triangle, Vec2


def main():
    """Testest"""

    # Initialize graphics
    gfx = Graphics(800, 600, "My Game", standalone=False, vsync=True)
    frame_count = 0
    target_fps = gfx.get_monitor_mode()[2] or 60  # Vscync refresh rate or 60

    # Define some colors
    red = Color.from_rgb(255, 0, 0)
    green = Color.from_rgb(0, 255, 0)
    blue = Color.from_rgb(0, 0, 255)

    semi_transparent_red = red.with_alpha(128)
    semi_transparent_green = green.with_alpha(128)

    try:
        # Run until window closes
        while not gfx.should_close():
            # vsync_tick() does not limit fps, it just returns the delta time
            delta_time = gfx.vsync_tick()

            frame_count += 1

            gfx.poll_events()
            gfx.begin_frame()

            gfx.clear(Color.from_hex("#1a1a2e"))  # Dark blue background

            # Draw a blue circle
            gfx.draw(Circle(Vec2(150, 350), 50, Color.from_hex("#0f3460")))

            # Draw a white outline rectangle
            gfx.draw(
                Rect.from_dimensions(
                    350, 100, 150, 100, Color.from_hex("#ffffff")
                ), draw_mode="wireframe"
            )

            # Draw an RGB triangle
            gfx.draw(
                Triangle(
                    Point(Vec2(400, 400), red),
                    Point(Vec2(500, 300), green),
                    Point(Vec2(300, 300), blue),
                )
            )

            # Draw a semi-transparent filled rectangle
            gfx.draw(
                Rect(
                    Point(Vec2(600, 100), semi_transparent_green),
                    Point(Vec2(700, 100), semi_transparent_green),
                    Point(Vec2(700, 200), semi_transparent_red),
                    Point(Vec2(600, 200), semi_transparent_red),
                )
            )

            # Draw a filled green rectangle
            gfx.draw(
                Rect(
                    Point(Vec2(150, 150), green),
                    Point(Vec2(250, 150), green),
                    Point(Vec2(300, 250), green),
                    Point(Vec2(100, 250), green),
                )
            )

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
