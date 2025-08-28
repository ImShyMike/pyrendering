"""Camera example showing how to use camera offset functionality."""

import traceback

from pyrendering import Color, Graphics, Rect, Vec2


def main():
    """Camera example"""

    # Initialize graphics
    gfx = Graphics(
        800,
        600,
        "Camera Example",
        standalone=False,
        vsync=True,
        resize_mode="letterbox",
    )
    frame_count = 0
    target_fps = gfx.get_monitor_mode()[2] or 60  # Vscync refresh rate or 60

    # Define some colors
    red = Color.from_rgb(255, 0, 0)
    green = Color.from_rgb(0, 255, 0)
    blue = Color.from_rgb(0, 0, 255)

    try:
        # Run until window closes
        while not gfx.should_close():
            delta_time = gfx.tick()

            frame_count += 1

            gfx.poll_events()
            gfx.begin_frame()

            gfx.clear(Color.from_hex("#1a1a2e"))  # Dark blue background

            # Move camera based on frame count
            camera_speed = 50.0  # pixels per second
            camera_x = (frame_count * 0.01 * camera_speed) % 400 - 200
            camera_y = 0.0

            # Set camera position to create a moving effect
            gfx.set_camera_position(Vec2(camera_x, camera_y))

            # Draw some shapes to help visualize camera movement
            gfx.draw(Rect.from_dimensions(100, 100, 100, 100, red), draw_mode="fill")

            gfx.draw(Rect.from_dimensions(300, 200, 150, 80, green), draw_mode="fill")

            gfx.draw(Rect.from_dimensions(500, 150, 120, 120, blue), draw_mode="fill")

            # Draw a stationary UI element (not affected by camera)
            gfx.draw_text(
                f"Camera X: {camera_x:.1f}",
                Vec2(10, 30) + gfx.get_camera_position(),
                Color.from_rgb(255, 255, 255),
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
