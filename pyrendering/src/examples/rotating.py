"""Simple example on how to use pyrendering."""

import traceback

import numpy as np

from pyrendering import Color, Graphics, Point, Rect, Vec2


def main():
    """Rotating rectangle example"""

    # Initialize graphics
    gfx = Graphics(800, 600, "Rotating cube", standalone=False, vsync=True)
    frame_count = 0
    target_fps = gfx.get_monitor_mode()[2] or 60  # Vscync refresh rate or 60

    semi_transparent_red = Color.from_rgba(255, 0, 0, 128)

    try:
        # Run until window closes
        while not gfx.should_close():
            delta_time = gfx.tick()

            frame_count += 1

            gfx.poll_events()
            gfx.begin_frame()

            gfx.clear(Color.from_hex("#1a1a2e"))  # Dark blue background

            # Draw a semi-transparent filled rotating rectangle
            base_vertices = np.array(
                [
                    [300, 200],
                    [500, 200],
                    [500, 400],
                    [300, 400],
                ]
            )
            rotated_vertices = []
            angle = frame_count * 0.01  # Rotate over time
            rotation_matrix = np.array(
                [
                    [np.cos(angle), -np.sin(angle)],
                    [np.sin(angle), np.cos(angle)],
                ]
            )
            center = np.array([400, 300])  # Center of the screen
            for vertex in base_vertices:
                rotated = rotation_matrix @ (vertex - center) + center
                rotated_vertices.append(rotated)
            gfx.draw(
                Rect(
                    Point(Vec2(*rotated_vertices[0]), semi_transparent_red),
                    Point(Vec2(*rotated_vertices[1]), semi_transparent_red),
                    Point(Vec2(*rotated_vertices[2]), semi_transparent_red),
                    Point(Vec2(*rotated_vertices[3]), semi_transparent_red),
                )
            )

            gfx.draw(
                Rect(
                    Point(Vec2(100, 100), semi_transparent_red),
                    Point(Vec2(100, 200), semi_transparent_red),
                    Point(Vec2(200, 200), semi_transparent_red),
                    Point(Vec2(200, 100), semi_transparent_red),
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
