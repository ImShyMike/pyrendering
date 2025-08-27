"""Texture example demonstrating how to use the Texture class."""

import traceback
from pathlib import Path

from pyrendering import Color, Graphics, Texture, Vec2


def main():
    """Texture rendering example"""

    # Initialize graphics
    gfx = Graphics(800, 600, "Texture demo", standalone=False, vsync=True)

    try:
        texture_path = Path(__file__).parent / "demo.png"

        if not texture_path.exists():
            print(f"Texture file not found at {texture_path}")
            print("Please ensure you have a demo.png file in the assets directory")
            return

        # Create a texture rectangle
        texture = Texture.from_path(50, 50, 400, 400, texture_path)

        warped_texture = Texture(
            Vec2(500, 400),
            Vec2(700, 400),
            Vec2(650, 500),
            Vec2(550, 500),
            texture_path,
        )

        # Run until window closes
        while not gfx.should_close():
            gfx.poll_events()
            gfx.begin_frame()

            gfx.clear(Color.from_hex("#1a1a2e"))  # Dark blue background

            # Draw the texture
            gfx.draw(texture)
            gfx.draw(warped_texture)

            gfx.display()

    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"Error: {e}")
        traceback.print_exc()
    finally:
        print("Cleaning up...")
        gfx.cleanup()
        print("Cleanup completed")


if __name__ == "__main__":
    main()
