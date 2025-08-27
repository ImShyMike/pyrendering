# pylint: disable=missing-function-docstring,missing-module-docstring
import os
from typing import Optional

from PIL import Image, ImageDraw, ImageFont


class FontManager:
    """Manages multiple font renderers"""

    def __init__(self):
        self.font_renderers = {}

    def get_font_renderer(
        self, ctx, font_size: int = 16, font_path: Optional[str] = None
    ):
        key = (font_size, font_path)
        if key not in self.font_renderers:
            self.font_renderers[key] = FontRenderer(ctx, font_size, font_path)
        return self.font_renderers[key]


class FontRenderer:
    """Handles font rendering and texture atlas generation"""

    def __init__(self, ctx, font_size: int = 16, font_path: Optional[str] = None):
        self.ctx = ctx
        self.font_size = font_size
        self.char_textures = {}
        self.char_metrics = {}

        # Load font (fallback to default if font_path is None)
        try:
            if font_path and os.path.exists(font_path):
                self.font = ImageFont.truetype(font_path, font_size)
                self.true_font = self.font.font
            else:
                # Try to load a system font
                self.font = ImageFont.load_default(font_size)
                self.true_font = self.font.font
        except Exception:  # pylint: disable=broad-except
            self.font = ImageFont.load_default(font_size)
            self.true_font = self.font.font

    def get_char_texture(self, char: str):
        """Get or create texture for a character"""
        if char in self.char_metrics:
            return self.char_textures[char], self.char_metrics[char]

        # Create a temporary image to measure the character
        temp_img = Image.new("RGBA", (200, 200), (0, 0, 0, 0))
        temp_draw = ImageDraw.Draw(temp_img)

        # Anchor (left, baseline)
        bbox = temp_draw.textbbox((0, 0), char, font=self.font, anchor="ls")

        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:  # Empty character
            width = self.font_size // 2
            advance = width
            char_height = self.font_size + 4
        else:
            width = bbox[2] - bbox[0]
            advance = width
            # Calculate height based on bbox for 'ls' anchor
            char_height = max(bbox[3] - bbox[1], self.font_size) + 4

        # Create character image with padding
        char_width = width + 4
        char_img = Image.new(
            "RGBA", (round(char_width), round(char_height)), (0, 0, 0, 0)
        )
        char_draw = ImageDraw.Draw(char_img)

        baseline_y = char_height - 6  # 6 pixels from bottom for padding

        char_draw.text(
            (2, baseline_y),
            char,
            font=self.font,
            fill=(255, 255, 255, 255),
            anchor="ls",
        )

        # Convert to texture
        texture = self.ctx.texture(char_img.size, 4)
        texture.write(char_img.tobytes())

        metrics = {
            "width": char_width,
            "height": char_height,
            "advance": advance + 1,  # Character spacing
            "offset_x": 0,
            "offset_y": 0,
        }

        self.char_textures[char] = texture
        self.char_metrics[char] = metrics

        return texture, metrics
