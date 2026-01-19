import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, __file__ as PIL_FILE


def _default_font_path():
    try:
        return str(Path(PIL_FILE).parent / "fonts" / "DejaVuSans.ttf")
    except Exception:
        return None


class TextRenderer:
    def __init__(self, font_size=48, font_path=None):
        self.font_size = font_size
        self.font_path = font_path
        self.font, self.loaded_path = self._load_font(font_path, font_size)
        if self.loaded_path:
            print(f"[TextRenderer] Using font: {self.loaded_path} (size {font_size})")
        else:
            print("[TextRenderer] Using default PIL font (TTF load failed)")

    def _load_font(self, font_path, font_size):
        cwd = Path.cwd()
        candidates = [
            font_path
        ]
        for candidate in candidates:
            if not candidate:
                continue
            candidate_path = Path(candidate)
            if candidate_path.exists():
                try:
                    return ImageFont.truetype(str(candidate_path), font_size), str(candidate_path)
                except Exception:
                    continue

        # Fallback bitmap font (size is fixed and small)
        return ImageFont.load_default(), None


    def size(self, text):
        bbox = self.font.getbbox(text)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]

    def draw(self, frame, items):
        """
        items: list of dicts with keys text, x, y, color (BGR)
        """
        if not items:
            return frame
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        for item in items:
            color = item.get("color", (255, 255, 255))
            draw.text(
                (item["x"], item["y"]),
                str(item["text"]),
                font=self.font,
                fill=(color[2], color[1], color[0]),
            )
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
