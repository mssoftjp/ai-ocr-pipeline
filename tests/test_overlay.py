from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ai_ocr_pipeline.models import PageResult, TextBox
from ai_ocr_pipeline.overlay import _confidence_color, generate_svg


class OverlayTests(unittest.TestCase):
    def test_confidence_color_uses_five_bands(self) -> None:
        self.assertEqual(_confidence_color(0.90), "#0B6E4F")
        self.assertEqual(_confidence_color(0.80), "#6BAF45")
        self.assertEqual(_confidence_color(0.65), "#E6B800")
        self.assertEqual(_confidence_color(0.50), "#FF8C00")
        self.assertEqual(_confidence_color(0.30), "#DC143C")

    def test_generate_svg_does_not_truncate_text_by_default(self) -> None:
        page = PageResult(
            source="sample.png",
            page=None,
            img_width=200,
            img_height=80,
            boxes=[
                TextBox(
                    text="This is a long OCR line that should remain fully visible in the overlay output.",
                    width=180,
                    height=20,
                    center_x=100.0,
                    center_y=40.0,
                    confidence=0.9,
                )
            ],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "sample.png"
            Image.new("RGB", (200, 80), (255, 255, 255)).save(image_path)
            svg = generate_svg(page, image_path)

        self.assertIn("This is a long OCR line that should remain fully visible in the overlay output.", svg)

    def test_generate_svg_can_still_limit_text_when_requested(self) -> None:
        page = PageResult(
            source="sample.png",
            page=None,
            img_width=200,
            img_height=80,
            boxes=[
                TextBox(
                    text="abcdefghijklmnopqrstuvwxyz",
                    width=180,
                    height=20,
                    center_x=100.0,
                    center_y=40.0,
                    confidence=0.9,
                )
            ],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "sample.png"
            Image.new("RGB", (200, 80), (255, 255, 255)).save(image_path)
            svg = generate_svg(page, image_path, max_text_len=10)

        self.assertIn("abcdefghij", svg)
        self.assertNotIn("abcdefghijkl", svg)
