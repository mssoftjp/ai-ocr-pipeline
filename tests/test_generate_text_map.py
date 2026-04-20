from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "scripts" / "generate_text_map.py"
SPEC = importlib.util.spec_from_file_location("generate_text_map", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
generate_text_map = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(generate_text_map)


class GenerateTextMapTests(unittest.TestCase):
    def _write_sample_inputs(self, tmpdir: str) -> tuple[Path, Path]:
        ocr_json_path = Path(tmpdir) / "sample.json"
        image_path = Path(tmpdir) / "sample.png"
        payload = {
            "results": [
                {
                    "img_width": 120,
                    "img_height": 80,
                    "boxes": [
                        {
                            "text": "horizontal",
                            "id": 1,
                            "x": 10 / 120,
                            "y": 9 / 80,
                            "width": 50 / 120,
                            "height": 18 / 80,
                            "confidence": 0.7,
                            "is_vertical": False,
                        },
                        {
                            "text": "縦書き",
                            "id": 2,
                            "x": 81 / 120,
                            "y": 15 / 80,
                            "width": 18 / 120,
                            "height": 50 / 80,
                            "confidence": 0.7,
                            "is_vertical": True,
                        },
                    ],
                }
            ]
        }
        ocr_json_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        Image.new("RGB", (120, 80), (255, 255, 255)).save(image_path)
        return ocr_json_path, image_path

    def test_generate_svg_uses_vertical_writing_mode_for_vertical_boxes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ocr_json_path, image_path = self._write_sample_inputs(tmpdir)
            svg = generate_text_map.generate_svg(ocr_json_path, image_path)

        self.assertIn('writing-mode="vertical-rl"', svg)
        self.assertIn('text-orientation="upright"', svg)
        self.assertIn(">縦書き</text>", svg)
        self.assertIn(">[0] horizontal</text>", svg)

    def test_page_relative_font_size_scales_up_for_large_pages(self) -> None:
        text_min, index_min = generate_text_map._page_relative_font_sizes(7000, 5000)
        self.assertGreaterEqual(text_min, 30)
        self.assertGreaterEqual(index_min, 16)

    def test_render_orientation_prefers_horizontal_for_wide_boxes(self) -> None:
        self.assertFalse(generate_text_map._should_render_vertical({"width": 120, "height": 40, "is_vertical": True}))

    def test_render_orientation_prefers_vertical_for_tall_boxes(self) -> None:
        self.assertTrue(generate_text_map._should_render_vertical({"width": 30, "height": 80, "is_vertical": False}))

    def test_render_orientation_uses_flag_for_ambiguous_boxes(self) -> None:
        self.assertTrue(generate_text_map._should_render_vertical({"width": 40, "height": 50, "is_vertical": True}))
        self.assertFalse(generate_text_map._should_render_vertical({"width": 40, "height": 50, "is_vertical": False}))

    def test_svg_to_png_falls_back_to_svg_path_when_cairosvg_missing(self) -> None:
        real_import = __import__

        def fake_import(name: str, *args, **kwargs):
            if name == "cairosvg":
                raise ImportError("missing cairosvg")
            return real_import(name, *args, **kwargs)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "overlay.png"
            with patch("builtins.__import__", side_effect=fake_import):
                actual_format, actual_path = generate_text_map.svg_to_png("<svg/>", output_path)

            self.assertEqual(actual_format, "svg")
            self.assertEqual(actual_path, output_path.with_suffix(".svg"))
            self.assertTrue(actual_path.exists())

    def test_main_reports_svg_fallback_instead_of_png_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ocr_json_path, image_path = self._write_sample_inputs(tmpdir)
            output_path = Path(tmpdir) / "overlay.png"
            fallback_path = output_path.with_suffix(".svg")
            stdout = StringIO()

            with (
                patch.object(generate_text_map, "write_overlay_artifact", return_value=("svg", fallback_path)),
                patch.object(
                    sys,
                    "argv",
                    [
                        "generate_text_map.py",
                        str(ocr_json_path),
                        str(image_path),
                        "--format",
                        "png",
                        "--output",
                        str(output_path),
                    ],
                ),
                redirect_stdout(stdout),
            ):
                generate_text_map.main()

        self.assertIn("SVG saved instead", stdout.getvalue())
        self.assertNotIn("PNG saved:", stdout.getvalue())
