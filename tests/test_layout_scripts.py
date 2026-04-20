from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_script_module(name: str):
    module_path = PROJECT_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, module)
    spec.loader.exec_module(module)
    return module


reconstruct_layout = _load_script_module("reconstruct_layout")
visualize_layout = _load_script_module("visualize_layout")


class LayoutScriptTests(unittest.TestCase):
    def test_reconstruct_layout_box_geometry_uses_ratio_coordinates(self) -> None:
        result = {
            "img_width": 200,
            "img_height": 100,
        }
        box = {
            "x": 0.1,
            "y": 0.2,
            "width": 0.25,
            "height": 0.3,
        }

        self.assertEqual(reconstruct_layout._box_geometry(result, box), (20.0, 20.0, 50.0, 30.0))

    def test_visualize_layout_box_geometry_prefers_pixel_fields(self) -> None:
        result = {
            "img_width": 200,
            "img_height": 100,
        }
        box = {
            "x": 0.1,
            "y": 0.2,
            "width": 0.25,
            "height": 0.3,
            "pixel_x": 22.0,
            "pixel_y": 24.0,
            "pixel_width": 48,
            "pixel_height": 28,
        }

        self.assertEqual(visualize_layout._box_geometry(result, box), (22.0, 24.0, 48.0, 28.0))

    def test_visualize_layout_svg_uses_id_and_ratio_geometry(self) -> None:
        result = {
            "page": 1,
            "img_width": 200,
            "img_height": 100,
            "boxes": [
                {
                    "id": 7,
                    "text": "Invoice",
                    "x": 0.1,
                    "y": 0.2,
                    "width": 0.25,
                    "height": 0.3,
                }
            ],
        }

        svg = visualize_layout._build_overlay_svg(
            page_png_name="page.png",
            result=result,
            show_image=False,
        )

        self.assertIn('rect x="20.0" y="20.0" width="50.0" height="30.0"', svg)
        self.assertIn(">7: Invoice</text>", svg)


if __name__ == "__main__":
    unittest.main()
