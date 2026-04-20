from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from reportlab.pdfgen import canvas

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ai_ocr_pipeline.models import TextBox
from ai_ocr_pipeline.pdf import extract_pdf_text_layers, pdf_to_images


class PdfTextLayerTests(unittest.TestCase):
    def test_textbox_serialization_rounds_floats(self) -> None:
        box = TextBox(
            text="sample",
            width=100,
            height=20,
            center_x=1231.8372567494712,
            center_y=764.2072041829427,
            confidence=0.98765,
        )

        data = box.to_dict()

        self.assertEqual(data["x"], 1181.8)
        self.assertEqual(data["y"], 754.2)
        self.assertEqual(data["confidence"], 0.988)

    def test_extract_pdf_text_layers_uses_embedded_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            pdf_path = tmp_path / "sample.pdf"

            c = canvas.Canvas(str(pdf_path), pagesize=(200, 200))
            c.drawString(40, 150, "Hello PDF")
            c.drawString(40, 120, "Second line")
            c.save()

            results = extract_pdf_text_layers(pdf_path, dpi=72, min_chars=3)

            self.assertEqual(len(results), 1)
            page_result = results[0]
            self.assertIsNotNone(page_result)
            assert page_result is not None
            self.assertEqual(page_result.page, 1)
            self.assertEqual(page_result.img_width, 200)
            self.assertEqual(page_result.img_height, 200)
            texts = [box.text for box in page_result.boxes]
            self.assertIn("Hello PDF", texts)
            self.assertIn("Second line", texts)
            hello_box = next(box for box in page_result.boxes if box.text == "Hello PDF")
            self.assertEqual(hello_box.type, "text_layer")
            self.assertEqual(hello_box.confidence, 1.0)
            self.assertGreater(hello_box.center_x, 40)
            self.assertLess(hello_box.center_y, 80)

    def test_extract_pdf_text_layers_preserves_spaces_between_latin_words(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            pdf_path = tmp_path / "sample.pdf"

            c = canvas.Canvas(str(pdf_path), pagesize=(200, 200))
            c.drawString(40, 150, "Hello")
            c.drawString(78, 150, "World")
            c.save()

            results = extract_pdf_text_layers(pdf_path, dpi=72, min_chars=3)

            self.assertEqual(len(results), 1)
            page_result = results[0]
            self.assertIsNotNone(page_result)
            assert page_result is not None
            texts = [box.text for box in page_result.boxes]
            self.assertIn("Hello World", texts)

    def test_pdf_to_images_can_render_selected_pages(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            pdf_path = tmp_path / "sample.pdf"

            c = canvas.Canvas(str(pdf_path), pagesize=(200, 200))
            c.drawString(40, 150, "Page 1")
            c.showPage()
            c.drawString(40, 150, "Page 2")
            c.save()

            output_dir = tmp_path / "images"
            output_dir.mkdir()
            image_paths = pdf_to_images(pdf_path, output_dir, dpi=72, page_numbers=[2])

            self.assertEqual(len(image_paths), 1)
            self.assertTrue(image_paths[0].name.endswith("page0002.png"))


if __name__ == "__main__":
    unittest.main()
