from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ai_ocr_pipeline.models import PageResult, TextBox
from ai_ocr_pipeline.template import (
    build_ocr_evidence,
    build_template_prompt_contexts,
    decide_target_box_actions,
    load_template,
    template_to_page_result,
)


class TemplateTests(unittest.TestCase):
    def _write_template(self, payload: dict) -> Path:
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        path = Path(tmpdir.name) / "template.json"
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return path

    def _valid_ratio_payload(self) -> dict:
        return {
            "template": {
                "name": "order-form",
                "version": 1,
                "coordinate_mode": "ratio",
            },
            "defaults": {"is_vertical": False},
            "boxes": [
                {
                    "id": 5,
                    "label": "金額",
                    "x": 0.4,
                    "y": 0.3,
                    "width": 0.2,
                    "height": 0.1,
                },
                {
                    "id": 1,
                    "label": "注文日",
                    "x": 0.1,
                    "y": 0.2,
                    "width": 0.3,
                    "height": 0.1,
                    "hint": "日付形式",
                },
            ],
        }

    def test_load_template_accepts_valid_ratio_template(self) -> None:
        template = load_template(self._write_template(self._valid_ratio_payload()))

        self.assertEqual(template.name, "order-form")
        self.assertEqual(template.coordinate_mode, "ratio")
        self.assertEqual(len(template.boxes), 2)
        self.assertFalse(template.boxes[0].is_vertical)
        self.assertIsNone(template.preprocess_deskew)
        self.assertIsNone(template.preprocess_remove_horizontal_lines)
        self.assertIsNone(template.preprocess_remove_vertical_lines)
        self.assertIsNone(template.preprocess_newline_handling)

    def test_load_template_accepts_optional_preprocess_defaults(self) -> None:
        payload = self._valid_ratio_payload()
        payload["preprocess"] = {
            "deskew": True,
            "remove_horizontal_lines": True,
            "remove_vertical_lines": False,
            "newline_handling": "join",
        }

        template = load_template(self._write_template(payload))

        self.assertTrue(template.preprocess_deskew)
        self.assertTrue(template.preprocess_remove_horizontal_lines)
        self.assertFalse(template.preprocess_remove_vertical_lines)
        self.assertEqual(template.preprocess_newline_handling, "join")

    def test_load_template_accepts_empty_box_label(self) -> None:
        payload = self._valid_ratio_payload()
        payload["boxes"][0]["label"] = ""

        template = load_template(self._write_template(payload))

        self.assertEqual(template.boxes[0].label, "")

    def test_load_template_rejects_missing_required_fields(self) -> None:
        path = self._write_template({"template": {"version": 1, "coordinate_mode": "ratio"}, "boxes": []})

        with self.assertRaisesRegex(ValueError, "template.name"):
            load_template(path)

    def test_load_template_rejects_invalid_coordinate_mode(self) -> None:
        payload = self._valid_ratio_payload()
        payload["template"]["coordinate_mode"] = "grid"

        with self.assertRaisesRegex(ValueError, "coordinate_mode"):
            load_template(self._write_template(payload))

    def test_load_template_rejects_invalid_version(self) -> None:
        payload = self._valid_ratio_payload()
        payload["template"]["version"] = 2

        with self.assertRaisesRegex(ValueError, "Expected 1"):
            load_template(self._write_template(payload))

    def test_load_template_rejects_invalid_preprocess_value(self) -> None:
        payload = self._valid_ratio_payload()
        payload["preprocess"] = {"deskew": "yes"}

        with self.assertRaisesRegex(ValueError, "preprocess.deskew"):
            load_template(self._write_template(payload))

    def test_load_template_rejects_invalid_newline_handling(self) -> None:
        payload = self._valid_ratio_payload()
        payload["preprocess"] = {"newline_handling": "trim"}

        with self.assertRaisesRegex(ValueError, "preprocess.newline_handling"):
            load_template(self._write_template(payload))

    def test_load_template_rejects_duplicate_ids(self) -> None:
        payload = self._valid_ratio_payload()
        payload["boxes"][1]["id"] = 5

        with self.assertRaisesRegex(ValueError, "Duplicate"):
            load_template(self._write_template(payload))

    def test_load_template_rejects_out_of_range_ratio(self) -> None:
        payload = self._valid_ratio_payload()
        payload["boxes"][0]["x"] = 0.95

        with self.assertRaisesRegex(ValueError, "exceeds ratio bounds"):
            load_template(self._write_template(payload))

    def test_load_template_rejects_pixel_mode_without_reference_size(self) -> None:
        payload = self._valid_ratio_payload()
        payload["template"]["coordinate_mode"] = "pixel"

        with self.assertRaisesRegex(ValueError, "reference_size"):
            load_template(self._write_template(payload))

    def test_template_to_page_result_converts_ratio_coordinates_and_sorts_by_id(self) -> None:
        template = load_template(self._write_template(self._valid_ratio_payload()))

        result = template_to_page_result(template, 200, 100, "sample.png", None)

        self.assertEqual([box.order for box in result.boxes], [1, 5])
        self.assertEqual([box.type for box in result.boxes], ["注文日", "金額"])
        self.assertEqual(result.boxes[0].width, 60)
        self.assertEqual(result.boxes[0].height, 10)
        self.assertEqual(result.boxes[0].center_x, 50.0)
        self.assertEqual(result.boxes[0].center_y, 25.0)
        self.assertEqual(result.boxes[0].text_source, "template")

    def test_template_to_page_result_scales_pixel_coordinates(self) -> None:
        payload = self._valid_ratio_payload()
        payload["template"]["coordinate_mode"] = "pixel"
        payload["template"]["reference_size"] = {"width": 1000, "height": 2000}
        payload["boxes"] = [
            {"id": 2, "label": "金額", "x": 100, "y": 400, "width": 200, "height": 200},
        ]
        template = load_template(self._write_template(payload))

        result = template_to_page_result(template, 500, 1000, "sample.png", None)

        self.assertEqual(result.boxes[0].width, 100)
        self.assertEqual(result.boxes[0].height, 100)
        self.assertEqual(result.boxes[0].center_x, 100.0)
        self.assertEqual(result.boxes[0].center_y, 250.0)

    def test_template_to_page_result_filters_box_ids(self) -> None:
        template = load_template(self._write_template(self._valid_ratio_payload()))

        result = template_to_page_result(template, 200, 100, "sample.png", None, box_ids=(5,))

        self.assertEqual(len(result.boxes), 1)
        self.assertEqual(result.boxes[0].order, 5)

    def test_template_to_page_result_rejects_unknown_box_ids(self) -> None:
        template = load_template(self._write_template(self._valid_ratio_payload()))

        with self.assertRaisesRegex(ValueError, "Unknown template box id"):
            template_to_page_result(template, 200, 100, "sample.png", None, box_ids=(99,))

    def test_build_template_prompt_contexts_includes_labels_for_all_selected_boxes(self) -> None:
        template = load_template(self._write_template(self._valid_ratio_payload()))

        contexts = build_template_prompt_contexts(template, None)

        self.assertEqual(contexts[0], {"label": "注文日", "hint": "日付形式"})
        self.assertEqual(contexts[1], {"label": "金額"})

    def test_build_template_prompt_contexts_omits_blank_label(self) -> None:
        payload = self._valid_ratio_payload()
        payload["boxes"][0]["label"] = ""
        template = load_template(self._write_template(payload))

        contexts = build_template_prompt_contexts(template, None)

        self.assertEqual(contexts[1], {})

    def test_build_ocr_evidence_matches_center_inside_boxes(self) -> None:
        template = load_template(self._write_template(self._valid_ratio_payload()))
        target = template_to_page_result(template, 200, 100, "sample.png", None)
        with_evidence = build_ocr_evidence(
            target,
            {
                0: PageResult(
                    source="sample.png",
                    page=None,
                    img_width=60,
                    img_height=10,
                    boxes=[
                        TextBox(
                            text="2024/04/18",
                            width=40,
                            height=10,
                            center_x=20.0,
                            center_y=5.0,
                            confidence=0.91,
                            order=2,
                            text_source="ocr",
                        ),
                    ],
                ),
                1: PageResult(
                    source="sample.png",
                    page=None,
                    img_width=30,
                    img_height=10,
                    boxes=[
                        TextBox(
                            text="123,456",
                            width=30,
                            height=10,
                            center_x=15.0,
                            center_y=5.0,
                            confidence=0.87,
                            order=1,
                            text_source="ocr",
                        ),
                    ],
                ),
            },
            low_ink_by_index={0: False, 1: False},
        )

        self.assertEqual(with_evidence.boxes[0].ocr_seed_text, "2024/04/18")
        self.assertEqual(with_evidence.boxes[0].ocr_match_count, 1)
        self.assertEqual(with_evidence.boxes[0].ocr_seed_confidence, 0.91)
        self.assertEqual(with_evidence.boxes[0].ocr_consensus_text, "2024/04/18")
        self.assertEqual(with_evidence.boxes[0].ocr_consensus_confidence, 0.91)
        self.assertFalse(with_evidence.boxes[0].low_ink)
        self.assertEqual(with_evidence.boxes[1].ocr_seed_text, "123,456")

    def test_build_ocr_evidence_joins_crop_boxes_in_reading_order(self) -> None:
        template = load_template(self._write_template(self._valid_ratio_payload()))
        target = template_to_page_result(template, 200, 100, "sample.png", None)
        with_evidence = build_ocr_evidence(
            target,
            {
                0: PageResult(
                    source="sample.png",
                    page=None,
                    img_width=80,
                    img_height=20,
                    boxes=[
                        TextBox(
                            text="2024/04/18",
                            width=40,
                            height=10,
                            center_x=20.0,
                            center_y=5.0,
                            confidence=0.82,
                            order=2,
                            text_source="ocr",
                        ),
                        TextBox(
                            text="123,456",
                            width=30,
                            height=10,
                            center_x=15.0,
                            center_y=15.0,
                            confidence=0.52,
                            order=3,
                            text_source="ocr",
                        ),
                    ],
                ),
            },
            low_ink_by_index={0: False, 1: True},
        )

        self.assertEqual(with_evidence.boxes[0].ocr_match_count, 2)
        self.assertEqual(with_evidence.boxes[1].ocr_match_count, 0)
        self.assertEqual(with_evidence.boxes[0].ocr_seed_text, "2024/04/18 123,456")
        self.assertIsNone(with_evidence.boxes[0].ocr_consensus_text)
        self.assertTrue(with_evidence.boxes[1].low_ink)

    def test_decide_target_box_actions_marks_blank_skip_when_no_matches(self) -> None:
        page = PageResult(
            source="sample.png",
            page=None,
            img_width=100,
            img_height=50,
            boxes=[
                TextBox(
                    text="",
                    width=20,
                    height=10,
                    center_x=10.0,
                    center_y=10.0,
                    confidence=0.0,
                    ocr_match_count=0,
                    low_ink=True,
                ),
                TextBox(
                    text="123,456",
                    width=20,
                    height=10,
                    center_x=40.0,
                    center_y=10.0,
                    confidence=0.72,
                    ocr_seed_text="123,456",
                    ocr_seed_confidence=0.72,
                    ocr_match_count=1,
                ),
            ],
        )

        decided, refine_indices = decide_target_box_actions(page)

        self.assertEqual(decided.boxes[0].decision, "blank_skip")
        self.assertEqual(decided.boxes[0].text_source, "blank_skip")
        self.assertEqual(decided.boxes[0].text, "")
        self.assertEqual(decided.boxes[1].decision, "ai")
        self.assertEqual(refine_indices, (1,))

    def test_decide_target_box_actions_keeps_ai_when_no_matches_but_not_low_ink(self) -> None:
        page = PageResult(
            source="sample.png",
            page=None,
            img_width=100,
            img_height=50,
            boxes=[
                TextBox(
                    text="",
                    width=20,
                    height=10,
                    center_x=10.0,
                    center_y=10.0,
                    confidence=0.0,
                    ocr_match_count=0,
                    low_ink=False,
                ),
            ],
        )

        decided, refine_indices = decide_target_box_actions(page)

        self.assertEqual(decided.boxes[0].decision, "ai")
        self.assertEqual(refine_indices, (0,))

    def test_decide_target_box_actions_blank_skips_low_ink_punctuation_noise(self) -> None:
        page = PageResult(
            source="sample.png",
            page=None,
            img_width=100,
            img_height=50,
            boxes=[
                TextBox(
                    text="- - - -",
                    width=20,
                    height=10,
                    center_x=10.0,
                    center_y=10.0,
                    confidence=0.0,
                    ocr_seed_text="- - - -",
                    ocr_match_count=1,
                    low_ink=True,
                ),
            ],
        )

        decided, refine_indices = decide_target_box_actions(page)

        self.assertEqual(decided.boxes[0].decision, "blank_skip")
        self.assertEqual(decided.boxes[0].text_source, "blank_skip")
        self.assertEqual(refine_indices, ())

    def test_decide_target_box_actions_uses_ocr_only_when_threshold_is_explicit(self) -> None:
        page = PageResult(
            source="sample.png",
            page=None,
            img_width=100,
            img_height=50,
            boxes=[
                TextBox(
                    text="123,456",
                    width=20,
                    height=10,
                    center_x=40.0,
                    center_y=10.0,
                    confidence=0.93,
                    ocr_seed_text="123,456",
                    ocr_seed_confidence=0.93,
                    ocr_match_count=1,
                    ocr_consensus_text="123,456",
                    ocr_consensus_confidence=0.93,
                ),
            ],
        )

        decided, refine_indices = decide_target_box_actions(page, confidence_threshold=0.9)

        self.assertEqual(decided.boxes[0].decision, "ocr")
        self.assertEqual(decided.boxes[0].text_source, "ocr")
        self.assertEqual(decided.boxes[0].text, "123,456")
        self.assertEqual(refine_indices, ())

    def test_decide_target_box_actions_does_not_use_threshold_without_consensus(self) -> None:
        page = PageResult(
            source="sample.png",
            page=None,
            img_width=100,
            img_height=50,
            boxes=[
                TextBox(
                    text="123,456 123,45G",
                    width=20,
                    height=10,
                    center_x=40.0,
                    center_y=10.0,
                    confidence=0.93,
                    ocr_seed_text="123,456 123,45G",
                    ocr_seed_confidence=0.93,
                    ocr_match_count=2,
                    ocr_consensus_text=None,
                    ocr_consensus_confidence=None,
                ),
            ],
        )

        decided, refine_indices = decide_target_box_actions(page, confidence_threshold=0.9)

        self.assertEqual(decided.boxes[0].decision, "ai")
        self.assertEqual(refine_indices, (0,))

    def test_decide_target_box_actions_keeps_symbol_only_seed_when_not_obvious_noise(self) -> None:
        page = PageResult(
            source="sample.png",
            page=None,
            img_width=100,
            img_height=50,
            boxes=[
                TextBox(
                    text="/",
                    width=20,
                    height=10,
                    center_x=10.0,
                    center_y=10.0,
                    confidence=0.0,
                    ocr_seed_text="/",
                    ocr_match_count=1,
                    low_ink=True,
                ),
            ],
        )

        decided, refine_indices = decide_target_box_actions(page)

        self.assertEqual(decided.boxes[0].decision, "ai")
        self.assertEqual(refine_indices, (0,))


if __name__ == "__main__":
    unittest.main()
