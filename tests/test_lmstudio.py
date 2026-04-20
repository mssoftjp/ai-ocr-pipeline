from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib import error

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ai_ocr_pipeline.llm.lmstudio import (
    LMStudioConfig,
    _box_bounds,
    _build_refine_prompt,
    _build_system_prompt,
    _crop_box_data_url,
    _detect_field_type,
    _find_neighbor_labels,
    _normalize_box_text,
    _padding_amounts,
    _request_box_response,
    _request_json,
    _validate_box_text,
    normalize_base_url,
    refine_page_result,
    refine_page_result_with_stats,
    resolve_model,
)
from ai_ocr_pipeline.models import PageResult, TextBox, effective_is_vertical


class _FakeHTTPResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> _FakeHTTPResponse:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class LMStudioTests(unittest.TestCase):
    def test_normalize_base_url_accepts_root_and_api_v1(self) -> None:
        self.assertEqual(normalize_base_url("http://127.0.0.1:1234"), "http://127.0.0.1:1234/v1")
        self.assertEqual(normalize_base_url("http://127.0.0.1:1234/v1"), "http://127.0.0.1:1234/v1")
        self.assertEqual(normalize_base_url("http://127.0.0.1:1234/api/v1"), "http://127.0.0.1:1234/v1")

    def test_resolve_model_uses_first_model_id(self) -> None:
        with patch(
            "ai_ocr_pipeline.llm.lmstudio.request.urlopen",
            return_value=_FakeHTTPResponse({"data": [{"id": "gemma-4"}]}),
        ):
            model = resolve_model(LMStudioConfig(base_url="http://127.0.0.1:1234"))

        self.assertEqual(model, "gemma-4")

    def test_refine_page_result_updates_text_only(self) -> None:
        page = PageResult(
            source="sample.png",
            page=None,
            img_width=100,
            img_height=60,
            boxes=[
                TextBox(
                    text="Inv0ice",
                    width=20,
                    height=10,
                    center_x=10.0,
                    center_y=10.0,
                    confidence=0.5,
                    order=1,
                )
            ],
        )
        config = LMStudioConfig(
            base_url="http://127.0.0.1:1234",
            model="gemma-4",
            timeout=5,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "sample.png"
            Image.new("RGB", (100, 60), (255, 255, 255)).save(image_path)
            with patch(
                "ai_ocr_pipeline.llm.lmstudio.request.urlopen",
                return_value=_FakeHTTPResponse({"choices": [{"message": {"content": "Invoice"}}]}),
            ):
                refined = refine_page_result(image_path, page, config)

        self.assertEqual(refined.boxes[0].text, "Invoice")
        self.assertEqual(refined.boxes[0].width, 20)
        self.assertEqual(refined.boxes[0].center_x, 10.0)

    def test_refine_page_result_keeps_original_text_for_blank_box_response(self) -> None:
        page = PageResult(
            source="sample.png",
            page=None,
            img_width=100,
            img_height=60,
            boxes=[
                TextBox(
                    text="Inv0ice",
                    width=20,
                    height=10,
                    center_x=10.0,
                    center_y=10.0,
                    confidence=0.5,
                    order=1,
                ),
                TextBox(
                    text="T0tal",
                    width=20,
                    height=10,
                    center_x=30.0,
                    center_y=10.0,
                    confidence=0.5,
                    order=2,
                ),
            ],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "sample.png"
            Image.new("RGB", (100, 60), (255, 255, 255)).save(image_path)
            responses = iter(
                [
                    _FakeHTTPResponse({"choices": [{"message": {"content": "Invoice"}}]}),
                    _FakeHTTPResponse({"choices": [{"message": {"content": ""}}]}),
                ]
            )
            with patch(
                "ai_ocr_pipeline.llm.lmstudio.request.urlopen",
                side_effect=lambda req, timeout=None: next(responses),
            ):
                refined = refine_page_result(
                    image_path,
                    page,
                    LMStudioConfig(base_url="http://127.0.0.1:1234", model="gemma-4", max_workers=1),
                )

        self.assertEqual(refined.boxes[0].text, "Invoice")
        self.assertEqual(refined.boxes[1].text, "T0tal")

    def test_refine_page_result_skips_all_text_layer_boxes(self) -> None:
        page = PageResult(
            source="sample.pdf",
            page=1,
            img_width=100,
            img_height=60,
            boxes=[
                TextBox(
                    text="Invoice",
                    width=20,
                    height=10,
                    center_x=10.0,
                    center_y=10.0,
                    confidence=1.0,
                    type="text_layer",
                )
            ],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "sample.png"
            Image.new("RGB", (100, 60), (255, 255, 255)).save(image_path)
            with patch("ai_ocr_pipeline.llm.lmstudio.request.urlopen") as urlopen:
                refined = refine_page_result(
                    image_path, page, LMStudioConfig(base_url="http://127.0.0.1:1234", model="gemma-4")
                )

        self.assertEqual(refined, page)
        urlopen.assert_not_called()

    def test_refine_page_result_preserves_text_layer_boxes_in_mixed_page(self) -> None:
        captured_request: list[object] = []
        page = PageResult(
            source="sample.png",
            page=None,
            img_width=500,
            img_height=500,
            boxes=[
                TextBox(
                    text="Trusted",
                    width=20,
                    height=10,
                    center_x=10.0,
                    center_y=400.0,
                    confidence=1.0,
                    type="text_layer",
                ),
                TextBox(
                    text="Inv0ice",
                    width=20,
                    height=10,
                    center_x=30.0,
                    center_y=10.0,
                    confidence=0.5,
                ),
            ],
        )

        def _fake_urlopen(req, timeout=None):
            captured_request.append(req)
            return _FakeHTTPResponse({"choices": [{"message": {"content": "Invoice"}}]})

        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "sample.png"
            Image.new("RGB", (500, 500), (255, 255, 255)).save(image_path)
            with patch("ai_ocr_pipeline.llm.lmstudio.request.urlopen", side_effect=_fake_urlopen):
                refined = refine_page_result(
                    image_path, page, LMStudioConfig(base_url="http://127.0.0.1:1234", model="gemma-4")
                )

        self.assertEqual(refined.boxes[0].text, "Trusted")
        self.assertEqual(refined.boxes[1].text, "Invoice")
        request_body = json.loads(captured_request[0].data.decode("utf-8"))
        prompt = request_body["messages"][1]["content"][0]["text"]
        self.assertIn("OCR hint: Inv0ice", prompt)
        self.assertNotIn("Trusted", prompt)
        self.assertEqual(request_body["max_tokens"], 4096)

    def test_refine_page_result_raises_on_malformed_model_response(self) -> None:
        page = PageResult(
            source="sample.png",
            page=None,
            img_width=100,
            img_height=60,
            boxes=[
                TextBox(
                    text="Inv0ice",
                    width=20,
                    height=10,
                    center_x=10.0,
                    center_y=10.0,
                    confidence=0.5,
                )
            ],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "sample.png"
            Image.new("RGB", (100, 60), (255, 255, 255)).save(image_path)
            with patch(
                "ai_ocr_pipeline.llm.lmstudio.request.urlopen",
                return_value=_FakeHTTPResponse({"choices": [{"message": {"content": ""}}]}),
            ):
                refined = refine_page_result(
                    image_path, page, LMStudioConfig(base_url="http://127.0.0.1:1234", model="gemma-4")
                )

        self.assertEqual(refined.boxes[0].text, "Inv0ice")

    def test_request_json_wraps_url_errors(self) -> None:
        with (
            patch("ai_ocr_pipeline.llm.lmstudio.request.urlopen", side_effect=error.URLError("offline")),
            self.assertRaises(RuntimeError),
        ):
            _request_json(
                "http://127.0.0.1:1234/v1/models",
                timeout=5,
                api_key=None,
            )

    def test_refine_page_result_continues_after_box_failure(self) -> None:
        captured_requests: list[object] = []
        page = PageResult(
            source="sample.png",
            page=None,
            img_width=100,
            img_height=60,
            boxes=[
                TextBox(
                    text="Inv0ice",
                    width=20,
                    height=10,
                    center_x=10.0,
                    center_y=10.0,
                    confidence=0.5,
                ),
                TextBox(
                    text="T0tal",
                    width=20,
                    height=10,
                    center_x=30.0,
                    center_y=10.0,
                    confidence=0.5,
                ),
                TextBox(
                    text="Am0unt",
                    width=20,
                    height=10,
                    center_x=50.0,
                    center_y=10.0,
                    confidence=0.5,
                ),
            ],
        )

        def _fake_urlopen(req, timeout=None):
            captured_requests.append(req)
            call_index = len(captured_requests)
            if call_index == 2:
                raise error.URLError("offline")
            if call_index == 1:
                return _FakeHTTPResponse({"choices": [{"message": {"content": "Invoice"}}]})
            return _FakeHTTPResponse({"choices": [{"message": {"content": "Amount"}}]})

        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "sample.png"
            Image.new("RGB", (100, 60), (255, 255, 255)).save(image_path)
            with (
                self.assertLogs("ai_ocr_pipeline.llm.lmstudio", level="DEBUG") as logs,
                patch("ai_ocr_pipeline.llm.lmstudio.request.urlopen", side_effect=_fake_urlopen),
            ):
                refined = refine_page_result(
                    image_path,
                    page,
                    LMStudioConfig(
                        base_url="http://127.0.0.1:1234",
                        model="gemma-4",
                        max_workers=1,  # Force sequential to make call order deterministic
                    ),
                )

        self.assertEqual(len(captured_requests), 3)
        self.assertEqual(refined.boxes[0].text, "Invoice")
        self.assertEqual(refined.boxes[1].text, "T0tal")
        self.assertEqual(refined.boxes[2].text, "Amount")
        self.assertTrue(any("LM Studio refinement skipped for box 1" in line for line in logs.output))

    def test_refine_page_result_rejects_overlong_multiline_box_output(self) -> None:
        page = PageResult(
            source="sample.png",
            page=None,
            img_width=100,
            img_height=60,
            boxes=[
                TextBox(
                    text="placeholder",
                    width=20,
                    height=10,
                    center_x=10.0,
                    center_y=10.0,
                    confidence=0.5,
                )
            ],
        )

        response_text = "header field label\n123456 789012\ndetail table summary"

        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "sample.png"
            Image.new("RGB", (100, 60), (255, 255, 255)).save(image_path)
            with (
                self.assertLogs("ai_ocr_pipeline.llm.lmstudio", level="DEBUG") as logs,
                patch(
                    "ai_ocr_pipeline.llm.lmstudio.request.urlopen",
                    return_value=_FakeHTTPResponse({"choices": [{"message": {"content": response_text}}]}),
                ),
            ):
                refined = refine_page_result(
                    image_path,
                    page,
                    LMStudioConfig(base_url="http://127.0.0.1:1234", model="gemma-4"),
                )

        self.assertEqual(refined.boxes[0].text, "placeholder")
        self.assertTrue(any("too many lines" in line for line in logs.output))

    def test_crop_box_data_url_respects_padding(self) -> None:
        box = TextBox(
            text="Test",
            width=20,
            height=10,
            center_x=50.0,
            center_y=30.0,
            confidence=0.5,
        )
        bounds = _box_bounds(box, (100, 80), padding_ratio=0.15)
        self.assertEqual(bounds, (36, 21, 64, 39))

        image = Image.new("RGB", (100, 80), (255, 255, 255))
        data_url = _crop_box_data_url(
            image,
            box,
            max_image_side=2048,
            padding_ratio=0.15,
        )
        self.assertTrue(data_url.startswith("data:image/png;base64,"))

    def test_box_bounds_can_use_asymmetric_padding(self) -> None:
        box = TextBox(
            text="単価",
            width=80,
            height=12,
            center_x=50.0,
            center_y=30.0,
            confidence=0.5,
        )
        symmetric = _box_bounds(box, (120, 80), padding_ratio=0.15)
        asymmetric = _box_bounds(box, (120, 80), padding_ratio=0.15, padding_ratio_y=0.45)
        self.assertEqual(symmetric[0], asymmetric[0])
        self.assertLess(asymmetric[1], symmetric[1])
        self.assertEqual(symmetric[2], asymmetric[2])
        self.assertGreater(asymmetric[3], symmetric[3])

    # --- hint_mode tests ---

    def test_hint_mode_none_omits_ocr_hint(self) -> None:
        box = TextBox(
            text="Inv0ice",
            width=20,
            height=10,
            center_x=10.0,
            center_y=10.0,
            confidence=0.5,
        )
        prompt = _build_refine_prompt(box, hint_mode="none")
        self.assertNotIn("OCR hint:", prompt)
        self.assertNotIn("Inv0ice", prompt)
        self.assertIn("Orientation:", prompt)
        self.assertIn("do not truncate", prompt)

    def test_hint_mode_weak_uses_softer_language(self) -> None:
        box = TextBox(
            text="Inv0ice",
            width=20,
            height=10,
            center_x=10.0,
            center_y=10.0,
            confidence=0.5,
        )
        prompt = _build_refine_prompt(box, hint_mode="weak")
        self.assertIn("Previous OCR (may have errors): Inv0ice", prompt)
        self.assertIn("Prioritize what you see", prompt)
        self.assertNotIn("OCR hint:", prompt)

    def test_hint_mode_full_preserves_original_format(self) -> None:
        box = TextBox(
            text="Inv0ice",
            width=20,
            height=10,
            center_x=10.0,
            center_y=10.0,
            confidence=0.5,
        )
        prompt = _build_refine_prompt(box, hint_mode="full")
        self.assertIn("OCR hint: Inv0ice", prompt)
        self.assertIn("return the OCR hint unchanged", prompt)
        self.assertIn("do not truncate", prompt)

    def test_template_prompt_uses_none_style_and_includes_label_and_hint(self) -> None:
        box = TextBox(
            text="",
            width=20,
            height=10,
            center_x=10.0,
            center_y=10.0,
            confidence=0.0,
            type="注文日",
            text_source="template",
        )
        prompt = _build_refine_prompt(
            box,
            hint_mode="full",
            template_context={"label": "注文日", "hint": "日付形式"},
        )
        self.assertIn("Return only the exact text you see", prompt)
        self.assertIn("Field: 注文日", prompt)
        self.assertIn("Expected content: 日付形式", prompt)
        self.assertNotIn("OCR hint:", prompt)

    def test_template_prompt_can_include_label_without_hint(self) -> None:
        box = TextBox(
            text="",
            width=20,
            height=10,
            center_x=10.0,
            center_y=10.0,
            confidence=0.0,
            type="金額",
            text_source="template",
        )
        prompt = _build_refine_prompt(
            box,
            hint_mode="full",
            template_context={"label": "金額"},
        )
        self.assertIn("Field: 金額", prompt)
        self.assertNotIn("Expected content:", prompt)

    def test_template_prompt_omits_blank_label_line(self) -> None:
        box = TextBox(
            text="",
            width=20,
            height=10,
            center_x=10.0,
            center_y=10.0,
            confidence=0.0,
            type="金額",
            text_source="template",
        )
        prompt = _build_refine_prompt(
            box,
            hint_mode="full",
            template_context={},
        )
        self.assertNotIn("Field:", prompt)

    def test_system_prompt_varies_by_hint_mode(self) -> None:
        full = _build_system_prompt("full")
        weak = _build_system_prompt("weak")
        none = _build_system_prompt("none")
        # System carries only format constraints, not task instructions
        for prompt in (full, weak, none):
            self.assertIn("No explanations", prompt)
        # hint_mode differences
        self.assertIn("correct", full)
        self.assertNotIn("hint", none.lower())

    def test_request_box_response_uses_none_system_prompt_for_template_context(self) -> None:
        captured_requests: list[object] = []

        def _fake_urlopen(req, timeout=None):
            captured_requests.append(req)
            return _FakeHTTPResponse({"choices": [{"message": {"content": "2024/04/18"}}]})

        with patch("ai_ocr_pipeline.llm.lmstudio.request.urlopen", side_effect=_fake_urlopen):
            _request_box_response(
                TextBox(
                    text="",
                    width=20,
                    height=10,
                    center_x=10.0,
                    center_y=10.0,
                    confidence=0.0,
                    type="注文日",
                    text_source="template",
                ),
                "data:image/png;base64,abc",
                LMStudioConfig(base_url="http://127.0.0.1:1234", model="gemma-4"),
                template_context={"label": "注文日", "hint": "日付形式"},
            )

        request_body = json.loads(captured_requests[0].data.decode("utf-8"))
        self.assertIn("Return only the exact text", request_body["messages"][0]["content"])
        self.assertIn("Field: 注文日", request_body["messages"][1]["content"][0]["text"])

    # --- field type detection tests ---

    def test_detect_field_type_numeric(self) -> None:
        self.assertEqual(_detect_field_type("3 6 6 6 6 0"), "numeric")
        self.assertEqual(_detect_field_type("25478 881 00"), "numeric")
        self.assertEqual(_detect_field_type("360460"), "numeric")

    def test_detect_field_type_date(self) -> None:
        self.assertEqual(_detect_field_type("平成1101年1月2日"), "date")
        self.assertEqual(_detect_field_type("平成 10 年 1 月 2 日"), "date")
        self.assertEqual(_detect_field_type("昭和45年1月2日"), "date")

    def test_detect_field_type_text(self) -> None:
        self.assertEqual(_detect_field_type("Invoice"), "text")
        self.assertEqual(_detect_field_type("数量"), "text")
        self.assertEqual(_detect_field_type(""), "text")

    def test_prompt_includes_numeric_hint_for_numeric_field(self) -> None:
        box = TextBox(
            text="360460",
            width=20,
            height=10,
            center_x=10.0,
            center_y=10.0,
            confidence=0.5,
        )
        for mode in ("full", "weak", "none"):
            prompt = _build_refine_prompt(box, hint_mode=mode)
            self.assertIn("numbers or codes", prompt, f"Failed for hint_mode={mode}")

    def test_validate_box_text_relaxes_length_for_empty_original_text(self) -> None:
        box = TextBox(
            text="",
            width=160,
            height=80,
            center_x=10.0,
            center_y=10.0,
            confidence=0.0,
            text_source="template",
        )
        text = "A" * 80

        validated = _validate_box_text(text, box)

        self.assertEqual(validated, text)

    def test_validate_box_text_uses_area_limit_for_template_even_with_long_seed_text(self) -> None:
        box = TextBox(
            text="X" * 200,
            width=60,
            height=20,
            center_x=10.0,
            center_y=10.0,
            confidence=0.0,
            text_source="template",
        )

        with self.assertRaises(RuntimeError):
            _validate_box_text("Y" * 80, box, template_context={"label": "商品名"})

    def test_effective_is_vertical_rejects_wide_horizontal_cjk_boxes_even_when_raw_flag_is_true(self) -> None:
        box = TextBox(
            text="商品名(大)",
            width=383,
            height=47,
            center_x=10.0,
            center_y=10.0,
            confidence=0.3,
            is_vertical=True,
        )
        self.assertFalse(effective_is_vertical(box))

    def test_effective_is_vertical_rejects_short_ascii_code_boxes(self) -> None:
        box = TextBox(
            text="AB",
            width=79,
            height=61,
            center_x=10.0,
            center_y=10.0,
            confidence=0.4,
            is_vertical=True,
        )
        self.assertFalse(effective_is_vertical(box))

    def test_effective_is_vertical_accepts_tall_cjk_boxes(self) -> None:
        box = TextBox(
            text="縦書き",
            width=30,
            height=90,
            center_x=10.0,
            center_y=10.0,
            confidence=0.9,
            is_vertical=True,
        )
        self.assertTrue(effective_is_vertical(box))

    def test_build_refine_prompt_uses_effective_orientation_not_raw_flag(self) -> None:
        box = TextBox(
            text="商品名(大)",
            width=383,
            height=47,
            center_x=10.0,
            center_y=10.0,
            confidence=0.3,
            is_vertical=True,
        )
        prompt = _build_refine_prompt(box, hint_mode="full")
        self.assertIn("Orientation: horizontal", prompt)

    def test_prompt_includes_date_hint_for_date_field(self) -> None:
        box = TextBox(
            text="平成10年1月2日",
            width=20,
            height=10,
            center_x=10.0,
            center_y=10.0,
            confidence=0.5,
        )
        for mode in ("full", "weak", "none"):
            prompt = _build_refine_prompt(box, hint_mode=mode)
            self.assertIn("date", prompt, f"Failed for hint_mode={mode}")

    # --- box_indices filtering tests ---

    def test_box_indices_limits_refinement(self) -> None:
        captured_requests: list[object] = []
        page = PageResult(
            source="sample.png",
            page=None,
            img_width=100,
            img_height=60,
            boxes=[
                TextBox(text="Inv0ice", width=20, height=10, center_x=10.0, center_y=10.0, confidence=0.5),
                TextBox(text="T0tal", width=20, height=10, center_x=30.0, center_y=10.0, confidence=0.5),
                TextBox(text="Am0unt", width=20, height=10, center_x=50.0, center_y=10.0, confidence=0.5),
            ],
        )

        def _fake_urlopen(req, timeout=None):
            captured_requests.append(req)
            return _FakeHTTPResponse({"choices": [{"message": {"content": "Corrected"}}]})

        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "sample.png"
            Image.new("RGB", (100, 60), (255, 255, 255)).save(image_path)
            with patch("ai_ocr_pipeline.llm.lmstudio.request.urlopen", side_effect=_fake_urlopen):
                refined = refine_page_result(
                    image_path,
                    page,
                    LMStudioConfig(base_url="http://127.0.0.1:1234", model="gemma-4", box_indices=(0, 2)),
                )

        self.assertEqual(len(captured_requests), 2)
        self.assertEqual(refined.boxes[0].text, "Corrected")
        self.assertEqual(refined.boxes[1].text, "T0tal")  # skipped
        self.assertEqual(refined.boxes[2].text, "Corrected")

    def test_selected_box_indices_limit_template_refinement(self) -> None:
        captured_requests: list[object] = []
        page = PageResult(
            source="sample.png",
            page=None,
            img_width=100,
            img_height=60,
            boxes=[
                TextBox(
                    text="seed0",
                    width=20,
                    height=10,
                    center_x=10.0,
                    center_y=10.0,
                    confidence=0.2,
                    text_source="template",
                ),
                TextBox(
                    text="seed1",
                    width=20,
                    height=10,
                    center_x=30.0,
                    center_y=10.0,
                    confidence=0.2,
                    text_source="template",
                ),
                TextBox(
                    text="seed2",
                    width=20,
                    height=10,
                    center_x=50.0,
                    center_y=10.0,
                    confidence=0.2,
                    text_source="template",
                ),
            ],
        )

        def _fake_urlopen(req, timeout=None):
            captured_requests.append(req)
            return _FakeHTTPResponse({"choices": [{"message": {"content": "Corrected"}}]})

        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "sample.png"
            Image.new("RGB", (100, 60), (255, 255, 255)).save(image_path)
            with patch("ai_ocr_pipeline.llm.lmstudio.request.urlopen", side_effect=_fake_urlopen):
                refined, _ = refine_page_result_with_stats(
                    image_path,
                    page,
                    LMStudioConfig(base_url="http://127.0.0.1:1234", model="gemma-4"),
                    template_contexts={0: {}, 1: {}, 2: {}},
                    selected_box_indices=(0, 2),
                )

        self.assertEqual(len(captured_requests), 2)
        self.assertEqual(refined.boxes[0].text, "Corrected")
        self.assertEqual(refined.boxes[1].text, "seed1")
        self.assertEqual(refined.boxes[2].text, "Corrected")

    # --- save_crops_dir tests ---

    def test_save_crops_writes_images(self) -> None:
        page = PageResult(
            source="sample.png",
            page=None,
            img_width=100,
            img_height=60,
            boxes=[
                TextBox(text="Hello", width=20, height=10, center_x=10.0, center_y=10.0, confidence=0.5),
                TextBox(text="World", width=20, height=10, center_x=30.0, center_y=10.0, confidence=0.5),
            ],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "sample.png"
            Image.new("RGB", (100, 60), (255, 255, 255)).save(image_path)
            crops_dir = Path(tmpdir) / "crops"

            responses = iter(
                [
                    _FakeHTTPResponse({"choices": [{"message": {"content": "Hello"}}]}),
                    _FakeHTTPResponse({"choices": [{"message": {"content": "World"}}]}),
                ]
            )
            with patch(
                "ai_ocr_pipeline.llm.lmstudio.request.urlopen",
                side_effect=lambda req, timeout=None: next(responses),
            ):
                refine_page_result(
                    image_path,
                    page,
                    LMStudioConfig(
                        base_url="http://127.0.0.1:1234",
                        model="gemma-4",
                        save_crops_dir=str(crops_dir),
                        max_workers=1,
                    ),
                )

            self.assertTrue(crops_dir.exists())
            saved = sorted(crops_dir.iterdir())
            self.assertEqual(len(saved), 2)
            self.assertEqual(saved[0].name, "box_0000.png")
            self.assertEqual(saved[1].name, "box_0001.png")
            # Verify they are valid PNG images
            with Image.open(saved[0]) as img:
                self.assertGreater(img.size[0], 0)

    # --- default padding tests ---

    def test_default_crop_padding_is_0_25(self) -> None:
        config = LMStudioConfig()
        self.assertEqual(config.crop_padding_ratio, 0.25)

    def test_wider_padding_produces_larger_bounds(self) -> None:
        box = TextBox(
            text="Test",
            width=20,
            height=10,
            center_x=50.0,
            center_y=30.0,
            confidence=0.5,
        )
        narrow = _box_bounds(box, (100, 80), padding_ratio=0.15)
        wide = _box_bounds(box, (100, 80), padding_ratio=0.25)
        # wider padding → smaller left, larger right
        self.assertLessEqual(wide[0], narrow[0])
        self.assertGreaterEqual(wide[2], narrow[2])

    def test_padding_amounts_use_short_side_for_wide_boxes(self) -> None:
        wide_box = TextBox(
            text="品名・規格",
            width=100,
            height=12,
            center_x=50.0,
            center_y=30.0,
            confidence=0.5,
        )
        pad_x, pad_y = _padding_amounts(wide_box, padding_ratio=0.25, padding_ratio_y=None)
        self.assertEqual(pad_x, pad_y)
        self.assertEqual(pad_x, 4.0)

    def test_padding_amounts_can_override_vertical_padding_from_short_side(self) -> None:
        box = TextBox(
            text="単価",
            width=80,
            height=12,
            center_x=50.0,
            center_y=30.0,
            confidence=0.5,
        )
        pad_x, pad_y = _padding_amounts(box, padding_ratio=0.15, padding_ratio_y=0.45)
        self.assertEqual(pad_x, 4.0)
        self.assertGreater(pad_y, pad_x)

    # --- hint_mode in refine_page_result integration ---

    def test_refine_page_result_with_hint_mode_none(self) -> None:
        captured_request: list[object] = []
        page = PageResult(
            source="sample.png",
            page=None,
            img_width=100,
            img_height=60,
            boxes=[
                TextBox(text="Inv0ice", width=20, height=10, center_x=10.0, center_y=10.0, confidence=0.5),
            ],
        )

        def _fake_urlopen(req, timeout=None):
            captured_request.append(req)
            return _FakeHTTPResponse({"choices": [{"message": {"content": "Invoice"}}]})

        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "sample.png"
            Image.new("RGB", (100, 60), (255, 255, 255)).save(image_path)
            with patch("ai_ocr_pipeline.llm.lmstudio.request.urlopen", side_effect=_fake_urlopen):
                refined = refine_page_result(
                    image_path,
                    page,
                    LMStudioConfig(base_url="http://127.0.0.1:1234", model="gemma-4", hint_mode="none"),
                )

        self.assertEqual(refined.boxes[0].text, "Invoice")
        body = json.loads(captured_request[0].data.decode("utf-8"))
        prompt = body["messages"][1]["content"][0]["text"]
        self.assertNotIn("OCR hint:", prompt)
        self.assertNotIn("Inv0ice", prompt)
        system = body["messages"][0]["content"]
        self.assertNotIn("hint", system.lower())

    # --- pipe character cleanup tests ---

    def test_normalize_box_text_strips_pipe_characters(self) -> None:
        self.assertEqual(_normalize_box_text("|3|6|0|4|6|0|", fallback="orig"), "360460")
        self.assertEqual(_normalize_box_text("平成|1|0|年 |1|月|2|2|日", fallback="orig"), "平成10年 1月22日")

    def test_normalize_box_text_collapses_extra_spaces_after_pipe_removal(self) -> None:
        self.assertEqual(_normalize_box_text("0| |2|5|4|7|8|9|1|0|0||著", fallback="orig"), "0 254789100著")

    def test_validate_box_text_accepts_reasonable_correction(self) -> None:
        box = TextBox(
            text="code-1",
            width=20,
            height=10,
            center_x=10.0,
            center_y=10.0,
            confidence=0.5,
        )
        self.assertEqual(_validate_box_text("alpha-2", box), "alpha-2")

    def test_validate_box_text_rejects_overlong_single_line_output(self) -> None:
        box = TextBox(
            text="10",
            width=20,
            height=10,
            center_x=10.0,
            center_y=10.0,
            confidence=0.5,
        )
        with self.assertRaises(RuntimeError):
            _validate_box_text("header field label account code amount total summary", box)

    def test_validate_box_text_rejects_non_numeric_output_for_numeric_field(self) -> None:
        box = TextBox(
            text="1234 5678 9012",
            width=40,
            height=10,
            center_x=10.0,
            center_y=10.0,
            confidence=0.5,
        )
        with self.assertRaises(RuntimeError):
            _validate_box_text("item-nameA123", box)

    def test_validate_box_text_rejects_low_digit_ratio_for_numeric_field(self) -> None:
        box = TextBox(
            text="1234 5678",
            width=40,
            height=10,
            center_x=10.0,
            center_y=10.0,
            confidence=0.5,
        )
        with self.assertRaises(RuntimeError):
            _validate_box_text("--//[]", box)

    def test_validate_box_text_rejects_non_date_output_for_date_field(self) -> None:
        box = TextBox(
            text="2024年1月2日",
            width=40,
            height=10,
            center_x=10.0,
            center_y=10.0,
            confidence=0.5,
        )
        with self.assertRaises(RuntimeError):
            _validate_box_text("product-code", box)

    def test_validate_box_text_accepts_reasonable_numeric_output(self) -> None:
        box = TextBox(
            text="1234 5678",
            width=40,
            height=10,
            center_x=10.0,
            center_y=10.0,
            confidence=0.5,
        )
        self.assertEqual(_validate_box_text("8 0 0 0", box), "8 0 0 0")

    def test_validate_box_text_accepts_alphanumeric_numeric_like_code(self) -> None:
        box = TextBox(
            text="1A41-1",
            width=40,
            height=10,
            center_x=10.0,
            center_y=10.0,
            confidence=0.5,
        )
        self.assertEqual(_validate_box_text("1A41-1", box), "1A41-1")

    # --- thinking model token budget tests ---

    def test_default_max_tokens_accommodates_thinking_models(self) -> None:
        config = LMStudioConfig()
        self.assertEqual(config.max_tokens_per_request, 4096)

    def test_refine_uses_configured_max_tokens_not_text_based_limit(self) -> None:
        captured_request: list[object] = []
        page = PageResult(
            source="sample.png",
            page=None,
            img_width=100,
            img_height=60,
            boxes=[
                TextBox(text="AB", width=20, height=10, center_x=10.0, center_y=10.0, confidence=0.5),
            ],
        )

        def _fake_urlopen(req, timeout=None):
            captured_request.append(req)
            return _FakeHTTPResponse({"choices": [{"message": {"content": "AB"}}]})

        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "sample.png"
            Image.new("RGB", (100, 60), (255, 255, 255)).save(image_path)
            with patch("ai_ocr_pipeline.llm.lmstudio.request.urlopen", side_effect=_fake_urlopen):
                refine_page_result(
                    image_path,
                    page,
                    LMStudioConfig(base_url="http://127.0.0.1:1234", model="gemma-4", max_tokens_per_request=4096),
                )

        body = json.loads(captured_request[0].data.decode("utf-8"))
        # Short text "AB" should still get the configured token budget.
        self.assertEqual(body["max_tokens"], 4096)

    def test_refine_page_result_retries_length_response_with_larger_token_budget(self) -> None:
        captured_requests: list[dict] = []
        page = PageResult(
            source="sample.png",
            page=None,
            img_width=100,
            img_height=60,
            boxes=[
                TextBox(text="Inv0ice", width=20, height=10, center_x=10.0, center_y=10.0, confidence=0.5),
            ],
        )

        responses = iter(
            [
                _FakeHTTPResponse(
                    {
                        "choices": [{"message": {"content": "Inv"}, "finish_reason": "length"}],
                        "usage": {"completion_tokens": 4096, "completion_tokens_details": {"reasoning_tokens": 4090}},
                    }
                ),
                _FakeHTTPResponse(
                    {
                        "choices": [{"message": {"content": "Invoice"}, "finish_reason": "stop"}],
                        "usage": {"completion_tokens": 64, "completion_tokens_details": {"reasoning_tokens": 40}},
                    }
                ),
            ]
        )

        def _fake_urlopen(req, timeout=None):
            captured_requests.append(json.loads(req.data.decode("utf-8")))
            return next(responses)

        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "sample.png"
            Image.new("RGB", (100, 60), (255, 255, 255)).save(image_path)
            with patch("ai_ocr_pipeline.llm.lmstudio.request.urlopen", side_effect=_fake_urlopen):
                refined, stats = refine_page_result_with_stats(
                    image_path,
                    page,
                    LMStudioConfig(base_url="http://127.0.0.1:1234", model="gemma-4", max_workers=1),
                )

        self.assertEqual(refined.boxes[0].text, "Invoice")
        self.assertEqual(len(captured_requests), 2)
        self.assertEqual(captured_requests[0]["max_tokens"], 4096)
        self.assertEqual(captured_requests[1]["max_tokens"], 16384)
        self.assertEqual(stats.finish_reason_length, 1)
        self.assertEqual(stats.finish_reason_stop, 1)
        self.assertEqual(stats.retried_on_length, 1)
        self.assertEqual(stats.fallback_count, 0)

    def test_refine_page_result_rejects_length_response_even_with_partial_content(self) -> None:
        page = PageResult(
            source="sample.png",
            page=None,
            img_width=100,
            img_height=60,
            boxes=[
                TextBox(text="49693-801-15336", width=50, height=10, center_x=10.0, center_y=10.0, confidence=0.5),
            ],
        )

        responses = iter(
            [
                _FakeHTTPResponse(
                    {
                        "choices": [{"message": {"content": "4969"}, "finish_reason": "length"}],
                        "usage": {"completion_tokens": 4096, "completion_tokens_details": {"reasoning_tokens": 4092}},
                    }
                ),
                _FakeHTTPResponse(
                    {
                        "choices": [{"message": {"content": "4969"}, "finish_reason": "length"}],
                        "usage": {"completion_tokens": 16384, "completion_tokens_details": {"reasoning_tokens": 16380}},
                    }
                ),
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "sample.png"
            Image.new("RGB", (100, 60), (255, 255, 255)).save(image_path)
            with patch(
                "ai_ocr_pipeline.llm.lmstudio.request.urlopen",
                side_effect=lambda req, timeout=None: next(responses),
            ):
                refined, stats = refine_page_result_with_stats(
                    image_path,
                    page,
                    LMStudioConfig(base_url="http://127.0.0.1:1234", model="gemma-4", max_workers=1),
                )

        self.assertEqual(refined.boxes[0].text, "49693-801-15336")
        self.assertEqual(stats.finish_reason_length, 2)
        self.assertEqual(stats.retried_on_length, 1)
        self.assertEqual(stats.fallback_count, 1)

    # --- confidence threshold tests ---

    def test_confidence_threshold_skips_high_confidence_boxes(self) -> None:
        captured_requests: list[object] = []
        page = PageResult(
            source="sample.png",
            page=None,
            img_width=100,
            img_height=60,
            boxes=[
                TextBox(text="Inv0ice", width=20, height=10, center_x=10.0, center_y=10.0, confidence=0.4),
                TextBox(text="Total", width=20, height=10, center_x=30.0, center_y=10.0, confidence=0.95),
                TextBox(text="Am0unt", width=20, height=10, center_x=50.0, center_y=10.0, confidence=0.6),
            ],
        )

        def _fake_urlopen(req, timeout=None):
            captured_requests.append(req)
            return _FakeHTTPResponse({"choices": [{"message": {"content": "Corrected"}}]})

        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "sample.png"
            Image.new("RGB", (100, 60), (255, 255, 255)).save(image_path)
            with patch("ai_ocr_pipeline.llm.lmstudio.request.urlopen", side_effect=_fake_urlopen):
                refined = refine_page_result(
                    image_path,
                    page,
                    LMStudioConfig(base_url="http://127.0.0.1:1234", model="gemma-4", confidence_threshold=0.9),
                )

        self.assertEqual(len(captured_requests), 2)
        self.assertEqual(refined.boxes[0].text, "Corrected")
        self.assertEqual(refined.boxes[1].text, "Total")  # skipped — high confidence
        self.assertEqual(refined.boxes[2].text, "Corrected")

    def test_confidence_threshold_none_refines_all(self) -> None:
        captured_requests: list[object] = []
        page = PageResult(
            source="sample.png",
            page=None,
            img_width=100,
            img_height=60,
            boxes=[
                TextBox(text="A", width=20, height=10, center_x=10.0, center_y=10.0, confidence=0.99),
            ],
        )

        def _fake_urlopen(req, timeout=None):
            captured_requests.append(req)
            return _FakeHTTPResponse({"choices": [{"message": {"content": "A"}}]})

        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "sample.png"
            Image.new("RGB", (100, 60), (255, 255, 255)).save(image_path)
            with patch("ai_ocr_pipeline.llm.lmstudio.request.urlopen", side_effect=_fake_urlopen):
                refine_page_result(image_path, page, LMStudioConfig(base_url="http://127.0.0.1:1234", model="gemma-4"))

        self.assertEqual(len(captured_requests), 1)

    # --- neighbor label tests ---

    def test_find_neighbor_labels_left(self) -> None:
        label = TextBox(text="金額", width=30, height=12, center_x=50.0, center_y=100.0, confidence=0.9)
        target = TextBox(text="1234", width=40, height=12, center_x=120.0, center_y=100.0, confidence=0.5)
        result = _find_neighbor_labels(target, [label, target])
        self.assertEqual(result, ["left: 金額"])

    def test_find_neighbor_labels_above(self) -> None:
        label = TextBox(text="日付", width=40, height=12, center_x=100.0, center_y=50.0, confidence=0.9)
        target = TextBox(text="2024", width=40, height=12, center_x=100.0, center_y=80.0, confidence=0.5)
        result = _find_neighbor_labels(target, [label, target])
        self.assertEqual(result, ["above: 日付"])

    def test_find_neighbor_labels_skips_numeric_boxes(self) -> None:
        numeric = TextBox(text="9999", width=30, height=12, center_x=50.0, center_y=100.0, confidence=0.9)
        target = TextBox(text="1234", width=40, height=12, center_x=120.0, center_y=100.0, confidence=0.5)
        result = _find_neighbor_labels(target, [numeric, target])
        self.assertEqual(result, [])

    def test_find_neighbor_labels_skips_long_text(self) -> None:
        long_label = TextBox(
            text="この欄には正確な金額を記入してくださいますようお願いします",
            width=200,
            height=12,
            center_x=50.0,
            center_y=100.0,
            confidence=0.9,
        )
        target = TextBox(text="1234", width=40, height=12, center_x=300.0, center_y=100.0, confidence=0.5)
        result = _find_neighbor_labels(target, [long_label, target])
        self.assertEqual(result, [])

    def test_find_neighbor_labels_max_labels(self) -> None:
        label1 = TextBox(text="項目A", width=30, height=12, center_x=50.0, center_y=100.0, confidence=0.9)
        label2 = TextBox(text="項目B", width=40, height=12, center_x=100.0, center_y=50.0, confidence=0.9)
        label3 = TextBox(text="項目C", width=40, height=12, center_x=100.0, center_y=30.0, confidence=0.9)
        target = TextBox(text="val", width=40, height=12, center_x=100.0, center_y=100.0, confidence=0.5)
        result = _find_neighbor_labels(target, [label1, label2, label3, target], max_labels=2)
        self.assertEqual(len(result), 2)

    # --- field type detection with neighbor labels ---

    def test_detect_field_type_uses_neighbor_for_date(self) -> None:
        # OCR text is garbled — doesn't look like a date
        self.assertEqual(_detect_field_type("令利6年3目", neighbor_labels=["left: 届出年月日"]), "date")

    def test_detect_field_type_uses_neighbor_for_numeric(self) -> None:
        # OCR text is garbled — doesn't look numeric
        self.assertEqual(_detect_field_type("l23A", neighbor_labels=["left: 金額"]), "numeric")

    def test_detect_field_type_primary_overrides_neighbor(self) -> None:
        # OCR text clearly says date — neighbor doesn't matter
        self.assertEqual(_detect_field_type("平成10年1月2日", neighbor_labels=["left: 金額"]), "date")

    def test_detect_field_type_no_neighbor_fallback(self) -> None:
        self.assertEqual(_detect_field_type("abcdef"), "text")
        self.assertEqual(_detect_field_type("abcdef", neighbor_labels=[]), "text")

    # --- prompt with neighbor context ---

    def test_prompt_includes_neighbor_context(self) -> None:
        box = TextBox(text="1234", width=20, height=10, center_x=10.0, center_y=10.0, confidence=0.5)
        prompt = _build_refine_prompt(box, hint_mode="full", neighbor_labels=["left: 金額"])
        self.assertIn("Context: left: 金額", prompt)

    def test_prompt_omits_context_when_no_neighbors(self) -> None:
        box = TextBox(text="1234", width=20, height=10, center_x=10.0, center_y=10.0, confidence=0.5)
        prompt = _build_refine_prompt(box, hint_mode="full", neighbor_labels=[])
        self.assertNotIn("Context:", prompt)

    # --- prompt structure: user prompt carries task instructions + data ---

    def test_user_prompt_has_truncation_guard(self) -> None:
        """All hint modes must include a do-not-truncate instruction."""
        box = TextBox(text="Hello", width=20, height=10, center_x=10.0, center_y=10.0, confidence=0.5)
        for mode in ("full", "weak", "none"):
            prompt = _build_refine_prompt(box, hint_mode=mode)
            self.assertIn("do not truncate", prompt, f"Missing truncation guard for hint_mode={mode}")

    def test_user_prompt_instructs_ignoring_edge_bleed(self) -> None:
        """All hint modes must warn the model about partial text at crop edges."""
        box = TextBox(text="Hello", width=20, height=10, center_x=10.0, center_y=10.0, confidence=0.5)
        for mode in ("full", "weak", "none"):
            prompt = _build_refine_prompt(box, hint_mode=mode)
            self.assertIn(
                "vertically centered",
                prompt,
                f"Missing edge-bleed guard for hint_mode={mode}",
            )

    def test_user_prompt_forbids_extra_newlines_for_single_line_hint(self) -> None:
        """With a single-line OCR hint, all hint modes must forbid introducing newlines."""
        box = TextBox(text="ID", width=80, height=60, center_x=100.0, center_y=100.0, confidence=0.4)
        for mode in ("full", "weak"):
            prompt = _build_refine_prompt(box, hint_mode=mode)
            self.assertIn(
                "single line",
                prompt,
                f"Missing single-line rule for hint_mode={mode}",
            )
            self.assertIn("Do not introduce newline", prompt, f"Missing newline ban for hint_mode={mode}")

    def test_user_prompt_allows_matching_newlines_for_multiline_hint(self) -> None:
        """If the OCR hint already has a newline, the prompt must not forbid all newlines."""
        box = TextBox(
            text="line one\nline two",
            width=80,
            height=60,
            center_x=100.0,
            center_y=100.0,
            confidence=0.4,
        )
        prompt = _build_refine_prompt(box, hint_mode="full")
        self.assertIn("line break", prompt)
        self.assertNotIn("Do not introduce newline", prompt)

    def test_user_prompt_none_mode_requires_single_line_by_default(self) -> None:
        """hint_mode=none has no OCR hint, so default to single-line output."""
        box = TextBox(text="whatever", width=80, height=60, center_x=100.0, center_y=100.0, confidence=0.4)
        prompt = _build_refine_prompt(box, hint_mode="none")
        self.assertIn("single line", prompt)
        self.assertIn("Do not emit newline", prompt)

    def test_user_prompt_full_has_conservative_fallback(self) -> None:
        """Full mode must instruct the model to return the hint when uncertain."""
        box = TextBox(text="Test", width=20, height=10, center_x=10.0, center_y=10.0, confidence=0.5)
        prompt = _build_refine_prompt(box, hint_mode="full")
        self.assertIn("return the OCR hint unchanged", prompt)

    def test_system_prompt_is_format_only(self) -> None:
        """System prompt should carry format constraints, not task instructions."""
        for mode in ("full", "weak", "none"):
            prompt = _build_system_prompt(mode)
            self.assertIn("No explanations", prompt)
            # Task instructions should NOT be in system prompt
            self.assertNotIn("truncate", prompt)
            self.assertNotIn("Orientation", prompt)


if __name__ == "__main__":
    unittest.main()
