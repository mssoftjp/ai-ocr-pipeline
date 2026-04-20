from __future__ import annotations

import inspect
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import typer
from PIL import Image
from typer.testing import CliRunner

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ai_ocr_pipeline import cli
from ai_ocr_pipeline.llm import LMStudioConfig
from ai_ocr_pipeline.models import PageResult, TextBox
from ai_ocr_pipeline.ocr import scoring as _scoring
from ai_ocr_pipeline.preprocess import image as _preprocess
from ai_ocr_pipeline.template import Template, load_template

RUNNER = CliRunner()


def _result(*, boxes: list[TextBox], source: str = "test") -> PageResult:
    return PageResult(
        source=source,
        page=None,
        img_width=100,
        img_height=100,
        boxes=boxes,
    )


def _run_cli(
    input_path: Path,
    *,
    output: Path | None = None,
    output_dir: Path | None = None,
    run_root: Path | None = None,
    overlay: bool | None = None,
    deskew: bool = False,
    remove_horizontal_lines: bool = False,
    remove_vertical_lines: bool = False,
    prefer_text_layer: bool = False,
    ocr_backend: str = "direct",
    filter_container_fallbacks: bool = True,
    split_wide_lines: bool = True,
    template: Path | None = None,
    template_boxes: str | None = None,
    engine: str | None = None,
    llm: str | None = None,
    use_lmstudio: bool = False,
    use_openai: bool = False,
    use_gemini: bool = False,
    use_ndl: bool = True,
    llm_box_indices: str | None = None,
    include_absolute_geometry: bool | None = None,
    include_debug_fields: bool | None = None,
    pretty: bool | None = None,
) -> None:
    cli.run(
        input_path,
        output=output,
        output_dir=output_dir,
        run_root=run_root,
        overlay=overlay,
        deskew=deskew,
        remove_horizontal_lines=remove_horizontal_lines,
        remove_vertical_lines=remove_vertical_lines,
        dpi=600,
        prefer_text_layer=prefer_text_layer,
        device="cpu",
        ocr_backend=ocr_backend,
        filter_container_fallbacks=filter_container_fallbacks,
        split_wide_lines=split_wide_lines,
        template=template,
        template_boxes=template_boxes,
        use_lmstudio=use_lmstudio,
        use_openai=use_openai,
        use_gemini=use_gemini,
        use_ndl=use_ndl,
        engine=engine,
        llm=llm,
        llm_base_url="http://127.0.0.1:1234/v1",
        llm_model=None,
        llm_api_key=None,
        llm_timeout=120.0,
        llm_max_tokens=4096,
        llm_hint_mode="full",
        llm_crop_padding=0.25,
        llm_confidence_threshold=None,
        llm_context_confidence=0.5,
        llm_max_workers=4,
        llm_box_indices=llm_box_indices,
        llm_save_crops=None,
        include_absolute_geometry=include_absolute_geometry,
        include_debug_fields=include_debug_fields,
        pretty=pretty,
    )


def _write_template(
    tmp_path: Path,
    *,
    preprocess: dict[str, object] | None = None,
) -> Path:
    template_path = tmp_path / "template.json"
    payload = {
        "template": {
            "name": "order-form",
            "version": 1,
            "coordinate_mode": "ratio",
        },
        "boxes": [
            {
                "id": 2,
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
    if preprocess is not None:
        payload["preprocess"] = preprocess
    template_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return template_path


class CliPreprocessingTests(unittest.TestCase):
    def test_apply_newline_handling_to_text_is_deterministic(self) -> None:
        self.assertEqual(cli._apply_newline_handling_to_text("A\nB", "first_line"), "A")
        self.assertEqual(cli._apply_newline_handling_to_text("A\r\nB", "join"), "AB")
        self.assertEqual(cli._apply_newline_handling_to_text("A\nB", "preserve"), "A\nB")

    def test_effective_template_newline_handling_defaults_to_preserve(self) -> None:
        template = Template(
            name="sample",
            version=1,
            coordinate_mode="ratio",
            boxes=[],
            preprocess_newline_handling=None,
        )

        self.assertEqual(cli._effective_template_newline_handling(template), "preserve")
        self.assertIsNone(cli._effective_template_newline_handling(None))

    def test_process_template_image_applies_newline_handling_from_template(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            image_path = tmp_path / "sample.png"
            Image.new("RGB", (40, 20), (255, 255, 255)).save(image_path)
            template = load_template(
                _write_template(
                    tmp_path,
                    preprocess={"newline_handling": "first_line"},
                )
            )
            page_result = _result(
                boxes=[
                    TextBox(
                        text="1行目\n2行目",
                        width=10,
                        height=10,
                        center_x=10.0,
                        center_y=10.0,
                        confidence=0.9,
                        order=1,
                        type="本文",
                        text_source="llm",
                    )
                ],
                source="sample.png",
            )

            with patch.object(
                cli, "refine_page_result_with_stats", return_value=(page_result, cli.LLMRefinementStats())
            ):
                result = cli._process_template_image(
                    image_path,
                    tmp_path,
                    template_obj=template,
                    lmstudio_config=LMStudioConfig(model="dummy"),
                )

        self.assertEqual(result.result.boxes[0].text, "1行目")

    def test_ocr_backend_defaults_to_direct(self) -> None:
        self.assertEqual(cli._run_ocr_for_image.__kwdefaults__["ocr_backend"], "direct")
        self.assertEqual(cli._process_image.__kwdefaults__["ocr_backend"], "direct")
        option = inspect.signature(cli.run).parameters["ocr_backend"].default
        self.assertEqual(option.default, "direct")

    def test_ensure_rgb_converts_bilevel_image(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            source = tmp_path / "sample.tif"
            Image.new("1", (4, 4), 1).save(source)

            converted = _preprocess.ensure_rgb(source, tmp_path)

            self.assertNotEqual(converted, source)
            with Image.open(converted) as image:
                self.assertEqual(image.mode, "RGB")

    def test_build_inverted_variant_creates_rgb_image_for_any_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            bilevel = tmp_path / "sample.tif"
            Image.new("1", (2, 1), 1).save(bilevel)

            inverted_path = _preprocess.build_inverted_variant(bilevel, tmp_path)

            self.assertIsNotNone(inverted_path)
            with Image.open(inverted_path) as image:
                self.assertEqual(image.mode, "RGB")
                self.assertEqual(image.getpixel((0, 0)), (0, 0, 0))

            rgb = tmp_path / "sample.png"
            Image.new("RGB", (2, 1), (255, 255, 255)).save(rgb)
            rgb_inverted_path = _preprocess.build_inverted_variant(rgb, tmp_path)
            with Image.open(rgb_inverted_path) as image:
                self.assertEqual(image.mode, "RGB")
                self.assertEqual(image.getpixel((0, 0)), (0, 0, 0))

    def test_build_line_removed_variant_creates_rgb_png(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            source = tmp_path / "sample.png"
            Image.new("RGB", (32, 32), (255, 255, 255)).save(source)

            cleaned = _preprocess.build_line_removed_variant(
                source,
                tmp_path,
                remove_horizontal_lines=True,
                remove_vertical_lines=False,
                invert_output=True,
            )

            self.assertTrue(cleaned.exists())
            with Image.open(cleaned) as image:
                self.assertEqual(image.mode, "RGB")


class CliRetryTests(unittest.TestCase):
    def test_score_result_penalizes_large_noisy_box(self) -> None:
        clean = _result(
            boxes=[
                TextBox(
                    text="Invoice",
                    width=30,
                    height=10,
                    center_x=20.0,
                    center_y=10.0,
                    confidence=0.6,
                ),
                TextBox(
                    text="Total 1234",
                    width=40,
                    height=10,
                    center_x=60.0,
                    center_y=10.0,
                    confidence=0.6,
                ),
            ]
        )
        noisy = _result(
            boxes=[
                *clean.boxes,
                TextBox(
                    text="0,,,,,,,,,00 0,,,,00",
                    width=90,
                    height=90,
                    center_x=50.0,
                    center_y=50.0,
                    confidence=0.9,
                ),
            ]
        )

        self.assertGreater(_scoring.score_result(clean), _scoring.score_result(noisy))

    def test_process_image_prefers_inverted_candidate_when_it_scores_better(self) -> None:
        primary = _result(
            boxes=[
                TextBox(
                    text="0201",
                    width=10,
                    height=10,
                    center_x=5.0,
                    center_y=5.0,
                    confidence=0.5,
                )
            ]
        )
        inverted = _result(
            boxes=[
                TextBox(
                    text="Invoice",
                    width=10,
                    height=10,
                    center_x=5.0,
                    center_y=5.0,
                    confidence=0.4,
                ),
                TextBox(
                    text="Total",
                    width=10,
                    height=10,
                    center_x=15.0,
                    center_y=5.0,
                    confidence=0.4,
                ),
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)
            image_path = work_dir / "sample.tif"
            image_path.write_bytes(b"placeholder")
            with (
                patch.object(cli, "ensure_rgb", return_value=work_dir / "primary.png"),
                patch.object(cli, "build_inverted_variant", return_value=work_dir / "inverted.png"),
                patch.object(cli, "_run_ocr_for_image", side_effect=[primary, inverted]) as run_ocr,
            ):
                result = cli._process_image(image_path, work_dir)

        self.assertIs(result.result, inverted)
        self.assertEqual(result.image_path, work_dir / "inverted.png")
        self.assertEqual(run_ocr.call_count, 2)

    def test_process_image_uses_non_inverted_image_for_lmstudio_refinement(self) -> None:
        primary = _result(
            boxes=[
                TextBox(
                    text="0201",
                    width=10,
                    height=10,
                    center_x=5.0,
                    center_y=5.0,
                    confidence=0.5,
                )
            ]
        )
        inverted = _result(
            boxes=[
                TextBox(
                    text="Invoice",
                    width=10,
                    height=10,
                    center_x=5.0,
                    center_y=5.0,
                    confidence=0.4,
                ),
                TextBox(
                    text="Total",
                    width=10,
                    height=10,
                    center_x=15.0,
                    center_y=5.0,
                    confidence=0.4,
                ),
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)
            image_path = work_dir / "sample.tif"
            image_path.write_bytes(b"placeholder")
            with (
                patch.object(cli, "ensure_rgb", return_value=work_dir / "primary.png"),
                patch.object(cli, "build_inverted_variant", return_value=work_dir / "inverted.png"),
                patch.object(cli, "_run_ocr_for_image", side_effect=[primary, inverted]),
                patch.object(
                    cli, "refine_page_result_with_stats", return_value=(inverted, cli.LLMRefinementStats())
                ) as refine,
            ):
                result = cli._process_image(
                    image_path,
                    work_dir,
                    engine="lmstudio-hybrid",
                    lmstudio_config=LMStudioConfig(base_url="http://127.0.0.1:1234", model="gemma-4"),
                )

        self.assertIs(result.result, inverted)
        self.assertEqual(result.image_path, work_dir / "inverted.png")
        self.assertEqual(refine.call_args.args[0], work_dir / "primary.png")

    def test_process_image_uses_natural_image_for_inverted_recognition(self) -> None:
        primary = _result(
            boxes=[
                TextBox(
                    text="Invoice",
                    width=10,
                    height=10,
                    center_x=5.0,
                    center_y=5.0,
                    confidence=0.5,
                )
            ]
        )
        inverted = _result(
            boxes=[
                TextBox(
                    text="Inv0ice",
                    width=10,
                    height=10,
                    center_x=5.0,
                    center_y=5.0,
                    confidence=0.6,
                ),
                TextBox(
                    text="Total",
                    width=10,
                    height=10,
                    center_x=15.0,
                    center_y=5.0,
                    confidence=0.6,
                ),
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)
            image_path = work_dir / "sample.tif"
            image_path.write_bytes(b"placeholder")
            with (
                patch.object(cli, "ensure_rgb", return_value=work_dir / "primary.png"),
                patch.object(cli, "build_inverted_variant", return_value=work_dir / "inverted.png"),
                patch.object(cli, "_run_ocr_for_image", side_effect=[primary, inverted]) as run_ocr,
            ):
                result = cli._process_image(image_path, work_dir)

        self.assertIs(result.result, inverted)
        first_call = run_ocr.call_args_list[0]
        second_call = run_ocr.call_args_list[1]
        self.assertEqual(first_call.args[0], work_dir / "primary.png")
        self.assertEqual(first_call.kwargs["recognition_image_path"], work_dir / "primary.png")
        self.assertEqual(second_call.args[0], work_dir / "inverted.png")
        self.assertEqual(second_call.kwargs["recognition_image_path"], work_dir / "primary.png")

    def test_process_image_with_deskew_uses_deskewed_variants_for_ocr(self) -> None:
        primary = _result(
            boxes=[
                TextBox(
                    text="Invoice",
                    width=10,
                    height=10,
                    center_x=5.0,
                    center_y=5.0,
                    confidence=0.4,
                )
            ]
        )
        inverted = _result(
            boxes=[
                TextBox(
                    text="Total",
                    width=10,
                    height=10,
                    center_x=15.0,
                    center_y=5.0,
                    confidence=0.4,
                )
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)
            image_path = work_dir / "sample.tif"
            image_path.write_bytes(b"placeholder")
            deskewed_primary = work_dir / "deskewed_primary.png"
            deskewed_inverted = work_dir / "deskewed_inverted.png"
            with (
                patch.object(cli, "ensure_rgb", return_value=work_dir / "primary.png"),
                patch.object(cli, "build_inverted_variant", return_value=work_dir / "inverted.png"),
                patch(
                    "ai_ocr_pipeline.preprocess.deskew.deskew_image",
                    side_effect=[deskewed_primary, deskewed_inverted],
                ),
                patch.object(cli, "_run_ocr_for_image", side_effect=[primary, inverted]) as run_ocr,
            ):
                cli._process_image(image_path, work_dir, deskew=True)

        called_paths = [call.args[0] for call in run_ocr.call_args_list]
        self.assertEqual(called_paths, [deskewed_primary, deskewed_inverted])

    def test_process_image_keeps_primary_when_it_scores_better_than_inverted(self) -> None:
        primary = _result(
            boxes=[
                TextBox(
                    text="Invoice",
                    width=10,
                    height=10,
                    center_x=5.0,
                    center_y=5.0,
                    confidence=0.4,
                ),
                TextBox(
                    text="Total",
                    width=10,
                    height=10,
                    center_x=15.0,
                    center_y=5.0,
                    confidence=0.4,
                ),
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)
            image_path = work_dir / "sample.tif"
            image_path.write_bytes(b"placeholder")
            with (
                patch.object(cli, "ensure_rgb", return_value=work_dir / "primary.png"),
                patch.object(cli, "build_inverted_variant", return_value=work_dir / "inverted.png"),
                patch.object(
                    cli,
                    "_run_ocr_for_image",
                    side_effect=[
                        primary,
                        _result(
                            boxes=[
                                TextBox(
                                    text="0,,,,,,,,,00 0,,,,00",
                                    width=90,
                                    height=90,
                                    center_x=50.0,
                                    center_y=50.0,
                                    confidence=0.9,
                                )
                            ]
                        ),
                    ],
                ) as run_ocr,
            ):
                result = cli._process_image(image_path, work_dir)

        self.assertIs(result.result, primary)
        self.assertEqual(result.image_path, work_dir / "primary.png")
        self.assertEqual(run_ocr.call_count, 2)

    def test_process_image_with_line_removal_prefers_best_candidate(self) -> None:
        primary = _result(boxes=[])
        inverted = _result(
            boxes=[
                TextBox(
                    text="Form",
                    width=10,
                    height=10,
                    center_x=5.0,
                    center_y=5.0,
                    confidence=0.3,
                )
            ]
        )
        cleaned = _result(
            boxes=[
                TextBox(
                    text="Invoice",
                    width=10,
                    height=10,
                    center_x=5.0,
                    center_y=5.0,
                    confidence=0.4,
                ),
                TextBox(
                    text="Code",
                    width=10,
                    height=10,
                    center_x=15.0,
                    center_y=5.0,
                    confidence=0.4,
                ),
            ]
        )
        cleaned_inverted = _result(
            boxes=[
                TextBox(
                    text="Invoice",
                    width=10,
                    height=10,
                    center_x=5.0,
                    center_y=5.0,
                    confidence=0.6,
                ),
                TextBox(
                    text="Code",
                    width=10,
                    height=10,
                    center_x=15.0,
                    center_y=5.0,
                    confidence=0.6,
                ),
                TextBox(
                    text="Total",
                    width=10,
                    height=10,
                    center_x=25.0,
                    center_y=5.0,
                    confidence=0.6,
                ),
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)
            image_path = work_dir / "sample.tif"
            image_path.write_bytes(b"placeholder")
            with (
                patch.object(cli, "ensure_rgb", return_value=work_dir / "primary.png"),
                patch.object(cli, "build_inverted_variant", return_value=work_dir / "inverted.png"),
                patch.object(
                    cli,
                    "build_line_removed_variant",
                    side_effect=[work_dir / "cleaned.png", work_dir / "cleaned_inverted.png"],
                ),
                patch.object(
                    cli,
                    "_run_ocr_for_image",
                    side_effect=[primary, inverted, cleaned, cleaned_inverted],
                ) as run_ocr,
            ):
                result = cli._process_image(
                    image_path,
                    work_dir,
                    remove_horizontal_lines=True,
                )

        self.assertIs(result.result, cleaned_inverted)
        self.assertEqual(result.image_path, work_dir / "cleaned_inverted.png")
        self.assertEqual(run_ocr.call_count, 4)

    def test_process_image_with_line_removal_tries_inverted_cleaned_variant_for_non_bilevel(self) -> None:
        primary = _result(
            boxes=[
                TextBox(
                    text="A",
                    width=10,
                    height=10,
                    center_x=5.0,
                    center_y=5.0,
                    confidence=0.3,
                )
            ]
        )
        cleaned = _result(
            boxes=[
                TextBox(
                    text="Invoice",
                    width=20,
                    height=10,
                    center_x=15.0,
                    center_y=5.0,
                    confidence=0.5,
                )
            ]
        )
        cleaned_inverted = _result(
            boxes=[
                TextBox(
                    text="Invoice",
                    width=20,
                    height=10,
                    center_x=15.0,
                    center_y=5.0,
                    confidence=0.6,
                ),
                TextBox(
                    text="Total",
                    width=18,
                    height=10,
                    center_x=40.0,
                    center_y=5.0,
                    confidence=0.6,
                ),
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)
            image_path = work_dir / "sample.png"
            image_path.write_bytes(b"placeholder")
            with (
                patch.object(cli, "ensure_rgb", return_value=work_dir / "primary.png"),
                patch.object(cli, "build_inverted_variant", return_value=work_dir / "inverted.png"),
                patch.object(
                    cli,
                    "build_line_removed_variant",
                    side_effect=[work_dir / "cleaned.png", work_dir / "cleaned_inverted.png"],
                ) as build_line_removed,
                patch.object(
                    cli,
                    "_run_ocr_for_image",
                    side_effect=[primary, _result(boxes=[]), cleaned, cleaned_inverted],
                ) as run_ocr,
            ):
                result = cli._process_image(
                    image_path,
                    work_dir,
                    remove_horizontal_lines=True,
                )

        self.assertIs(result.result, cleaned_inverted)
        self.assertEqual(result.image_path, work_dir / "cleaned_inverted.png")
        self.assertEqual(build_line_removed.call_count, 2)
        self.assertEqual(run_ocr.call_count, 4)

    def test_process_image_uses_non_inverted_cleaned_image_for_inverted_cleaned_recognition(self) -> None:
        primary = _result(
            boxes=[
                TextBox(
                    text="A",
                    width=10,
                    height=10,
                    center_x=5.0,
                    center_y=5.0,
                    confidence=0.3,
                )
            ]
        )
        cleaned = _result(
            boxes=[
                TextBox(
                    text="Invoice",
                    width=20,
                    height=10,
                    center_x=15.0,
                    center_y=5.0,
                    confidence=0.5,
                )
            ]
        )
        cleaned_inverted = _result(
            boxes=[
                TextBox(
                    text="Invoice",
                    width=20,
                    height=10,
                    center_x=15.0,
                    center_y=5.0,
                    confidence=0.6,
                ),
                TextBox(
                    text="Total",
                    width=18,
                    height=10,
                    center_x=40.0,
                    center_y=5.0,
                    confidence=0.6,
                ),
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)
            image_path = work_dir / "sample.png"
            image_path.write_bytes(b"placeholder")
            with (
                patch.object(cli, "ensure_rgb", return_value=work_dir / "primary.png"),
                patch.object(cli, "build_inverted_variant", return_value=work_dir / "inverted.png"),
                patch.object(
                    cli,
                    "build_line_removed_variant",
                    side_effect=[work_dir / "cleaned.png", work_dir / "cleaned_inverted.png"],
                ),
                patch.object(
                    cli,
                    "_run_ocr_for_image",
                    side_effect=[primary, _result(boxes=[]), cleaned, cleaned_inverted],
                ) as run_ocr,
            ):
                result = cli._process_image(
                    image_path,
                    work_dir,
                    remove_horizontal_lines=True,
                )

        self.assertIs(result.result, cleaned_inverted)
        fourth_call = run_ocr.call_args_list[3]
        self.assertEqual(fourth_call.args[0], work_dir / "cleaned_inverted.png")
        self.assertEqual(fourth_call.kwargs["recognition_image_path"], work_dir / "cleaned.png")

    def test_process_image_with_lmstudio_hybrid_refines_best_candidate(self) -> None:
        primary = _result(
            boxes=[
                TextBox(
                    text="Inv0ice",
                    width=10,
                    height=10,
                    center_x=5.0,
                    center_y=5.0,
                    confidence=0.3,
                )
            ]
        )
        inverted = _result(
            boxes=[
                TextBox(
                    text="Invoice",
                    width=10,
                    height=10,
                    center_x=5.0,
                    center_y=5.0,
                    confidence=0.6,
                ),
                TextBox(
                    text="Total",
                    width=10,
                    height=10,
                    center_x=15.0,
                    center_y=5.0,
                    confidence=0.6,
                ),
            ]
        )
        refined = replace(inverted, boxes=[replace(inverted.boxes[0], text="INVOICE"), inverted.boxes[1]])

        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)
            image_path = work_dir / "sample.png"
            image_path.write_bytes(b"placeholder")
            with (
                patch.object(cli, "ensure_rgb", return_value=work_dir / "primary.png"),
                patch.object(cli, "build_inverted_variant", return_value=work_dir / "inverted.png"),
                patch.object(cli, "_run_ocr_for_image", side_effect=[primary, inverted]) as run_ocr,
                patch.object(
                    cli, "refine_page_result_with_stats", return_value=(refined, cli.LLMRefinementStats())
                ) as refine,
            ):
                result = cli._process_image(
                    image_path,
                    work_dir,
                    engine="lmstudio-hybrid",
                    lmstudio_config=LMStudioConfig(base_url="http://127.0.0.1:1234", model="gemma-4"),
                )

        self.assertIs(result.result, refined)
        self.assertEqual(result.image_path, work_dir / "inverted.png")
        self.assertEqual(run_ocr.call_count, 2)
        refine.assert_called_once_with(
            work_dir / "primary.png", inverted, LMStudioConfig(base_url="http://127.0.0.1:1234", model="gemma-4")
        )

    def test_process_image_with_lmstudio_hybrid_falls_back_on_refine_error(self) -> None:
        primary = _result(
            boxes=[
                TextBox(
                    text="Inv0ice",
                    width=10,
                    height=10,
                    center_x=5.0,
                    center_y=5.0,
                    confidence=0.3,
                )
            ]
        )
        inverted = _result(
            boxes=[
                TextBox(
                    text="Invoice",
                    width=10,
                    height=10,
                    center_x=5.0,
                    center_y=5.0,
                    confidence=0.6,
                )
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)
            image_path = work_dir / "sample.png"
            image_path.write_bytes(b"placeholder")
            with (
                patch.object(cli, "ensure_rgb", return_value=work_dir / "primary.png"),
                patch.object(cli, "build_inverted_variant", return_value=work_dir / "inverted.png"),
                patch.object(cli, "_run_ocr_for_image", side_effect=[primary, inverted]),
                patch.object(cli, "refine_page_result_with_stats", side_effect=RuntimeError("bad response")),
            ):
                result = cli._process_image(
                    image_path,
                    work_dir,
                    engine="lmstudio-hybrid",
                    lmstudio_config=LMStudioConfig(base_url="http://127.0.0.1:1234", model="gemma-4"),
                )

        self.assertIs(result.result, inverted)
        self.assertEqual(result.image_path, work_dir / "inverted.png")


class CliTemplateModeTests(unittest.TestCase):
    def test_parse_int_csv_option_rejects_empty_selection(self) -> None:
        with self.assertRaisesRegex(ValueError, "Specify at least one integer"):
            cli._parse_int_csv_option(",,,", option_name="template-boxes")

    def test_process_template_image_uses_preprocessed_image_for_llm(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            source_path = tmp_path / "sample.png"
            prepared_path = tmp_path / "prepared.png"
            Image.new("RGB", (40, 20), (255, 255, 255)).save(source_path)
            Image.new("RGB", (200, 100), (255, 255, 255)).save(prepared_path)
            template = load_template(_write_template(tmp_path))
            page_result = cli.template_to_page_result(template, 200, 100, "sample.png", None)
            refined_result = replace(
                page_result,
                boxes=[
                    replace(page_result.boxes[0], text="2024/04/18", text_source="llm"),
                    replace(page_result.boxes[1], text="123,456", text_source="llm"),
                ],
            )

            with (
                patch.object(cli, "_prepare_single_image", return_value=prepared_path) as prepare,
                patch.object(
                    cli,
                    "_build_template_crop_ocr_evidence",
                    return_value=replace(
                        page_result,
                        boxes=[
                            replace(
                                page_result.boxes[0],
                                ocr_seed_text="2024/04/18",
                                ocr_seed_confidence=0.8,
                                ocr_match_count=1,
                                low_ink=False,
                            ),
                            replace(
                                page_result.boxes[1],
                                ocr_seed_text="123,456",
                                ocr_seed_confidence=0.9,
                                ocr_match_count=1,
                                low_ink=False,
                            ),
                        ],
                    ),
                ) as evidence,
                patch.object(
                    cli,
                    "refine_page_result_with_stats",
                    return_value=(refined_result, cli.LLMRefinementStats()),
                ) as refine,
            ):
                result = cli._process_template_image(
                    source_path,
                    tmp_path,
                    template_obj=template,
                    source_name="sample.png",
                    page=None,
                    lmstudio_config=LMStudioConfig(base_url="http://127.0.0.1:1234", model="gemma-4"),
                )

        prepare.assert_called_once()
        evidence.assert_called_once()
        self.assertEqual(refine.call_args.args[0], prepared_path)
        self.assertEqual(refine.call_args.kwargs["selected_box_indices"], (0, 1))
        self.assertEqual(result.variant_name, "template")
        self.assertEqual(result.image_path, prepared_path)

    def test_process_template_image_raises_when_any_box_is_unresolved(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            source_path = tmp_path / "sample.png"
            prepared_path = tmp_path / "prepared.png"
            Image.new("RGB", (40, 20), (255, 255, 255)).save(source_path)
            Image.new("RGB", (200, 100), (255, 255, 255)).save(prepared_path)
            template = load_template(_write_template(tmp_path))
            page_result = cli.template_to_page_result(template, 200, 100, "sample.png", None)

            with (
                patch.object(cli, "_prepare_single_image", return_value=prepared_path),
                patch.object(cli, "_build_template_crop_ocr_evidence", return_value=page_result),
                patch.object(
                    cli,
                    "refine_page_result_with_stats",
                    return_value=(page_result, cli.LLMRefinementStats(fallback_count=1)),
                ),
                self.assertRaisesRegex(RuntimeError, "box ids: 1, 2"),
            ):
                cli._process_template_image(
                    source_path,
                    tmp_path,
                    template_obj=template,
                    source_name="sample.png",
                    page=None,
                    lmstudio_config=LMStudioConfig(base_url="http://127.0.0.1:1234", model="gemma-4"),
                )

    def test_process_template_image_marks_unmatched_boxes_as_blank_skip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            source_path = tmp_path / "sample.png"
            prepared_path = tmp_path / "prepared.png"
            Image.new("RGB", (40, 20), (255, 255, 255)).save(source_path)
            Image.new("RGB", (200, 100), (255, 255, 255)).save(prepared_path)
            template = load_template(_write_template(tmp_path))

            with (
                patch.object(cli, "_prepare_single_image", return_value=prepared_path),
                patch.object(
                    cli,
                    "_build_template_crop_ocr_evidence",
                    return_value=replace(
                        cli.template_to_page_result(template, 200, 100, "sample.png", None),
                        boxes=[
                            replace(
                                cli.template_to_page_result(template, 200, 100, "sample.png", None).boxes[0],
                                ocr_match_count=0,
                                low_ink=True,
                            ),
                            replace(
                                cli.template_to_page_result(template, 200, 100, "sample.png", None).boxes[1],
                                ocr_match_count=0,
                                low_ink=True,
                            ),
                        ],
                    ),
                ),
                patch.object(
                    cli,
                    "refine_page_result_with_stats",
                    side_effect=lambda *args, **kwargs: (args[1], cli.LLMRefinementStats()),
                ) as refine,
            ):
                result = cli._process_template_image(
                    source_path,
                    tmp_path,
                    template_obj=template,
                    source_name="sample.png",
                    page=None,
                    lmstudio_config=LMStudioConfig(base_url="http://127.0.0.1:1234", model="gemma-4"),
                )

        self.assertEqual(refine.call_args.kwargs["selected_box_indices"], ())
        self.assertEqual([box.text_source for box in result.result.boxes], ["blank_skip", "blank_skip"])
        self.assertEqual([box.decision for box in result.result.boxes], ["blank_skip", "blank_skip"])

    def test_run_template_mode_rejects_ndl(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            image_path = tmp_path / "sample.png"
            Image.new("RGB", (40, 20), (255, 255, 255)).save(image_path)
            template_path = _write_template(tmp_path)

            with self.assertRaises(typer.Exit) as exc:
                _run_cli(image_path, template=template_path, use_ndl=True)

        self.assertEqual(exc.exception.exit_code, 1)

    def test_run_template_mode_rejects_llm_box_indices(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            image_path = tmp_path / "sample.png"
            Image.new("RGB", (40, 20), (255, 255, 255)).save(image_path)
            template_path = _write_template(tmp_path)

            with self.assertRaises(typer.Exit) as exc:
                _run_cli(
                    image_path,
                    template=template_path,
                    llm_box_indices="0,1",
                    use_ndl=False,
                    use_lmstudio=True,
                )

        self.assertEqual(exc.exception.exit_code, 1)

    def test_run_template_mode_promotes_auto_to_lmstudio_and_fails_hard(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            image_path = tmp_path / "sample.png"
            Image.new("RGB", (40, 20), (255, 255, 255)).save(image_path)
            template_path = _write_template(tmp_path)

            with (
                self.assertRaises(typer.Exit) as exc,
                patch.object(cli, "resolve_model", side_effect=RuntimeError("offline")),
            ):
                _run_cli(image_path, template=template_path, use_ndl=False)

        self.assertEqual(exc.exception.exit_code, 1)

    def test_run_template_mode_rejects_empty_template_boxes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            image_path = tmp_path / "sample.png"
            Image.new("RGB", (40, 20), (255, 255, 255)).save(image_path)
            template_path = _write_template(tmp_path)

            with self.assertRaises(typer.Exit) as exc:
                _run_cli(
                    image_path,
                    template=template_path,
                    template_boxes=",,,",
                    use_ndl=False,
                    use_lmstudio=True,
                )

        self.assertEqual(exc.exception.exit_code, 1)

    def test_run_template_mode_bypasses_pdf_text_layer(self) -> None:
        processed = cli._CandidateResult(
            image_path=Path("/tmp/template_page1.png"),
            variant_name="template",
            result=PageResult(
                source="sample.pdf",
                page=1,
                img_width=100,
                img_height=200,
                boxes=[
                    TextBox(
                        text="2024/04/18",
                        width=40,
                        height=10,
                        center_x=20.0,
                        center_y=10.0,
                        confidence=0.0,
                        order=1,
                        type="注文日",
                        text_source="llm",
                    )
                ],
            ),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            pdf_path = tmp_path / "sample.pdf"
            output_path = tmp_path / "result.json"
            template_path = _write_template(tmp_path)
            pdf_path.write_bytes(b"%PDF-1.4\n")

            with (
                patch.object(cli, "resolve_model", return_value="gemma-4"),
                patch.object(cli, "extract_pdf_text_layers") as extract_text_layers,
                patch.object(cli, "pdf_to_images", return_value=[processed.image_path]) as pdf_to_images,
                patch("pypdfium2.PdfDocument") as pdf_document,
                patch.object(cli, "_process_template_image", return_value=processed),
            ):
                pdf_document.return_value.__len__.return_value = 1
                _run_cli(
                    pdf_path,
                    output=output_path,
                    template=template_path,
                    use_ndl=False,
                    use_lmstudio=True,
                )

        extract_text_layers.assert_not_called()
        pdf_to_images.assert_called_once()

    def test_run_template_mode_writes_template_metadata(self) -> None:
        processed = cli._CandidateResult(
            image_path=Path("/tmp/template.png"),
            variant_name="template",
            result=PageResult(
                source="sample.png",
                page=None,
                img_width=100,
                img_height=50,
                boxes=[
                    TextBox(
                        text="2024/04/18",
                        width=30,
                        height=10,
                        center_x=20.0,
                        center_y=10.0,
                        confidence=0.0,
                        order=1,
                        type="注文日",
                        text_source="llm",
                    )
                ],
            ),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            image_path = tmp_path / "sample.png"
            output_path = tmp_path / "result.json"
            template_path = _write_template(
                tmp_path,
                preprocess={
                    "deskew": True,
                    "remove_horizontal_lines": True,
                    "remove_vertical_lines": False,
                    "newline_handling": "join",
                },
            )
            Image.new("RGB", (40, 20), (255, 255, 255)).save(image_path)

            with (
                patch.object(cli, "resolve_model", return_value="gemma-4"),
                patch.object(cli, "_process_template_image", return_value=processed),
            ):
                _run_cli(
                    image_path,
                    output=output_path,
                    template=template_path,
                    template_boxes="1",
                    use_ndl=False,
                    use_lmstudio=True,
                )

            payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["meta"]["ocr"]["engine"], "template")
        self.assertEqual(payload["meta"]["ocr"]["backend"], "direct")
        self.assertTrue(payload["meta"]["preprocess"]["deskew"])
        self.assertTrue(payload["meta"]["preprocess"]["remove_horizontal_lines"])
        self.assertFalse(payload["meta"]["preprocess"]["remove_vertical_lines"])
        self.assertEqual(payload["meta"]["preprocess"]["newline_handling"], "join")
        self.assertEqual(payload["meta"]["template"]["name"], "order-form")
        self.assertEqual(payload["meta"]["template"]["box_ids_requested"], [1])
        self.assertEqual(payload["meta"]["llm"]["hint_mode"], "none")
        self.assertNotIn("arguments", payload["meta"]["run"])
        self.assertNotIn("preprocess", payload["meta"]["template"])
        self.assertNotIn("ocr_image_variant", payload["results"][0])
        self.assertEqual(payload["results"][0]["img_width"], 100)
        self.assertEqual(payload["results"][0]["img_height"], 50)
        self.assertEqual(
            payload["results"][0]["boxes"][0],
            {
                "text": "2024/04/18",
                "x": 0.05,
                "y": 0.1,
                "width": 0.3,
                "height": 0.2,
                "confidence": 0.0,
                "id": 1,
            },
        )

    def test_run_template_mode_reports_effective_newline_handling_when_template_omits_it(self) -> None:
        processed = cli._CandidateResult(
            image_path=Path("/tmp/template.png"),
            variant_name="template",
            result=_result(boxes=[], source="sample.png"),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            image_path = tmp_path / "sample.png"
            output_path = tmp_path / "result.json"
            template_path = _write_template(
                tmp_path,
                preprocess={
                    "deskew": True,
                    "remove_horizontal_lines": False,
                    "remove_vertical_lines": False,
                },
            )
            Image.new("RGB", (40, 20), (255, 255, 255)).save(image_path)

            with (
                patch.object(cli, "resolve_model", return_value="gemma-4"),
                patch.object(cli, "_process_template_image", return_value=processed),
            ):
                _run_cli(
                    image_path,
                    output=output_path,
                    template=template_path,
                    use_ndl=False,
                    use_lmstudio=True,
                )

            payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["meta"]["preprocess"]["newline_handling"], "preserve")
        self.assertNotIn("preprocess", payload["meta"]["template"])

    def test_run_template_mode_can_include_absolute_geometry_and_debug_fields(self) -> None:
        processed = cli._CandidateResult(
            image_path=Path("/tmp/template.png"),
            variant_name="template",
            result=PageResult(
                source="sample.png",
                page=None,
                img_width=100,
                img_height=50,
                boxes=[
                    TextBox(
                        text="2024/04/18",
                        width=30,
                        height=10,
                        center_x=20.0,
                        center_y=10.0,
                        confidence=0.5,
                        order=1,
                        type="注文日",
                        text_source="llm",
                        box_source="template",
                        decision="ai",
                        ocr_seed_text="2024/O4/18",
                        ocr_seed_confidence=0.3,
                        ocr_match_count=1,
                        ocr_consensus_text="2024/O4/18",
                        ocr_consensus_confidence=0.3,
                        low_ink=False,
                    )
                ],
            ),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            image_path = tmp_path / "sample.png"
            output_path = tmp_path / "result.json"
            template_path = _write_template(
                tmp_path,
                preprocess={
                    "deskew": True,
                    "remove_horizontal_lines": True,
                    "remove_vertical_lines": False,
                    "newline_handling": "join",
                },
            )
            Image.new("RGB", (40, 20), (255, 255, 255)).save(image_path)

            with (
                patch.object(cli, "resolve_model", return_value="gemma-4"),
                patch.object(cli, "_process_template_image", return_value=processed),
            ):
                _run_cli(
                    image_path,
                    output=output_path,
                    template=template_path,
                    use_ndl=False,
                    use_lmstudio=True,
                    include_absolute_geometry=True,
                    include_debug_fields=True,
                )

            payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertIn("arguments", payload["meta"]["run"])
        self.assertEqual(
            payload["meta"]["template"]["preprocess"],
            {
                "deskew": True,
                "remove_horizontal_lines": True,
                "remove_vertical_lines": False,
                "newline_handling": "join",
            },
        )
        self.assertEqual(payload["results"][0]["ocr_image_variant"], "template")
        self.assertEqual(payload["results"][0]["img_width"], 100)
        self.assertEqual(payload["results"][0]["img_height"], 50)
        self.assertEqual(payload["results"][0]["boxes"][0]["pixel_x"], 5.0)
        self.assertEqual(payload["results"][0]["boxes"][0]["pixel_y"], 5.0)
        self.assertEqual(payload["results"][0]["boxes"][0]["pixel_width"], 30)
        self.assertEqual(payload["results"][0]["boxes"][0]["pixel_height"], 10)
        self.assertEqual(payload["results"][0]["boxes"][0]["text_source"], "llm")
        self.assertEqual(payload["results"][0]["boxes"][0]["ocr_seed_text"], "2024/O4/18")

    def test_cli_template_mode_uses_template_preprocess_defaults_when_flags_omitted(self) -> None:
        processed = cli._CandidateResult(
            image_path=Path("/tmp/template.png"),
            variant_name="template",
            result=_result(boxes=[], source="sample.png"),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            image_path = tmp_path / "sample.png"
            template_path = _write_template(
                tmp_path,
                preprocess={
                    "deskew": True,
                    "remove_horizontal_lines": True,
                    "remove_vertical_lines": False,
                    "newline_handling": "join",
                },
            )
            Image.new("RGB", (40, 20), (255, 255, 255)).save(image_path)

            with (
                patch.object(cli, "resolve_model", return_value="gemma-4"),
                patch.object(cli, "_process_template_image", return_value=processed) as process_template_image,
            ):
                result = RUNNER.invoke(cli.app, [str(image_path), "--template", str(template_path), "--lmstudio"])

        self.assertEqual(result.exit_code, 0)
        self.assertTrue(process_template_image.call_args.kwargs["deskew"])
        self.assertTrue(process_template_image.call_args.kwargs["remove_horizontal_lines"])
        self.assertFalse(process_template_image.call_args.kwargs["remove_vertical_lines"])

    def test_cli_template_mode_cli_flags_override_template_preprocess_defaults(self) -> None:
        processed = cli._CandidateResult(
            image_path=Path("/tmp/template.png"),
            variant_name="template",
            result=_result(boxes=[], source="sample.png"),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            image_path = tmp_path / "sample.png"
            template_path = _write_template(
                tmp_path,
                preprocess={
                    "deskew": True,
                    "remove_horizontal_lines": True,
                    "remove_vertical_lines": False,
                    "newline_handling": "join",
                },
            )
            Image.new("RGB", (40, 20), (255, 255, 255)).save(image_path)

            with (
                patch.object(cli, "resolve_model", return_value="gemma-4"),
                patch.object(cli, "_process_template_image", return_value=processed) as process_template_image,
            ):
                result = RUNNER.invoke(
                    cli.app,
                    [
                        str(image_path),
                        "--template",
                        str(template_path),
                        "--lmstudio",
                        "--no-deskew",
                        "--no-remove-horizontal-lines",
                        "--remove-vertical-lines",
                    ],
                )

        self.assertEqual(result.exit_code, 0)
        self.assertFalse(process_template_image.call_args.kwargs["deskew"])
        self.assertFalse(process_template_image.call_args.kwargs["remove_horizontal_lines"])
        self.assertTrue(process_template_image.call_args.kwargs["remove_vertical_lines"])


class CliOutputTests(unittest.TestCase):
    def test_run_output_file_defaults_to_pretty_json(self) -> None:
        result = _result(
            boxes=[
                TextBox(
                    text="Invoice",
                    width=20,
                    height=10,
                    center_x=15.0,
                    center_y=10.0,
                    confidence=0.8,
                )
            ],
            source="sample.png",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            image_path = tmp_path / "sample.png"
            output_path = tmp_path / "result.json"
            Image.new("RGB", (40, 20), (255, 255, 255)).save(image_path)

            with patch.object(
                cli,
                "_process_image",
                return_value=cli._CandidateResult(image_path=image_path, variant_name="natural", result=result),
            ):
                _run_cli(image_path, output=output_path)

            content = output_path.read_text(encoding="utf-8")

        self.assertIn('\n  "meta": {', content)

    def test_run_root_creates_timestamped_bundle_under_input_name(self) -> None:
        result = _result(
            boxes=[
                TextBox(
                    text="Invoice",
                    width=20,
                    height=10,
                    center_x=15.0,
                    center_y=10.0,
                    confidence=0.8,
                )
            ],
            source="sample.png",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            image_path = tmp_path / "sample.png"
            run_root = tmp_path / "runs"
            Image.new("RGB", (40, 20), (255, 255, 255)).save(image_path)

            with (
                patch.object(
                    cli,
                    "_process_image",
                    return_value=cli._CandidateResult(image_path=image_path, variant_name="natural", result=result),
                ),
                patch.object(cli, "_now_iso", return_value="2026-04-16T15:42:30+09:00"),
            ):
                _run_cli(image_path, run_root=run_root)

            expected_dir = run_root / "sample" / "20260416-154230"
            self.assertTrue((expected_dir / "sample.json").exists())

    def test_run_does_not_write_overlay_for_output_file_by_default(self) -> None:
        result = _result(
            boxes=[
                TextBox(
                    text="Invoice",
                    width=20,
                    height=10,
                    center_x=15.0,
                    center_y=10.0,
                    confidence=0.8,
                )
            ],
            source="sample.png",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            image_path = tmp_path / "sample.png"
            output_path = tmp_path / "result.json"
            Image.new("RGB", (40, 20), (255, 255, 255)).save(image_path)

            with (
                patch.object(
                    cli,
                    "_process_image",
                    return_value=cli._CandidateResult(image_path=image_path, variant_name="natural", result=result),
                ),
                patch.object(cli, "write_overlay_artifact") as write_overlay,
            ):
                _run_cli(image_path, output=output_path)

                self.assertTrue(output_path.exists())
                write_overlay.assert_not_called()

    def test_run_writes_overlay_for_output_dir_by_default(self) -> None:
        result = _result(
            boxes=[
                TextBox(
                    text="Invoice",
                    width=20,
                    height=10,
                    center_x=15.0,
                    center_y=10.0,
                    confidence=0.8,
                )
            ],
            source="sample.png",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            image_path = tmp_path / "sample.png"
            output_dir = tmp_path / "results"
            overlay_path = output_dir / "sample_overlay.png"
            Image.new("RGB", (40, 20), (255, 255, 255)).save(image_path)

            with (
                patch.object(
                    cli,
                    "_process_image",
                    return_value=cli._CandidateResult(image_path=image_path, variant_name="natural", result=result),
                ),
                patch.object(cli, "write_overlay_artifact", return_value=("png", overlay_path)) as write_overlay,
            ):
                _run_cli(image_path, output_dir=output_dir)

                self.assertTrue((output_dir / "sample.json").exists())
                write_overlay.assert_called_once_with(result, image_path, overlay_path)

    def test_run_skips_overlay_for_output_dir_when_disabled(self) -> None:
        result = _result(
            boxes=[
                TextBox(
                    text="Invoice",
                    width=20,
                    height=10,
                    center_x=15.0,
                    center_y=10.0,
                    confidence=0.8,
                )
            ],
            source="sample.png",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            image_path = tmp_path / "sample.png"
            output_dir = tmp_path / "results"
            Image.new("RGB", (40, 20), (255, 255, 255)).save(image_path)

            with (
                patch.object(
                    cli,
                    "_process_image",
                    return_value=cli._CandidateResult(image_path=image_path, variant_name="natural", result=result),
                ),
                patch.object(cli, "write_overlay_artifact") as write_overlay,
            ):
                _run_cli(image_path, output_dir=output_dir, overlay=False)

                self.assertTrue((output_dir / "sample.json").exists())
                write_overlay.assert_not_called()

    def test_run_stdout_mode_writes_json_only_by_default(self) -> None:
        result = _result(
            boxes=[
                TextBox(
                    text="Invoice",
                    width=20,
                    height=10,
                    center_x=15.0,
                    center_y=10.0,
                    confidence=0.8,
                )
            ],
            source="sample.png",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            image_path = tmp_path / "sample.png"
            Image.new("RGB", (40, 20), (255, 255, 255)).save(image_path)
            stdout = StringIO()

            with (
                patch.object(
                    cli,
                    "_process_image",
                    return_value=cli._CandidateResult(image_path=image_path, variant_name="natural", result=result),
                ),
                patch.object(cli, "write_overlay_artifact") as write_overlay,
                redirect_stdout(stdout),
            ):
                _run_cli(image_path)

        payload = json.loads(stdout.getvalue())
        self.assertIsNone(payload["meta"]["llm"])
        self.assertEqual(payload["meta"]["ocr"]["engine"], "ndlocr-lite")
        self.assertEqual(payload["meta"]["image"]["dpi"], None)
        self.assertEqual(payload["results"][0]["source"], "sample.png")
        self.assertEqual(payload["results"][0]["ocr_image_variant"], "natural")
        self.assertEqual(payload["results"][0]["box_count"], 1)
        self.assertAlmostEqual(payload["results"][0]["boxes"][0]["x"], 0.05)
        self.assertAlmostEqual(payload["results"][0]["boxes"][0]["y"], 0.05)
        self.assertAlmostEqual(payload["results"][0]["boxes"][0]["width"], 0.2)
        self.assertAlmostEqual(payload["results"][0]["boxes"][0]["height"], 0.1)
        write_overlay.assert_not_called()

    def test_run_renders_overlay_backgrounds_for_pdf(self) -> None:
        result = PageResult(
            source="sample.pdf",
            page=1,
            img_width=100,
            img_height=200,
            boxes=[
                TextBox(
                    text="Receipt",
                    width=40,
                    height=10,
                    center_x=20.0,
                    center_y=10.0,
                    confidence=1.0,
                    type="text_layer",
                )
            ],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            pdf_path = tmp_path / "sample.pdf"
            output_dir = tmp_path / "results"
            rendered_page = tmp_path / "page1.png"
            pdf_path.write_bytes(b"%PDF-1.4\n")

            with (
                patch.object(cli, "extract_pdf_text_layers", return_value=[result]),
                patch.object(cli, "pdf_to_images", return_value=[rendered_page]) as pdf_to_images,
                patch.object(
                    cli, "write_overlay_artifact", return_value=("png", output_dir / "sample_p0001_overlay.png")
                ) as write_overlay,
            ):
                _run_cli(pdf_path, output_dir=output_dir, prefer_text_layer=True)

                self.assertTrue((output_dir / "sample.json").exists())
                pdf_to_images.assert_called_once()
                write_overlay.assert_called_once_with(result, rendered_page, output_dir / "sample_p0001_overlay.png")
                payload = json.loads((output_dir / "sample.json").read_text(encoding="utf-8"))
                self.assertIsNone(payload["results"][0]["ocr_image_variant"])

    def test_run_reuses_ocr_page_image_for_pdf_overlay(self) -> None:
        processed = cli._CandidateResult(
            image_path=Path("/tmp/ocr_page1.png"),
            variant_name="natural",
            result=PageResult(
                source="sample.pdf",
                page=1,
                img_width=100,
                img_height=200,
                boxes=[
                    TextBox(
                        text="Receipt",
                        width=40,
                        height=10,
                        center_x=20.0,
                        center_y=10.0,
                        confidence=0.9,
                    )
                ],
            ),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            pdf_path = tmp_path / "sample.pdf"
            output_dir = tmp_path / "results"
            pdf_path.write_bytes(b"%PDF-1.4\n")

            with (
                patch.object(cli, "extract_pdf_text_layers", return_value=[None]),
                patch.object(cli, "pdf_to_images", return_value=[processed.image_path]) as pdf_to_images,
                patch.object(cli, "_process_image", return_value=processed),
                patch.object(
                    cli, "write_overlay_artifact", return_value=("png", output_dir / "sample_p0001_overlay.png")
                ) as write_overlay,
            ):
                _run_cli(pdf_path, output_dir=output_dir, prefer_text_layer=True)

                self.assertTrue((output_dir / "sample.json").exists())
                pdf_to_images.assert_called_once()
                write_overlay.assert_called_once_with(
                    processed.result,
                    processed.image_path,
                    output_dir / "sample_p0001_overlay.png",
                )

    def test_run_pdf_defaults_to_ocr_without_text_layer_opt_in(self) -> None:
        processed = cli._CandidateResult(
            image_path=Path("/tmp/ocr_page1.png"),
            variant_name="natural",
            result=PageResult(
                source="sample.pdf",
                page=1,
                img_width=100,
                img_height=200,
                boxes=[
                    TextBox(
                        text="Receipt",
                        width=40,
                        height=10,
                        center_x=20.0,
                        center_y=10.0,
                        confidence=0.9,
                    )
                ],
            ),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            pdf_path = tmp_path / "sample.pdf"
            output_path = tmp_path / "result.json"
            pdf_path.write_bytes(b"%PDF-1.4\n")

            with (
                patch.object(cli, "extract_pdf_text_layers") as extract_text_layers,
                patch.object(cli, "pdf_to_images", return_value=[processed.image_path]) as pdf_to_images,
                patch.object(cli, "_process_image", return_value=processed),
                patch("pypdfium2.PdfDocument") as pdf_document,
            ):
                pdf_document.return_value.__len__.return_value = 1
                _run_cli(pdf_path, output=output_path)

        extract_text_layers.assert_not_called()
        pdf_to_images.assert_called_once()

    def test_run_uses_deterministic_directory_overlay_names_for_same_stem_files(self) -> None:
        results = [
            cli._CandidateResult(
                image_path=Path("/tmp/receipt_png.png"),
                variant_name="natural",
                result=_result(
                    boxes=[TextBox(text="A", width=10, height=10, center_x=5.0, center_y=5.0, confidence=0.9)],
                    source="receipt.png",
                ),
            ),
            cli._CandidateResult(
                image_path=Path("/tmp/receipt_jpg.png"),
                variant_name="natural",
                result=_result(
                    boxes=[TextBox(text="B", width=10, height=10, center_x=5.0, center_y=5.0, confidence=0.9)],
                    source="receipt.jpg",
                ),
            ),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_dir = tmp_path / "invoices"
            output_dir = tmp_path / "results"
            input_dir.mkdir()
            (input_dir / "receipt.png").write_bytes(b"png")
            (input_dir / "receipt.jpg").write_bytes(b"jpg")

            with (
                patch.object(cli, "_process_image", side_effect=results),
                patch.object(
                    cli, "write_overlay_artifact", side_effect=lambda result, image, dest: ("png", dest)
                ) as write_overlay,
            ):
                _run_cli(input_dir, output_dir=output_dir)

        overlay_paths = sorted(call.args[2] for call in write_overlay.call_args_list)
        self.assertEqual(
            overlay_paths,
            sorted(
                [
                    output_dir / "receipt_png_overlay.png",
                    output_dir / "receipt_jpg_overlay.png",
                ]
            ),
        )


class CliInterfaceTests(unittest.TestCase):
    def test_cli_ndl_flag_skips_ai(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            image_path = tmp_path / "sample.png"
            Image.new("RGB", (10, 10), (255, 255, 255)).save(image_path)
            processed = cli._CandidateResult(
                image_path=image_path,
                variant_name="natural",
                result=_result(
                    boxes=[TextBox(text="A", width=10, height=10, center_x=5.0, center_y=5.0, confidence=0.9)],
                    source="sample.png",
                ),
            )

            with patch.object(cli, "_process_image", return_value=processed):
                result = RUNNER.invoke(cli.app, [str(image_path), "--ndl"])

        self.assertEqual(result.exit_code, 0)
        self.assertIn("AI: off", result.output)
        self.assertIn('"llm": null', result.output)

    def test_cli_auto_mode_falls_back_when_lmstudio_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            image_path = tmp_path / "sample.png"
            Image.new("RGB", (10, 10), (255, 255, 255)).save(image_path)
            processed = cli._CandidateResult(
                image_path=image_path,
                variant_name="natural",
                result=_result(
                    boxes=[TextBox(text="A", width=10, height=10, center_x=5.0, center_y=5.0, confidence=0.9)],
                    source="sample.png",
                ),
            )

            with (
                patch.object(cli, "_process_image", return_value=processed),
                patch.object(cli, "resolve_model", side_effect=RuntimeError("Connection refused")),
            ):
                result = RUNNER.invoke(cli.app, [str(image_path)])

        self.assertEqual(result.exit_code, 0)
        self.assertIn("Falling back to OCR-only", result.output)
        self.assertIn("AI: off", result.output)

    def test_cli_lmstudio_flag_errors_on_connection_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            image_path = tmp_path / "sample.png"
            Image.new("RGB", (10, 10), (255, 255, 255)).save(image_path)

            with patch.object(cli, "resolve_model", side_effect=RuntimeError("Connection refused")):
                result = RUNNER.invoke(cli.app, [str(image_path), "--lmstudio"])

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Failed to connect to LM Studio", result.output)

    def test_cli_rejects_multiple_ai_flags(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            image_path = tmp_path / "sample.png"
            Image.new("RGB", (10, 10), (255, 255, 255)).save(image_path)

            result = RUNNER.invoke(cli.app, [str(image_path), "--lmstudio", "--ndl"])

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("mutually exclusive", result.output)

    def test_cli_rejects_new_flag_with_deprecated_llm(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            image_path = tmp_path / "sample.png"
            Image.new("RGB", (10, 10), (255, 255, 255)).save(image_path)

            result = RUNNER.invoke(cli.app, [str(image_path), "--ndl", "--llm", "lmstudio"])

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Cannot combine", result.output)

    def test_cli_openai_not_yet_implemented(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            image_path = tmp_path / "sample.png"
            Image.new("RGB", (10, 10), (255, 255, 255)).save(image_path)

            result = RUNNER.invoke(cli.app, [str(image_path), "--openai"])

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("not yet implemented", result.output)

    def test_cli_deprecated_engine_shows_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            image_path = tmp_path / "sample.png"
            Image.new("RGB", (10, 10), (255, 255, 255)).save(image_path)

            with patch.object(cli, "resolve_model", side_effect=RuntimeError("no server")):
                result = RUNNER.invoke(cli.app, [str(image_path), "--engine", "lmstudio-hybrid"])

        self.assertIn("deprecated", result.output.lower())
        self.assertIn("--lmstudio", result.output)

    def test_cli_rejects_output_and_output_dir_together(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            image_path = tmp_path / "sample.png"
            output_dir = tmp_path / "results"
            Image.new("RGB", (10, 10), (255, 255, 255)).save(image_path)

            result = RUNNER.invoke(cli.app, [str(image_path), "-o", "out.json", "-d", str(output_dir)])

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("--output, --output-dir, and --run-root are mutually exclusive", result.output)

    def test_cli_rejects_output_dir_and_run_root_together(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            image_path = tmp_path / "sample.png"
            output_dir = tmp_path / "results"
            run_root = tmp_path / "runs"
            Image.new("RGB", (10, 10), (255, 255, 255)).save(image_path)

            result = RUNNER.invoke(cli.app, [str(image_path), "-d", str(output_dir), "--run-root", str(run_root)])

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("--output, --output-dir, and --run-root are mutually exclusive", result.output)

    def test_cli_rejects_engine_and_llm_together(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            image_path = tmp_path / "sample.png"
            Image.new("RGB", (10, 10), (255, 255, 255)).save(image_path)

            result = RUNNER.invoke(cli.app, [str(image_path), "--engine", "lmstudio-hybrid", "--llm", "lmstudio"])

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("--engine is deprecated", result.output)

    def test_cli_ndl_warns_on_explicit_llm_params(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            image_path = tmp_path / "sample.png"
            Image.new("RGB", (10, 10), (255, 255, 255)).save(image_path)
            processed = cli._CandidateResult(
                image_path=image_path,
                variant_name="natural",
                result=_result(
                    boxes=[TextBox(text="A", width=10, height=10, center_x=5.0, center_y=5.0, confidence=0.9)],
                    source="sample.png",
                ),
            )

            with patch.object(cli, "_process_image", return_value=processed):
                result = RUNNER.invoke(cli.app, [str(image_path), "--ndl", "--llm-model", "gemma-test"])

        self.assertEqual(result.exit_code, 0)
        self.assertIn("--llm-* options ignored in --ndl mode", result.output)

    def test_cli_pretty_forces_pretty_json_on_non_tty(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            image_path = tmp_path / "sample.png"
            Image.new("RGB", (10, 10), (255, 255, 255)).save(image_path)
            processed = cli._CandidateResult(
                image_path=image_path,
                variant_name="natural",
                result=_result(
                    boxes=[TextBox(text="A", width=10, height=10, center_x=5.0, center_y=5.0, confidence=0.9)],
                    source="sample.png",
                ),
            )

            with patch.object(cli, "_process_image", return_value=processed):
                result = RUNNER.invoke(cli.app, [str(image_path), "--pretty"])

        self.assertEqual(result.exit_code, 0)
        self.assertIn('\n  "meta": {', result.output)

    def test_run_no_pretty_forces_compact_json_even_when_tty(self) -> None:
        processed = cli._CandidateResult(
            image_path=Path("/tmp/sample.png"),
            variant_name="natural",
            result=_result(
                boxes=[TextBox(text="A", width=10, height=10, center_x=5.0, center_y=5.0, confidence=0.9)],
                source="sample.png",
            ),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            image_path = tmp_path / "sample.png"
            Image.new("RGB", (10, 10), (255, 255, 255)).save(image_path)
            stdout = StringIO()

            with (
                patch.object(cli, "_process_image", return_value=processed),
                patch("ai_ocr_pipeline.cli.sys.stdout.isatty", return_value=True),
                redirect_stdout(stdout),
            ):
                _run_cli(image_path, pretty=False)

        self.assertTrue(stdout.getvalue().startswith('{"meta":'))


if __name__ == "__main__":
    unittest.main()
