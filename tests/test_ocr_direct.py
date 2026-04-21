"""Tests for ai_ocr_pipeline.ocr.direct."""

from __future__ import annotations

import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from ai_ocr_pipeline.models import TextBox
from ai_ocr_pipeline.ocr.direct import (
    _detections_to_resultobj,
    _extract_line_crop,
    _find_gap_intervals,
    _find_text_and_gap_runs,
    _recognition_image_for_box,
    _score_and_filter_gaps,
    deduplicate_lines,
    drop_container_fallback_lines,
    filter_oversized_lines,
    run_direct_ocr,
    split_wide_lines_at_whitespace,
    suppress_contained_fragments,
)


class DropContainerFallbackLinesTests(unittest.TestCase):
    """Verify the D1 filter that removes container-as-LINE degenerations.

    ``ndl_parser.convert_to_xml_string3`` emits ``<LINE>`` without a ``CONF``
    attribute when a text_block / block_table / block_ad has no detected
    ``line_*`` children. Those LINEs cause PARSEQ to read an oversized crop
    and produce noisy/concatenated text.
    """

    def test_drops_line_without_conf(self) -> None:
        xml = """<OCRDATASET><PAGE>
            <TEXTBLOCK>
              <LINE TYPE="本文" X="0" Y="0" WIDTH="100" HEIGHT="20"/>
            </TEXTBLOCK>
        </PAGE></OCRDATASET>"""
        root = ET.fromstring(xml)

        dropped = drop_container_fallback_lines(root)

        self.assertEqual(dropped, 1)
        self.assertEqual(len(list(root.iter("LINE"))), 0)

    def test_keeps_line_with_conf(self) -> None:
        xml = """<OCRDATASET><PAGE>
            <TEXTBLOCK>
              <LINE TYPE="本文" X="0" Y="0" WIDTH="100" HEIGHT="20" CONF="0.812"/>
            </TEXTBLOCK>
        </PAGE></OCRDATASET>"""
        root = ET.fromstring(xml)

        dropped = drop_container_fallback_lines(root)

        self.assertEqual(dropped, 0)
        self.assertEqual(len(list(root.iter("LINE"))), 1)

    def test_mixed_keeps_only_confident_lines(self) -> None:
        xml = """<OCRDATASET><PAGE>
            <BLOCK TYPE="表">
              <LINE TYPE="本文" X="0" Y="0" WIDTH="100" HEIGHT="20"/>
              <LINE TYPE="本文" X="0" Y="30" WIDTH="50" HEIGHT="18" CONF="0.6"/>
            </BLOCK>
            <TEXTBLOCK>
              <LINE TYPE="本文" X="0" Y="60" WIDTH="80" HEIGHT="22" CONF="0.4"/>
              <LINE TYPE="本文" X="0" Y="90" WIDTH="400" HEIGHT="400"/>
            </TEXTBLOCK>
            <LINE TYPE="本文" X="200" Y="10" WIDTH="30" HEIGHT="15" CONF="0.3"/>
        </PAGE></OCRDATASET>"""
        root = ET.fromstring(xml)

        dropped = drop_container_fallback_lines(root)

        self.assertEqual(dropped, 2)
        kept = list(root.iter("LINE"))
        self.assertEqual(len(kept), 3)
        self.assertTrue(all(line.get("CONF") is not None for line in kept))


class DetectionsToResultObjTests(unittest.TestCase):
    """``_detections_to_resultobj`` must group detections by class as
    ``ocr.py`` does so that ``convert_to_xml_string3`` receives the shape
    it expects."""

    def test_groups_detections_by_class_and_isolates_class_zero(self) -> None:
        detections = [
            {"box": (10, 20, 110, 40), "confidence": 0.9, "pred_char_count": 3.0, "class_index": 0},
            {"box": (0, 0, 50, 10), "confidence": 0.5, "pred_char_count": 2.0, "class_index": 1},
            {"box": (60, 60, 80, 90), "confidence": 0.3, "pred_char_count": 1.0, "class_index": 1},
            {"box": (0, 0, 10, 10), "confidence": 0.2, "pred_char_count": 1.0, "class_index": 15},
        ]

        result = _detections_to_resultobj(detections)

        # class 0 has a dedicated slot on resultobj[0]
        self.assertEqual(result[0][0], [[10, 20, 110, 40]])
        # every class also has per-class entries on resultobj[1]
        self.assertEqual(len(result[1][0]), 1)
        self.assertEqual(len(result[1][1]), 2)
        self.assertEqual(len(result[1][15]), 1)
        # unused classes are present but empty
        self.assertEqual(result[1][2], [])
        # per-class rows carry bbox + conf + char_count (6 values)
        self.assertEqual(len(result[1][1][0]), 6)


class FindGapIntervalsTests(unittest.TestCase):
    def test_returns_intervals_meeting_min_length(self) -> None:
        # Positions: 0..2 text, 3..7 gap (5 wide), 8..10 text, 11..12 small gap (2 wide)
        density = np.array([0.5, 0.5, 0.5, 0, 0, 0, 0, 0, 0.5, 0.5, 0.5, 0, 0], dtype=np.float32)
        gaps = _find_gap_intervals(density, text_threshold=0.02, min_gap_px=5)

        self.assertEqual(gaps, [(3, 8)])

    def test_returns_trailing_gap_when_eligible(self) -> None:
        density = np.array([0.5, 0.5, 0, 0, 0, 0], dtype=np.float32)
        gaps = _find_gap_intervals(density, text_threshold=0.02, min_gap_px=3)

        self.assertEqual(gaps, [(2, 6)])


class FindTextAndGapRunsTests(unittest.TestCase):
    def test_promotes_narrow_blank_runs_into_text(self) -> None:
        density = np.array([0.6, 0.6, 0.0, 0.6, 0.6, 0.0, 0.0, 0.0, 0.6], dtype=np.float32)

        runs = _find_text_and_gap_runs(density, text_threshold=0.02, min_gap_px=3)

        self.assertEqual(runs, [(0, 5, True), (5, 8, False), (8, 9, True)])

    def test_keeps_leading_and_trailing_blank_runs(self) -> None:
        density = np.array([0.0, 0.0, 0.7, 0.7, 0.0, 0.0], dtype=np.float32)

        runs = _find_text_and_gap_runs(density, text_threshold=0.02, min_gap_px=2)

        self.assertEqual(runs, [(0, 2, False), (2, 4, True), (4, 6, False)])


class ScoreAndFilterGapsTests(unittest.TestCase):
    def test_keeps_gap_when_ratio_exceeds_threshold(self) -> None:
        runs = [(0, 100, True), (100, 170, False), (170, 260, True)]

        gaps = _score_and_filter_gaps(runs, threshold=0.5, min_text_run_px=20)

        self.assertEqual(gaps, [(100, 170)])

    def test_uses_min_text_run_floor_to_avoid_noise_splits(self) -> None:
        runs = [(0, 8, True), (8, 18, False), (18, 180, True)]

        gaps = _score_and_filter_gaps(runs, threshold=0.5, min_text_run_px=40)

        self.assertEqual(gaps, [])

    def test_skips_gaps_without_text_on_both_sides(self) -> None:
        runs = [(0, 20, False), (20, 80, True), (80, 140, False)]

        gaps = _score_and_filter_gaps(runs, threshold=0.1, min_text_run_px=10)

        self.assertEqual(gaps, [])


class ExtractLineCropTests(unittest.TestCase):
    def test_keeps_horizontal_crop_unchanged(self) -> None:
        image = np.full((20, 40, 3), 255, dtype=np.uint8)
        image[5:15, 10:20] = 0

        crop = _extract_line_crop(
            image,
            x=10,
            y=5,
            width=10,
            height=10,
            min_aspect_ratio=1.0,
        )

        self.assertEqual(crop.shape[:2], (10, 10))
        self.assertTrue(np.all(crop == 0))

    def test_expands_narrow_crop_to_avoid_rotation(self) -> None:
        image = np.full((30, 30, 3), 255, dtype=np.uint8)
        image[8:22, 14:16] = 0

        crop = _extract_line_crop(
            image,
            x=14,
            y=8,
            width=2,
            height=14,
            min_aspect_ratio=1.0,
        )

        self.assertEqual(crop.shape[:2], (14, 14))
        self.assertTrue(np.all(crop[:, 6:8] == 0))
        self.assertTrue(np.all(crop[:, :6] == 255))
        self.assertTrue(np.all(crop[:, 8:] == 255))

    def test_uses_available_image_context_near_edge(self) -> None:
        image = np.full((12, 12, 3), 255, dtype=np.uint8)
        image[:, 0:2] = 0

        crop = _extract_line_crop(
            image,
            x=0,
            y=1,
            width=2,
            height=10,
            min_aspect_ratio=1.0,
        )

        self.assertEqual(crop.shape[:2], (10, 10))
        self.assertTrue(np.all(crop[:, :6] == 0))
        self.assertTrue(np.all(crop[:, 6:] == 255))


class RecognitionImageSelectionTests(unittest.TestCase):
    def test_uses_detection_image_for_horizontal_boxes(self) -> None:
        detection = np.zeros((20, 20, 3), dtype=np.uint8)
        recognition = np.full((20, 20, 3), 255, dtype=np.uint8)

        selected = _recognition_image_for_box(
            detection,
            recognition,
            width=20,
            height=10,
        )

        self.assertIs(selected, detection)

    def test_uses_recognition_image_for_narrow_boxes(self) -> None:
        detection = np.zeros((20, 20, 3), dtype=np.uint8)
        recognition = np.full((20, 20, 3), 255, dtype=np.uint8)

        selected = _recognition_image_for_box(
            detection,
            recognition,
            width=8,
            height=20,
        )

        self.assertIs(selected, recognition)


class SplitWideLinesTests(unittest.TestCase):
    """``split_wide_lines_at_whitespace`` should replace an oversized
    horizontal LINE with one LINE per text segment."""

    def _make_line_image(
        self,
        page_size: tuple[int, int],
        line_box: tuple[int, int, int, int],
        segments: list[tuple[int, int]],
    ) -> np.ndarray:
        """Paint a thin dark text stripe at ``segments`` (x-ranges within the line).

        The stripe occupies the middle 25% of the line height so that text
        is a minority of pixels in the crop — mirroring real OCR crops.
        """
        img = np.full((page_size[1], page_size[0], 3), 255, dtype=np.uint8)
        x, y, _w, h = line_box
        stripe_y0 = y + int(h * 0.375)
        stripe_y1 = y + int(h * 0.625)
        for seg_start, seg_end in segments:
            img[stripe_y0:stripe_y1, x + seg_start : x + seg_end] = 0
        return img

    def test_splits_when_gap_exceeds_threshold(self) -> None:
        xml = """<OCRDATASET><PAGE>
            <LINE TYPE="本文" X="10" Y="10" WIDTH="500" HEIGHT="40"
                  CONF="0.6" PRED_CHAR_CNT="20"/>
        </PAGE></OCRDATASET>"""
        root = ET.fromstring(xml)
        # gap score = 150 / min(150, 200) = 1.0 -> split under the default threshold 0.5.
        image = self._make_line_image(
            (600, 200),
            line_box=(10, 10, 500, 40),
            segments=[(0, 150), (300, 500)],
        )

        replaced = split_wide_lines_at_whitespace(root, image, min_gap_height_ratio=0.15, gap_score_threshold=0.5)
        lines = list(root.iter("LINE"))

        self.assertEqual(replaced, 1)
        self.assertEqual(len(lines), 2)
        pred_char_counts_by_x = {int(line.get("X")): line.get("PRED_CHAR_CNT") for line in lines}
        for line in lines:
            self.assertEqual(line.get("CONF"), "0.6")
        xs = sorted(pred_char_counts_by_x)
        self.assertEqual(xs[0], 10)
        self.assertGreater(xs[1], 150)
        self.assertEqual(pred_char_counts_by_x[10], "6.000")
        self.assertEqual(pred_char_counts_by_x[xs[1]], "8.000")

    def test_keeps_gap_within_threshold(self) -> None:
        """Narrow gaps belong to one logical cell and must not split."""
        xml = """<OCRDATASET><PAGE>
            <LINE TYPE="本文" X="0" Y="0" WIDTH="400" HEIGHT="40" CONF="0.6"/>
        </PAGE></OCRDATASET>"""
        root = ET.fromstring(xml)
        # gap score = 80 / min(160, 160) = 0.5 -> strict '>' keeps the line intact.
        image = self._make_line_image(
            (500, 100),
            line_box=(0, 0, 400, 40),
            segments=[(0, 160), (240, 400)],
        )

        replaced = split_wide_lines_at_whitespace(root, image, min_gap_height_ratio=0.15, gap_score_threshold=0.5)

        self.assertEqual(replaced, 0)
        self.assertEqual(len(list(root.iter("LINE"))), 1)

    def test_higher_sensitivity_splits_borderline_gap(self) -> None:
        xml = """<OCRDATASET><PAGE>
            <LINE TYPE="本文" X="0" Y="0" WIDTH="380" HEIGHT="40" CONF="0.6"/>
        </PAGE></OCRDATASET>"""
        root_default = ET.fromstring(xml)
        root_sensitive = ET.fromstring(xml)
        image = self._make_line_image(
            (500, 100),
            line_box=(0, 0, 380, 40),
            segments=[(0, 160), (220, 380)],
        )

        replaced_default = split_wide_lines_at_whitespace(
            root_default,
            image,
            min_gap_height_ratio=0.15,
            gap_score_threshold=0.5,
        )
        replaced_sensitive = split_wide_lines_at_whitespace(
            root_sensitive,
            image,
            min_gap_height_ratio=0.075,
            gap_score_threshold=0.25,
        )

        self.assertEqual(replaced_default, 0)
        self.assertEqual(len(list(root_default.iter("LINE"))), 1)
        self.assertEqual(replaced_sensitive, 1)
        self.assertEqual(len(list(root_sensitive.iter("LINE"))), 2)

    def test_low_sensitivity_keeps_split_segments_when_gap_is_strong(self) -> None:
        xml = """<OCRDATASET><PAGE>
            <LINE TYPE="本文" X="0" Y="0" WIDTH="380" HEIGHT="40" CONF="0.6"/>
        </PAGE></OCRDATASET>"""
        root = ET.fromstring(xml)
        image = self._make_line_image(
            (480, 100),
            line_box=(0, 0, 380, 40),
            segments=[(0, 160), (320, 380)],
        )

        replaced = split_wide_lines_at_whitespace(root, image, min_gap_height_ratio=0.75, gap_score_threshold=2.5)

        self.assertEqual(replaced, 1)
        lines = list(root.iter("LINE"))
        self.assertEqual(len(lines), 2)
        widths = sorted(int(line.get("WIDTH")) for line in lines)
        self.assertEqual(widths, [60, 160])

    def test_merges_narrow_segment_instead_of_dropping_it(self) -> None:
        xml = """<OCRDATASET><PAGE>
            <LINE TYPE="本文" X="0" Y="0" WIDTH="580" HEIGHT="40" CONF="0.6"/>
        </PAGE></OCRDATASET>"""
        root = ET.fromstring(xml)
        image = self._make_line_image(
            (700, 120),
            line_box=(0, 0, 580, 40),
            segments=[(0, 150), (270, 278), (400, 580)],
        )

        replaced = split_wide_lines_at_whitespace(root, image, min_gap_height_ratio=0.15, gap_score_threshold=0.5)

        lines = list(root.iter("LINE"))
        self.assertEqual(replaced, 1)
        self.assertEqual(len(lines), 2)
        widths = sorted(int(line.get("WIDTH")) for line in lines)
        self.assertEqual(widths, [180, 278])

    def test_keeps_single_strip_when_no_gap(self) -> None:
        xml = """<OCRDATASET><PAGE>
            <LINE TYPE="本文" X="0" Y="0" WIDTH="400" HEIGHT="40" CONF="0.5"/>
        </PAGE></OCRDATASET>"""
        root = ET.fromstring(xml)
        # Continuous text across the full line.
        image = self._make_line_image(
            (500, 100),
            line_box=(0, 0, 400, 40),
            segments=[(0, 400)],
        )

        replaced = split_wide_lines_at_whitespace(root, image)

        self.assertEqual(replaced, 0)
        self.assertEqual(len(list(root.iter("LINE"))), 1)

    def test_skips_short_and_vertical_lines(self) -> None:
        xml = """<OCRDATASET><PAGE>
            <LINE TYPE="本文" X="0" Y="0" WIDTH="100" HEIGHT="200" CONF="0.5"/>
            <LINE TYPE="本文" X="0" Y="300" WIDTH="120" HEIGHT="40" CONF="0.5"/>
        </PAGE></OCRDATASET>"""
        root = ET.fromstring(xml)
        image = np.full((500, 500, 3), 255, dtype=np.uint8)

        replaced = split_wide_lines_at_whitespace(root, image)

        # Tall vertical line ignored; short horizontal line ignored.
        self.assertEqual(replaced, 0)
        self.assertEqual(len(list(root.iter("LINE"))), 2)


class FilterOversizedLinesTests(unittest.TestCase):
    def test_removes_lines_exceeding_height_ratio(self) -> None:
        xml = """<OCRDATASET><PAGE>
            <LINE TYPE="本文" X="0" Y="0" WIDTH="200" HEIGHT="20" CONF="0.8"/>
            <LINE TYPE="本文" X="0" Y="100" WIDTH="800" HEIGHT="400" CONF="0.3"/>
            <LINE TYPE="本文" X="0" Y="600" WIDTH="300" HEIGHT="30" CONF="0.6"/>
        </PAGE></OCRDATASET>"""
        root = ET.fromstring(xml)

        removed = filter_oversized_lines(root, img_height=1000, max_height_ratio=0.05)

        self.assertEqual(removed, 1)
        lines = list(root.iter("LINE"))
        self.assertEqual(len(lines), 2)
        heights = [int(line.get("HEIGHT")) for line in lines]
        self.assertEqual(sorted(heights), [20, 30])

    def test_keeps_all_when_none_oversized(self) -> None:
        xml = """<OCRDATASET><PAGE>
            <LINE TYPE="本文" X="0" Y="0" WIDTH="200" HEIGHT="20" CONF="0.8"/>
        </PAGE></OCRDATASET>"""
        root = ET.fromstring(xml)

        removed = filter_oversized_lines(root, img_height=1000)

        self.assertEqual(removed, 0)
        self.assertEqual(len(list(root.iter("LINE"))), 1)


class DeduplicateLinesTests(unittest.TestCase):
    def test_keeps_contained_line_when_geometry_is_not_similar(self) -> None:
        xml = """<OCRDATASET><PAGE>
            <LINE TYPE="本文" X="100" Y="100" WIDTH="200" HEIGHT="30" CONF="0.8"/>
            <LINE TYPE="本文" X="50" Y="80" WIDTH="400" HEIGHT="200" CONF="0.9"/>
        </PAGE></OCRDATASET>"""
        root = ET.fromstring(xml)

        removed = deduplicate_lines(root)

        self.assertEqual(removed, 0)
        lines = list(root.iter("LINE"))
        self.assertEqual(len(lines), 2)

    def test_removes_near_duplicate(self) -> None:
        xml = """<OCRDATASET><PAGE>
            <LINE TYPE="本文" X="100" Y="100" WIDTH="200" HEIGHT="30" CONF="0.8"/>
            <LINE TYPE="本文" X="102" Y="101" WIDTH="198" HEIGHT="29" CONF="0.5"/>
        </PAGE></OCRDATASET>"""
        root = ET.fromstring(xml)

        removed = deduplicate_lines(root)

        self.assertEqual(removed, 1)
        self.assertEqual(len(list(root.iter("LINE"))), 1)

    def test_keeps_nested_line_when_width_similarity_is_low(self) -> None:
        xml = """<OCRDATASET><PAGE>
            <LINE TYPE="本文" X="700" Y="2142" WIDTH="1142" HEIGHT="177" CONF="0.697"/>
            <LINE TYPE="本文" X="700" Y="2151" WIDTH="251" HEIGHT="157" CONF="0.266"/>
        </PAGE></OCRDATASET>"""
        root = ET.fromstring(xml)

        removed = deduplicate_lines(root)

        self.assertEqual(removed, 0)
        self.assertEqual(len(list(root.iter("LINE"))), 2)

    def test_keeps_non_overlapping_lines(self) -> None:
        xml = """<OCRDATASET><PAGE>
            <LINE TYPE="本文" X="0" Y="0" WIDTH="100" HEIGHT="20" CONF="0.8"/>
            <LINE TYPE="本文" X="500" Y="500" WIDTH="100" HEIGHT="20" CONF="0.6"/>
        </PAGE></OCRDATASET>"""
        root = ET.fromstring(xml)

        removed = deduplicate_lines(root)

        self.assertEqual(removed, 0)
        self.assertEqual(len(list(root.iter("LINE"))), 2)


class SuppressContainedFragmentsTests(unittest.TestCase):
    def test_suppresses_contained_substring_fragment(self) -> None:
        boxes = [
            TextBox(text="伝票番号", x=5104, y=1745, width=676, height=149, confidence=0.627, order=0),
            TextBox(text="号", x=5618, y=1747, width=163, height=144, confidence=0.264, order=1),
        ]

        filtered = suppress_contained_fragments(boxes)

        self.assertEqual([box.text for box in filtered], ["伝票番号"])

    def test_suppresses_partial_overlap_substring_fragment(self) -> None:
        boxes = [
            TextBox(text="単価", x=4665, y=2234, width=492, height=166, confidence=0.63, order=0),
            TextBox(text="価", x=5006, y=2240, width=334, height=154, confidence=0.295, order=1),
        ]

        filtered = suppress_contained_fragments(boxes)

        self.assertEqual([box.text for box in filtered], ["単価"])

    def test_suppresses_garbage_fragment_by_geometry_only(self) -> None:
        boxes = [
            TextBox(text="* I", x=1855, y=2236, width=310, height=142, confidence=0.5, order=0),
            TextBox(text="14", x=1998, y=2238, width=159, height=135, confidence=0.7, order=1),
        ]

        filtered = suppress_contained_fragments(boxes)

        self.assertEqual([box.text for box in filtered], ["* I"])

    def test_keeps_contained_box_when_text_is_not_substring(self) -> None:
        boxes = [
            TextBox(text="合計金額", x=100, y=100, width=400, height=120, confidence=0.7, order=0),
            TextBox(text="請求先", x=300, y=105, width=120, height=110, confidence=0.8, order=1),
        ]

        filtered = suppress_contained_fragments(boxes)

        self.assertEqual([box.text for box in filtered], ["合計金額", "請求先"])

    def test_keeps_large_row_and_nested_cell_when_width_similarity_is_too_low(self) -> None:
        boxes = [
            TextBox(
                text="No. 商品コード 品名 数量 単価 金額", x=100, y=100, width=3000, height=120, confidence=0.5, order=0
            ),
            TextBox(text="品名", x=1800, y=104, width=170, height=112, confidence=0.7, order=1),
        ]

        filtered = suppress_contained_fragments(boxes)

        self.assertEqual([box.text for box in filtered], ["No. 商品コード 品名 数量 単価 金額", "品名"])

    def test_keeps_contained_box_when_height_similarity_is_too_low(self) -> None:
        boxes = [
            TextBox(text="数量単価", x=100, y=100, width=500, height=120, confidence=0.4, order=0),
            TextBox(text="単価", x=350, y=100, width=180, height=60, confidence=0.6, order=1),
        ]

        filtered = suppress_contained_fragments(boxes, text_height_similarity_threshold=0.7)

        self.assertEqual([box.text for box in filtered], ["数量単価", "単価"])

    def test_preserves_original_order_of_remaining_boxes(self) -> None:
        boxes = [
            TextBox(text="A", x=0, y=0, width=100, height=40, confidence=0.9, order=0),
            TextBox(text="数量単価", x=100, y=100, width=500, height=120, confidence=0.4, order=1),
            TextBox(text="単価", x=350, y=102, width=180, height=110, confidence=0.6, order=2),
            TextBox(text="B", x=700, y=100, width=80, height=40, confidence=0.9, order=3),
        ]

        filtered = suppress_contained_fragments(boxes)

        self.assertEqual([box.text for box in filtered], ["A", "数量単価", "B"])


class RunDirectOCRTests(unittest.TestCase):
    def test_defaults_source_to_filename_with_extension(self) -> None:
        fake_engine = Mock()
        fake_engine.detector.detect.return_value = []
        fake_engine.classes = []

        with TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "sample.png"
            Image.new("RGB", (16, 16), "white").save(image_path)
            with (
                patch("ai_ocr_pipeline.ocr.direct._get_engine", return_value=fake_engine),
                patch("ndl_parser.convert_to_xml_string3", return_value="<PAGE></PAGE>"),
                patch("reading_order.xy_cut.eval.eval_xml"),
            ):
                result = run_direct_ocr(image_path)

        self.assertEqual(result.source, "sample.png")

    def test_reuses_detection_image_when_recognition_path_matches(self) -> None:
        fake_engine = Mock()
        fake_engine.detector.detect.return_value = []
        fake_engine.classes = []

        with TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "sample.png"
            Image.new("RGB", (16, 16), "white").save(image_path)
            with (
                patch("ai_ocr_pipeline.ocr.direct._get_engine", return_value=fake_engine),
                patch("ndl_parser.convert_to_xml_string3", return_value="<PAGE></PAGE>"),
                patch("reading_order.xy_cut.eval.eval_xml"),
                patch("ai_ocr_pipeline.ocr.direct.Image.open", wraps=Image.open) as image_open,
            ):
                run_direct_ocr(image_path, recognition_image_path=image_path)

        self.assertEqual(image_open.call_count, 1)

    def test_logs_and_falls_back_when_recognition_image_size_mismatches(self) -> None:
        fake_engine = Mock()
        fake_engine.detector.detect.return_value = []
        fake_engine.classes = []

        with TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "sample.png"
            alt_path = Path(tmpdir) / "sample_alt.png"
            Image.new("RGB", (16, 16), "white").save(image_path)
            Image.new("RGB", (12, 12), "white").save(alt_path)
            with (
                patch("ai_ocr_pipeline.ocr.direct._get_engine", return_value=fake_engine),
                patch("ndl_parser.convert_to_xml_string3", return_value="<PAGE></PAGE>"),
                patch("reading_order.xy_cut.eval.eval_xml"),
                patch("ai_ocr_pipeline.ocr.direct.logger.debug") as debug_log,
            ):
                run_direct_ocr(image_path, recognition_image_path=alt_path)

        self.assertTrue(debug_log.called)

    def test_clips_line_boxes_to_image_bounds_before_recognition(self) -> None:
        fake_engine = Mock()
        fake_engine.detector.detect.return_value = []
        fake_engine.classes = []
        seen_shapes: list[tuple[int, ...]] = []

        class _FakeRecogLine:
            def __init__(self, lineimg, idx, pred_char_cnt) -> None:
                seen_shapes.append(tuple(lineimg.shape))

        with TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "sample.png"
            Image.new("RGB", (10, 10), "white").save(image_path)
            with (
                patch("ai_ocr_pipeline.ocr.direct._get_engine", return_value=fake_engine),
                patch(
                    "ndl_parser.convert_to_xml_string3",
                    return_value=(
                        '<PAGE><LINE TYPE="本文" X="-3" Y="-2" WIDTH="8" HEIGHT="7" '
                        'CONF="0.8" PRED_CHAR_CNT="5"/></PAGE>'
                    ),
                ),
                patch("reading_order.xy_cut.eval.eval_xml"),
                patch("ocr.RecogLine", _FakeRecogLine),
                patch("ocr.process_cascade", return_value=["A"]),
            ):
                result = run_direct_ocr(image_path)

        self.assertEqual(seen_shapes, [(5, 5, 3)])
        self.assertEqual(result.boxes[0].width, 5)
        self.assertEqual(result.boxes[0].height, 5)
        self.assertEqual(result.boxes[0].center_x, 2.5)
        self.assertEqual(result.boxes[0].center_y, 2.5)

    def test_sets_raw_is_vertical_from_geometry(self) -> None:
        fake_engine = Mock()
        fake_engine.detector.detect.return_value = []
        fake_engine.classes = []

        with TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "sample.png"
            Image.new("RGB", (700, 500), "white").save(image_path)
            with (
                patch("ai_ocr_pipeline.ocr.direct._get_engine", return_value=fake_engine),
                patch(
                    "ndl_parser.convert_to_xml_string3",
                    return_value=(
                        "<PAGE>"
                        '<LINE TYPE="本文" X="10" Y="10" WIDTH="200" HEIGHT="20" CONF="0.8" PRED_CHAR_CNT="5"/>'
                        '<LINE TYPE="本文" X="10" Y="50" WIDTH="15" HEIGHT="24" CONF="0.7" PRED_CHAR_CNT="4"/>'
                        "</PAGE>"
                    ),
                ),
                patch("reading_order.xy_cut.eval.eval_xml"),
                patch("ocr.process_cascade", return_value=["A", "B"]),
            ):
                result = run_direct_ocr(image_path)

        self.assertFalse(result.boxes[0].is_vertical)
        self.assertTrue(result.boxes[1].is_vertical)

    def test_suppresses_contained_fragments_after_recognition(self) -> None:
        fake_engine = Mock()
        fake_engine.detector.detect.return_value = []
        fake_engine.classes = []

        with TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "sample.png"
            Image.new("RGB", (800, 3000), "white").save(image_path)
            with (
                patch("ai_ocr_pipeline.ocr.direct._get_engine", return_value=fake_engine),
                patch(
                    "ndl_parser.convert_to_xml_string3",
                    return_value=(
                        "<PAGE>"
                        '<LINE TYPE="本文" X="100" Y="100" WIDTH="400" HEIGHT="120" CONF="0.7" PRED_CHAR_CNT="8"/>'
                        '<LINE TYPE="本文" X="300" Y="105" WIDTH="120" HEIGHT="110" CONF="0.8" PRED_CHAR_CNT="2"/>'
                        "</PAGE>"
                    ),
                ),
                patch("reading_order.xy_cut.eval.eval_xml"),
                patch("ocr.process_cascade", return_value=["伝票番号", "号"]),
            ):
                result = run_direct_ocr(image_path)

        self.assertEqual([box.text for box in result.boxes], ["伝票番号"])


if __name__ == "__main__":
    unittest.main()
