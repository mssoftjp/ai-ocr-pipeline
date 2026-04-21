"""Direct in-process integration with ndlocr-lite.

Instead of invoking ``ndlocr-lite`` as a subprocess and parsing its JSON, this
module imports the detector / recognizer / XML-builder from the installed
``ndlocr-lite`` package and runs the full pipeline inline. This removes the
subprocess launch cost, keeps models cached across invocations, and — most
importantly — exposes the intermediate XML produced by
``ndl_parser.convert_to_xml_string3`` so that we can apply surgical fixes
before recognition.

One such fix is ``drop_container_fallback_lines`` (see below): when
``text_block`` / ``block_table`` / ``block_ad`` detections have no nested
``line_*`` children, ``convert_to_xml_string3`` emits the container bbox as
a single ``<LINE>`` and PARSEQ later reads that giant crop, producing
concatenated / hallucinated text. We remove those LINEs here.
"""

from __future__ import annotations

import logging
import threading
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from ai_ocr_pipeline.models import PageResult, TextBox

# Default DEIM thresholds mirror ``ndlocr-lite``'s CLI defaults.
DEFAULT_DET_SCORE_THRESHOLD = 0.2
DEFAULT_DET_CONF_THRESHOLD = 0.25
DEFAULT_DET_IOU_THRESHOLD = 0.2
NDL_NUM_CLASSES = 17


@dataclass(frozen=True)
class _EngineBundle:
    detector: object
    recognizer30: object
    recognizer50: object
    recognizer100: object
    classes: list[str]


_engine_cache: dict[str, _EngineBundle] = {}
_engine_lock = threading.Lock()
logger = logging.getLogger(__name__)

SPLIT_LEVEL_PARAMS: dict[int, tuple[float, float]] = {
    1: (1.50, 5.00),
    2: (1.00, 10.0 / 3.0),
    3: (0.60, 2.00),
    4: (0.30, 1.00),
    5: (0.15, 0.50),
}
DEFAULT_SPLIT_LEVEL = 2
DEFAULT_MIN_GAP_HEIGHT_RATIO, DEFAULT_GAP_SCORE_THRESHOLD = SPLIT_LEVEL_PARAMS[DEFAULT_SPLIT_LEVEL]


def _ndlocr_lite_root() -> Path:
    """Return the installed ndlocr-lite site-packages root (for weights)."""
    import ocr as ndlocr_main  # top-level ``ocr`` in ndlocr-lite

    return Path(ndlocr_main.__file__).resolve().parent


def split_params_for_level(level: int) -> tuple[float, float]:
    """Return ``(min_gap_height_ratio, gap_score_threshold)`` for a split level."""
    try:
        return SPLIT_LEVEL_PARAMS[level]
    except KeyError as exc:
        raise ValueError(f"Unsupported OCR split level: {level}") from exc


def split_params_for_legacy_sensitivity(sensitivity: float) -> tuple[float, float]:
    """Translate the legacy sensitivity float into direct split parameters."""
    return 0.15 / sensitivity, 0.5 / sensitivity


def _build_engine(device: str) -> _EngineBundle:
    """Instantiate DEIM + three PARSEQ recognizers with ndlocr-lite defaults."""
    from deim import DEIM
    from parseq import PARSEQ
    from yaml import safe_load

    root = _ndlocr_lite_root()
    det_weights = root / "model" / "deim-s-1024x1024.onnx"
    det_classes_path = root / "config" / "ndl.yaml"
    rec_classes_path = root / "config" / "NDLmoji.yaml"
    rec_weights_100 = root / "model" / "parseq-ndl-16x768-100-tiny-165epoch-tegaki2.onnx"
    rec_weights_30 = root / "model" / "parseq-ndl-16x256-30-tiny-192epoch-tegaki3.onnx"
    rec_weights_50 = root / "model" / "parseq-ndl-16x384-50-tiny-146epoch-tegaki2.onnx"

    for path in (
        det_weights,
        det_classes_path,
        rec_classes_path,
        rec_weights_100,
        rec_weights_30,
        rec_weights_50,
    ):
        if not path.exists():
            raise FileNotFoundError(f"ndlocr-lite asset not found: {path}")

    detector = DEIM(
        model_path=str(det_weights),
        class_mapping_path=str(det_classes_path),
        score_threshold=DEFAULT_DET_SCORE_THRESHOLD,
        conf_threshold=DEFAULT_DET_CONF_THRESHOLD,
        iou_threshold=DEFAULT_DET_IOU_THRESHOLD,
        device=device,
    )

    with open(rec_classes_path, encoding="utf-8") as f:
        charobj = safe_load(f)
    charlist = list(charobj["model"]["charset_train"])

    def _recognizer(path: Path) -> object:
        return PARSEQ(model_path=str(path), charlist=charlist, device=device)

    return _EngineBundle(
        detector=detector,
        recognizer30=_recognizer(rec_weights_30),
        recognizer50=_recognizer(rec_weights_50),
        recognizer100=_recognizer(rec_weights_100),
        classes=list(detector.classes.values()),
    )


def _get_engine(device: str) -> _EngineBundle:
    with _engine_lock:
        bundle = _engine_cache.get(device)
        if bundle is None:
            bundle = _build_engine(device)
            _engine_cache[device] = bundle
        return bundle


def drop_container_fallback_lines(root: ET.Element) -> int:
    """Remove ``<LINE>`` elements that have no ``CONF`` attribute.

    ``ndl_parser.convert_to_xml_string3`` emits such LINEs in three fallback
    paths (empty ``text_block`` / ``block_table`` / ``block_ad``) where the
    container bbox itself is promoted to a LINE because no nested
    ``line_*`` detection was present. Those LINEs invariably cause PARSEQ
    to read an oversized crop and return noisy or concatenated text.

    Legitimate LINEs coming from actual ``line_*`` detections carry a
    ``CONF`` attribute (written from ``line[4]`` in the detector output),
    so the absence of ``CONF`` is a reliable discriminator.

    Returns the number of LINEs removed so that callers can report it.
    """
    dropped = 0
    # Walk the tree and collect parent/child pairs first; modifying during
    # iteration would skip elements.
    victims: list[tuple[ET.Element, ET.Element]] = []
    for parent in root.iter():
        for child in list(parent):
            if child.tag == "LINE" and child.get("CONF") is None:
                victims.append((parent, child))
    for parent, child in victims:
        parent.remove(child)
        dropped += 1
    return dropped


def _find_gap_intervals(
    projection: np.ndarray,
    *,
    text_threshold: float,
    min_gap_px: int,
) -> list[tuple[int, int]]:
    """Return ``[(start, end), ...]`` gap intervals within ``projection``.

    ``projection`` is a 1D array of per-column text-density fractions in
    ``[0, 1]``. A column counts as "blank" when its density is
    ``<= text_threshold``. Consecutive blank columns of length
    ``>= min_gap_px`` form a gap.
    """
    if projection.size == 0 or min_gap_px <= 0:
        return []
    blank = projection <= text_threshold
    gaps: list[tuple[int, int]] = []
    run_start: int | None = None
    for i, is_blank in enumerate(blank):
        if is_blank and run_start is None:
            run_start = i
        elif not is_blank and run_start is not None:
            if i - run_start >= min_gap_px:
                gaps.append((run_start, i))
            run_start = None
    if run_start is not None and blank.size - run_start >= min_gap_px:
        gaps.append((run_start, int(blank.size)))
    return gaps


def _find_text_and_gap_runs(
    projection: np.ndarray,
    *,
    text_threshold: float,
    min_gap_px: int,
) -> list[tuple[int, int, bool]]:
    """Return ``[(start, end, is_text), ...]`` runs for a 1D density profile.

    Blank runs narrower than ``min_gap_px`` are promoted to text so that
    inter-character spacing does not become a split candidate.
    """
    if projection.size == 0:
        return []

    blank = projection <= text_threshold
    runs: list[tuple[int, int, bool]] = []
    run_start = 0
    current_is_text = not bool(blank[0])
    for idx in range(1, blank.size):
        is_text = not bool(blank[idx])
        if is_text == current_is_text:
            continue
        runs.append((run_start, idx, current_is_text))
        run_start = idx
        current_is_text = is_text
    runs.append((run_start, int(blank.size), current_is_text))

    promoted: list[tuple[int, int, bool]] = []
    for start, end, is_text in runs:
        if not is_text and end - start < min_gap_px:
            promoted.append((start, end, True))
        else:
            promoted.append((start, end, is_text))

    merged: list[tuple[int, int, bool]] = []
    for start, end, is_text in promoted:
        if not merged or merged[-1][2] != is_text:
            merged.append((start, end, is_text))
            continue
        prev_start, _prev_end, prev_is_text = merged[-1]
        merged[-1] = (prev_start, end, prev_is_text)
    return merged


def _score_and_filter_gaps(
    runs: list[tuple[int, int, bool]],
    *,
    threshold: float,
    min_text_run_px: int,
) -> list[tuple[int, int]]:
    """Keep only gap runs that are proportionally large relative to neighbors."""
    accepted: list[tuple[int, int]] = []
    for idx in range(1, len(runs) - 1):
        start, end, is_text = runs[idx]
        if is_text:
            continue
        left_start, left_end, left_is_text = runs[idx - 1]
        right_start, right_end, right_is_text = runs[idx + 1]
        if not left_is_text or not right_is_text:
            continue
        gap_width = end - start
        left_width = left_end - left_start
        right_width = right_end - right_start
        denom = max(min_text_run_px, min(left_width, right_width))
        if denom <= 0:
            continue
        if gap_width / denom > threshold:
            accepted.append((start, end))
    return accepted


def _compute_column_text_density(crop: np.ndarray) -> np.ndarray:
    """Return per-column fraction of text pixels in a line crop.

    Uses Otsu's threshold on the grayscale crop, treating the darker
    (lower-luminance) side as text. Works for both light-on-dark and
    dark-on-light inputs.
    """
    import cv2  # local import keeps the module cheap to import

    if crop.ndim == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    else:
        gray = crop
    if gray.size == 0:
        return np.zeros(0, dtype=np.float32)

    # Otsu decides the cut; we binarize so that text = 255, background = 0.
    _thr, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # If the majority of pixels are "white" post-threshold, text was dark —
    # invert so that text is always 255.
    white_fraction = float(np.count_nonzero(binary)) / float(binary.size)
    if white_fraction > 0.5:
        binary = 255 - binary

    text_pixels_per_column = (binary > 0).sum(axis=0).astype(np.float32)
    return text_pixels_per_column / float(max(1, binary.shape[0]))


def filter_oversized_lines(
    root: ET.Element,
    *,
    img_height: int,
    max_height_ratio: float = 0.05,
) -> int:
    """Remove ``<LINE>`` elements whose height exceeds a page-relative cap.

    After grid-line removal DEIM sometimes classifies block-level regions
    as ``line_*``, producing oversized LINEs that span multiple text rows.
    Recognising these generates noisy, concatenated text. This filter
    removes them so that the smaller, more precise detections are used.

    The default ratio of ``0.05`` corresponds to ~250 px at 600 DPI —
    well above the tallest normal printed line (~160 px) but below the
    smallest observed garbage detection (~340 px).
    """
    max_h = int(img_height * max_height_ratio)
    if max_h <= 0:
        return 0
    victims: list[tuple[ET.Element, ET.Element]] = []
    for parent in root.iter():
        for child in list(parent):
            if child.tag != "LINE":
                continue
            try:
                h = int(child.get("HEIGHT") or 0)
            except (TypeError, ValueError):
                continue
            if h > max_h:
                victims.append((parent, child))
    for parent, child in victims:
        parent.remove(child)
    return len(victims)


def deduplicate_lines(
    root: ET.Element,
    *,
    iou_threshold: float = 0.5,
    containment_threshold: float = 0.85,
    min_side_similarity: float = 0.6,
) -> int:
    """Remove redundant ``<LINE>`` elements via geometry-aware NMS.

    We only suppress boxes that are truly duplicate hypotheses for the
    same line: either their IoU is high, or one nearly contains the other
    *and* their width/height are still broadly similar. This avoids
    deleting a large valid line merely because a smaller spurious box sits
    inside it.
    """
    entries: list[tuple[ET.Element, ET.Element, int, int, int, int, float]] = []
    for parent in root.iter():
        for child in list(parent):
            if child.tag != "LINE":
                continue
            try:
                x = int(child.get("X") or 0)
                y = int(child.get("Y") or 0)
                w = int(child.get("WIDTH") or 0)
                h = int(child.get("HEIGHT") or 0)
                conf = float(child.get("CONF") or 0.0)
            except (TypeError, ValueError):
                continue
            if w > 0 and h > 0:
                entries.append((parent, child, x, y, w, h, conf))

    entries.sort(key=lambda e: (-e[6], -(e[4] * e[5]), e[3], e[2]))

    kept_boxes: list[tuple[int, int, int, int]] = []
    to_remove: list[tuple[ET.Element, ET.Element]] = []
    for parent, child, x, y, w, h, _conf in entries:
        suppressed = False
        area = w * h
        for kx, ky, kw, kh in kept_boxes:
            inter_x = max(0, min(x + w, kx + kw) - max(x, kx))
            inter_y = max(0, min(y + h, ky + kh) - max(y, ky))
            inter = inter_x * inter_y
            kept_area = kw * kh
            union = area + kept_area - inter
            smaller_area = min(area, kept_area)
            width_similarity = min(w, kw) / max(w, kw)
            height_similarity = min(h, kh) / max(h, kh)
            is_near_duplicate = union > 0 and inter / union >= iou_threshold
            is_similar_containment = (
                smaller_area > 0
                and inter / smaller_area >= containment_threshold
                and width_similarity >= min_side_similarity
                and height_similarity >= min_side_similarity
            )
            if is_near_duplicate or is_similar_containment:
                suppressed = True
                break
        if suppressed:
            to_remove.append((parent, child))
        else:
            kept_boxes.append((x, y, w, h))

    for parent, child in to_remove:
        parent.remove(child)
    return len(to_remove)


def suppress_contained_fragments(
    boxes: list[TextBox],
    *,
    text_containment_threshold: float = 0.3,
    text_height_similarity_threshold: float = 0.7,
    text_width_similarity_threshold: float = 0.2,
    geometric_containment_threshold: float = 0.95,
    geometric_height_similarity_threshold: float = 0.85,
    geometric_width_similarity_threshold: float = 0.4,
) -> list[TextBox]:
    """Remove smaller fragment boxes that duplicate a larger parent box.

    Tier 1 removes text-confirmed fragments where the child text is a
    substring of the parent text and the boxes overlap on the same line.
    Tier 2 removes near-fully-contained garbage fragments even when OCR text
    does not match, as long as the geometry is still strongly indicative of a
    nested duplicate rather than a distinct table cell.
    """
    if len(boxes) <= 1:
        return boxes

    indexed_boxes = list(enumerate(boxes))
    indexed_boxes.sort(
        key=lambda item: (
            -(item[1].width * item[1].height),
            -item[1].height,
            item[1].y,
            item[1].x,
        )
    )

    suppressed_indices: set[int] = set()
    for parent_idx, parent in indexed_boxes:
        if parent_idx in suppressed_indices:
            continue
        parent_text = parent.text.strip()
        if not parent_text:
            continue
        parent_x1 = parent.x
        parent_y1 = parent.y
        parent_x2 = parent.x + parent.width
        parent_y2 = parent.y + parent.height
        parent_area = parent.width * parent.height
        if parent_area <= 0:
            continue

        for child_idx, child in indexed_boxes:
            if child_idx == parent_idx or child_idx in suppressed_indices:
                continue
            child_area = child.width * child.height
            if child_area <= 0 or child_area >= parent_area:
                continue
            child_text = child.text.strip()

            child_x1 = child.x
            child_y1 = child.y
            child_x2 = child.x + child.width
            child_y2 = child.y + child.height
            inter_x = max(0.0, min(parent_x2, child_x2) - max(parent_x1, child_x1))
            inter_y = max(0.0, min(parent_y2, child_y2) - max(parent_y1, child_y1))
            inter = inter_x * inter_y
            containment = inter / child_area

            height_similarity = min(parent.height, child.height) / max(parent.height, child.height)
            width_similarity = min(parent.width, child.width) / max(parent.width, child.width)

            is_text_fragment = (
                bool(child_text)
                and child_text in parent_text
                and containment >= text_containment_threshold
                and height_similarity >= text_height_similarity_threshold
                and width_similarity >= text_width_similarity_threshold
            )
            is_geometric_fragment = (
                containment >= geometric_containment_threshold
                and height_similarity >= geometric_height_similarity_threshold
                and width_similarity >= geometric_width_similarity_threshold
            )
            if not is_text_fragment and not is_geometric_fragment:
                continue

            suppressed_indices.add(child_idx)

    return [box for idx, box in enumerate(boxes) if idx not in suppressed_indices]


def _clip_rect_to_image(
    x: int,
    y: int,
    width: int,
    height: int,
    *,
    img_width: int,
    img_height: int,
) -> tuple[int, int, int, int] | None:
    """Clip a rectangle to the image bounds."""
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(img_width, x + width)
    y2 = min(img_height, y + height)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2 - x1, y2 - y1


def _background_fill_value(crop: np.ndarray) -> np.ndarray:
    """Estimate a neutral background color from a crop's border pixels."""
    if crop.size == 0:
        return np.array([255, 255, 255], dtype=np.uint8)

    top = crop[0:1, :, :]
    bottom = crop[-1:, :, :]
    left = crop[:, 0:1, :]
    right = crop[:, -1:, :]
    border = np.concatenate(
        [
            top.reshape(-1, crop.shape[2]),
            bottom.reshape(-1, crop.shape[2]),
            left.reshape(-1, crop.shape[2]),
            right.reshape(-1, crop.shape[2]),
        ],
        axis=0,
    )
    return np.median(border, axis=0).astype(crop.dtype, copy=False)


def _extract_line_crop(
    image: np.ndarray,
    *,
    x: int,
    y: int,
    width: int,
    height: int,
    min_aspect_ratio: float = 1.0,
) -> np.ndarray:
    """Extract a recognition crop, widening only narrow boxes to avoid rotation.

    PARSEQ rotates crops when ``height > width``. For narrow horizontal boxes
    such as single-character table cells, that rotation destroys the glyph
    shape. We therefore expand the crop horizontally from the original image
    until it reaches the requested minimum aspect ratio, while leaving normal
    horizontal lines untouched.
    """
    img_h, img_w = image.shape[:2]
    clipped = _clip_rect_to_image(x, y, width, height, img_width=img_w, img_height=img_h)
    if clipped is None:
        return np.zeros((0, 0, image.shape[2]), dtype=image.dtype)

    x1, y1, clipped_w, clipped_h = clipped
    base_crop = image[y1 : y1 + clipped_h, x1 : x1 + clipped_w, :]
    if base_crop.size == 0:
        return base_crop

    target_w = max(clipped_w, round(clipped_h * min_aspect_ratio))
    if target_w <= clipped_w:
        return base_crop

    box_center = x1 + clipped_w / 2
    target_x1 = round(box_center - target_w / 2)
    source_x1 = max(0, target_x1)
    source_x2 = min(img_w, target_x1 + target_w)
    if source_x2 <= source_x1:
        return base_crop

    expanded = image[y1 : y1 + clipped_h, source_x1:source_x2, :]
    current_w = expanded.shape[1]
    if current_w >= target_w:
        return expanded

    background = _background_fill_value(base_crop)
    canvas = np.empty((clipped_h, target_w, image.shape[2]), dtype=image.dtype)
    canvas[...] = background
    dest_x1 = max(0, source_x1 - target_x1)
    dest_x2 = min(target_w, dest_x1 + current_w)
    canvas[:, dest_x1:dest_x2, :] = expanded[:, : dest_x2 - dest_x1, :]
    return canvas


def _recognition_image_for_box(
    detection_image: np.ndarray,
    recognition_image: np.ndarray,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    """Use alternate recognition pixels only for narrow boxes.

    This keeps normal horizontal lines on the exact raster that produced the
    geometry, while still letting narrow box crops avoid PARSEQ's forced
    rotation path.
    """
    if recognition_image is detection_image:
        return detection_image
    if width < height:
        return recognition_image
    return detection_image


def _clip_line_elements_to_image(root: ET.Element, *, img_width: int, img_height: int) -> list[ET.Element]:
    """Clamp ``<LINE>`` boxes to the image and drop zero-area elements."""
    clipped_nodes: list[ET.Element] = []
    to_remove: list[tuple[ET.Element, ET.Element]] = []
    for parent in root.iter():
        for child in list(parent):
            if child.tag != "LINE":
                continue
            try:
                x = int(child.get("X") or 0)
                y = int(child.get("Y") or 0)
                width = int(child.get("WIDTH") or 0)
                height = int(child.get("HEIGHT") or 0)
            except (TypeError, ValueError):
                to_remove.append((parent, child))
                continue

            clipped = _clip_rect_to_image(x, y, width, height, img_width=img_width, img_height=img_height)
            if clipped is None:
                to_remove.append((parent, child))
                continue

            clipped_x, clipped_y, clipped_width, clipped_height = clipped
            child.set("X", str(clipped_x))
            child.set("Y", str(clipped_y))
            child.set("WIDTH", str(clipped_width))
            child.set("HEIGHT", str(clipped_height))
            clipped_nodes.append(child)

    for parent, child in to_remove:
        parent.remove(child)

    return clipped_nodes


def _merge_narrow_segments(
    segment_bounds: list[tuple[int, int]],
    *,
    min_segment_width: int,
) -> list[tuple[int, int]]:
    """Merge undersized text segments into the nearest neighbour.

    Small edge fragments are often real content (for example, a short code in
    the first table column). Dropping them would lose OCRable pixels, so we
    absorb them into an adjacent segment instead.
    """
    if len(segment_bounds) <= 1:
        return segment_bounds

    merged = [[start, end] for start, end in segment_bounds]
    idx = 0
    while idx < len(merged):
        start, end = merged[idx]
        if end - start >= min_segment_width:
            idx += 1
            continue
        if len(merged) == 1:
            break
        if idx == 0:
            merged[1][0] = start
            del merged[0]
            continue
        if idx == len(merged) - 1:
            merged[idx - 1][1] = end
            del merged[idx]
            idx -= 1
            continue

        left_gap = start - merged[idx - 1][1]
        right_gap = merged[idx + 1][0] - end
        if left_gap <= right_gap:
            merged[idx - 1][1] = end
            del merged[idx]
            idx -= 1
        else:
            merged[idx + 1][0] = start
            del merged[idx]

        if idx < 0:
            idx = 0

    return [(start, end) for start, end in merged]


def split_wide_lines_at_whitespace(
    root: ET.Element,
    image: np.ndarray,
    *,
    min_aspect_ratio: float = 4.0,
    min_width_ratio: float = 0.05,
    min_gap_height_ratio: float = DEFAULT_MIN_GAP_HEIGHT_RATIO,
    gap_score_threshold: float = DEFAULT_GAP_SCORE_THRESHOLD,
    text_threshold: float = 0.02,
    min_segment_height_ratio: float = 0.25,
) -> int:
    """Split oversized horizontal ``<LINE>`` elements at whitespace gaps.

    DEIM frequently emits a single ``line_*`` detection spanning an entire
    table row (e.g. ``"1 ITEM-A01 ○○○○資材"``) when vertical column
    separators have been removed. Recognizing such a strip as one line
    either concatenates several cells into one token or drops cells
    entirely. This function inspects each wide LINE's crop, finds vertical
    whitespace wide enough to be a column gap, and replaces the LINE with
    one child LINE per segment.

    Args:
        root: Parsed XML (``OCRDATASET`` root) with ``<LINE>`` elements.
        image: The full-page RGB/gray ``np.ndarray`` used to extract crops.
        min_aspect_ratio: Only consider LINEs whose ``width / height`` is at
            least this value. Avoids splitting stacked vertical text while
            still allowing moderately wide table rows through.
        min_width_ratio: Only consider LINEs whose width is at least this
            fraction of the full page width. Avoids splitting short phrases
            that happen to be wide.
        min_gap_height_ratio: Minimum blank-run width, relative to line
            height, to treat as a candidate gap.
        gap_score_threshold: Minimum gap score required to accept a split.
        text_threshold: A column counts as blank if the fraction of text
            pixels in it is ``<=`` this value.
        min_segment_height_ratio: Post-split, each segment must be at least
            ``height * this_ratio`` wide. Narrower slivers are merged back
            into the nearest non-sliver neighbour. This quality gate is kept
            independent from ``sensitivity`` so that the user-controlled
            split knob only affects gap detection.

    Returns the number of LINEs that were replaced with >=2 segments.
    """
    img_h, img_w = image.shape[:2]
    if img_w <= 0 or img_h <= 0:
        return 0

    replaced = 0
    # Collect victims first; we'll mutate the tree after iteration.
    to_process: list[tuple[ET.Element, ET.Element]] = []
    for parent in root.iter():
        for child in list(parent):
            if child.tag != "LINE":
                continue
            try:
                line_w = int(child.get("WIDTH") or 0)
                line_h = int(child.get("HEIGHT") or 0)
            except (TypeError, ValueError):
                continue
            if line_w <= 0 or line_h <= 0:
                continue
            if line_w / line_h < min_aspect_ratio:
                continue
            if line_w / img_w < min_width_ratio:
                continue
            to_process.append((parent, child))

    for parent, child in to_process:
        x = int(child.get("X") or 0)
        y = int(child.get("Y") or 0)
        w = int(child.get("WIDTH") or 0)
        h = int(child.get("HEIGHT") or 0)

        clipped = _clip_rect_to_image(x, y, w, h, img_width=img_w, img_height=img_h)
        if clipped is None:
            continue
        x1, y1, clipped_w, clipped_h = clipped

        crop = image[y1 : y1 + clipped_h, x1 : x1 + clipped_w]
        density = _compute_column_text_density(crop)
        if density.size == 0:
            continue

        noise_floor = max(3, int(clipped_h * min_gap_height_ratio))
        split_threshold = gap_score_threshold
        min_seg_px = max(1, int(clipped_h * min_segment_height_ratio))
        min_text_run_px = max(3, int(clipped_h * 0.3))

        runs = _find_text_and_gap_runs(
            density,
            text_threshold=text_threshold,
            min_gap_px=noise_floor,
        )
        gaps = _score_and_filter_gaps(
            runs,
            threshold=split_threshold,
            min_text_run_px=min_text_run_px,
        )
        if not gaps:
            continue

        # Convert gap intervals -> segment intervals (inclusive of content).
        segment_bounds: list[tuple[int, int]] = []
        cursor = 0
        for gs, ge in gaps:
            if gs > cursor:
                segment_bounds.append((cursor, gs))
            cursor = ge
        if cursor < density.size:
            segment_bounds.append((cursor, int(density.size)))

        kept_bounds = _merge_narrow_segments(segment_bounds, min_segment_width=min_seg_px)
        if len(kept_bounds) < 2:
            # Once narrow slivers are merged back, there is no meaningful split.
            continue

        # Build replacement LINEs. Inherit attrs from the original.
        attrs = dict(child.attrib)
        try:
            original_pred_char_cnt = float(attrs.get("PRED_CHAR_CNT") or 100.0)
        except (TypeError, ValueError):
            original_pred_char_cnt = 100.0
        original_index = list(parent).index(child)
        parent.remove(child)
        for offset, (seg_start, seg_end) in enumerate(kept_bounds):
            seg_x = x1 + int(seg_start)
            seg_w = int(seg_end - seg_start)
            scaled_pred_char_cnt = original_pred_char_cnt * (seg_w / max(1, clipped_w))
            new_elem = ET.Element("LINE", attrib=attrs.copy())
            new_elem.set("X", str(seg_x))
            new_elem.set("Y", str(y1))
            new_elem.set("WIDTH", str(seg_w))
            new_elem.set("HEIGHT", str(clipped_h))
            new_elem.set("PRED_CHAR_CNT", f"{scaled_pred_char_cnt:0.3f}")
            parent.insert(original_index + offset, new_elem)
        replaced += 1

    return replaced


def _detections_to_resultobj(detections: list[dict], num_classes: int = NDL_NUM_CLASSES) -> list[dict]:
    """Replicate ``ocr.process``'s per-class bucketing of detections."""
    resultobj: list[dict] = [dict(), dict()]
    resultobj[0][0] = []
    for i in range(num_classes):
        resultobj[1][i] = []
    for det in detections:
        xmin, ymin, xmax, ymax = det["box"]
        conf = det["confidence"]
        char_count = det["pred_char_count"]
        if det["class_index"] == 0:
            resultobj[0][0].append([xmin, ymin, xmax, ymax])
        resultobj[1][det["class_index"]].append([xmin, ymin, xmax, ymax, conf, char_count])
    return resultobj


def _promote_detections_to_lines(
    root: ET.Element,
    detection_image: np.ndarray,
    detections: list[dict],
    *,
    recognition_image: np.ndarray | None = None,
) -> list:
    """Fallback when layout produced no LINEs: treat raw detections as lines.

    Mirrors the branch in ``ocr.process`` at lines 224-246.
    """
    from ocr import RecogLine  # type: ignore[import-not-found]

    page = root.find("PAGE")
    if page is None:
        return []

    lines: list = []
    for idx, det in enumerate(detections):
        xmin, ymin, xmax, ymax = det["box"]
        clipped = _clip_rect_to_image(
            int(xmin),
            int(ymin),
            int(xmax - xmin),
            int(ymax - ymin),
            img_width=detection_image.shape[1],
            img_height=detection_image.shape[0],
        )
        if clipped is None:
            continue
        clipped_x, clipped_y, line_w, line_h = clipped
        line_elem = ET.SubElement(page, "LINE")
        line_elem.set("TYPE", "本文")
        line_elem.set("X", str(clipped_x))
        line_elem.set("Y", str(clipped_y))
        line_elem.set("WIDTH", str(line_w))
        line_elem.set("HEIGHT", str(line_h))
        line_elem.set("CONF", f"{det['confidence']:0.3f}")
        pred_char_cnt = det.get("pred_char_count", 100.0)
        line_elem.set("PRED_CHAR_CNT", f"{pred_char_cnt:0.3f}")
        recognition_base = _recognition_image_for_box(
            detection_image,
            recognition_image if recognition_image is not None else detection_image,
            width=line_w,
            height=line_h,
        )
        lineimg = _extract_line_crop(
            recognition_base,
            x=clipped_x,
            y=clipped_y,
            width=line_w,
            height=line_h,
        )
        lines.append(RecogLine(lineimg, idx, pred_char_cnt))
    return lines


def run_direct_ocr(
    image_path: Path,
    *,
    recognition_image_path: Path | None = None,
    source: str = "",
    page: int | None = None,
    device: str = "cpu",
    filter_container_fallbacks: bool = True,
    split_wide_lines: bool = True,
    split_min_gap_height_ratio: float = DEFAULT_MIN_GAP_HEIGHT_RATIO,
    split_gap_score_threshold: float = DEFAULT_GAP_SCORE_THRESHOLD,
    filter_oversized: bool = True,
    deduplicate: bool = True,
    xml_hook: Callable[[ET.Element], None] | None = None,
) -> PageResult:
    """Run the full ndlocr-lite pipeline in-process and return a PageResult."""
    from ndl_parser import convert_to_xml_string3  # type: ignore[import-not-found]
    from ocr import RecogLine, process_cascade  # type: ignore[import-not-found]
    from reading_order.xy_cut.eval import (  # type: ignore[import-not-found]
        eval_xml,
    )

    engine = _get_engine(device)

    with Image.open(image_path) as pil:
        if pil.mode != "RGB":
            pil = pil.convert("RGB")
        img = np.array(pil)

    if recognition_image_path is None or recognition_image_path == image_path:
        recognition_img = img
    else:
        with Image.open(recognition_image_path) as pil:
            if pil.mode != "RGB":
                pil = pil.convert("RGB")
            recognition_img = np.array(pil)

    img_h, img_w = img.shape[:2]
    rec_h, rec_w = recognition_img.shape[:2]
    if (rec_w, rec_h) != (img_w, img_h):
        logger.debug(
            "Recognition image size %sx%s does not match detection image %sx%s; falling back to detection image.",
            rec_w,
            rec_h,
            img_w,
            img_h,
        )
        recognition_img = img

    detections = engine.detector.detect(img)
    resultobj = _detections_to_resultobj(detections)

    img_name = Path(source).name if source else Path(image_path).name
    xmlstr = convert_to_xml_string3(img_w, img_h, img_name, engine.classes, resultobj)
    xmlstr = "<OCRDATASET>" + xmlstr + "</OCRDATASET>"
    root = ET.fromstring(xmlstr)

    if filter_container_fallbacks:
        drop_container_fallback_lines(root)

    if split_wide_lines:
        split_wide_lines_at_whitespace(
            root,
            img,
            min_gap_height_ratio=split_min_gap_height_ratio,
            gap_score_threshold=split_gap_score_threshold,
        )

    if filter_oversized:
        filter_oversized_lines(root, img_height=img_h)

    if deduplicate:
        deduplicate_lines(root)

    if xml_hook is not None:
        xml_hook(root)

    eval_xml(root, logger=None)

    line_nodes = _clip_line_elements_to_image(root, img_width=img_w, img_height=img_h)
    alllineobj: list = []
    for idx, lineobj in enumerate(line_nodes):
        xmin = int(lineobj.get("X", 0))
        ymin = int(lineobj.get("Y", 0))
        line_w = int(lineobj.get("WIDTH", 0))
        line_h = int(lineobj.get("HEIGHT", 0))
        try:
            pred_char_cnt = float(lineobj.get("PRED_CHAR_CNT") or 100.0)
        except (TypeError, ValueError):
            pred_char_cnt = 100.0
        assert line_w > 0 and line_h > 0, f"LINE with zero dims passed clipping: {lineobj.attrib}"
        lineimg = _extract_line_crop(
            _recognition_image_for_box(
                img,
                recognition_img,
                width=line_w,
                height=line_h,
            ),
            x=xmin,
            y=ymin,
            width=line_w,
            height=line_h,
        )
        alllineobj.append(RecogLine(lineimg, idx, pred_char_cnt))

    # Layout produced nothing: retry with raw detections as lines (matches
    # the safety net in ocr.process).
    if not alllineobj and detections:
        alllineobj = _promote_detections_to_lines(root, img, detections, recognition_image=recognition_img)
        line_nodes = list(root.findall(".//LINE"))

    resultlinesall = (
        process_cascade(
            alllineobj,
            engine.recognizer30,
            engine.recognizer50,
            engine.recognizer100,
            is_cascade=True,
        )
        if alllineobj
        else []
    )

    boxes: list[TextBox] = []
    for idx, lineobj in enumerate(line_nodes):
        x = int(lineobj.get("X", 0))
        y = int(lineobj.get("Y", 0))
        w = int(lineobj.get("WIDTH", 0))
        h = int(lineobj.get("HEIGHT", 0))
        try:
            conf = float(lineobj.get("CONF") or 0.0)
        except (TypeError, ValueError):
            conf = 0.0
        text = resultlinesall[idx] if idx < len(resultlinesall) else ""
        boxes.append(
            TextBox(
                text=text,
                width=w,
                height=h,
                x=x,
                y=y,
                confidence=conf,
                order=idx,
                # The upstream CLI emits ``isVertical=true`` for every line,
                # but our downstream consumers expect the raw flag to be
                # informative for ambiguous boxes.
                is_vertical=h > w,
                text_source="ocr",
            )
        )

    boxes = suppress_contained_fragments(boxes)

    return PageResult(
        source=source or Path(image_path).name,
        page=page,
        img_width=img_w,
        img_height=img_h,
        boxes=boxes,
    )
