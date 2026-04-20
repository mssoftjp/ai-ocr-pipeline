"""Template loading and box generation for template-driven OCR."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TypedDict

from ai_ocr_pipeline.models import PageResult, TextBox

NEWLINE_HANDLING_MODES = {"first_line", "join", "preserve"}


class TemplatePromptContext(TypedDict, total=False):
    """Prompt context for one template box."""

    label: str
    hint: str


@dataclass(frozen=True)
class TemplateBox:
    id: int
    label: str
    x: float
    y: float
    width: float
    height: float
    is_vertical: bool | None = None
    hint: str | None = None


@dataclass(frozen=True)
class Template:
    name: str
    version: int
    coordinate_mode: str
    boxes: list[TemplateBox]
    description: str | None = None
    reference_width: int | None = None
    reference_height: int | None = None
    default_is_vertical: bool | None = None
    preprocess_deskew: bool | None = None
    preprocess_remove_horizontal_lines: bool | None = None
    preprocess_remove_vertical_lines: bool | None = None
    preprocess_newline_handling: str | None = None


def _expect_mapping(value: object, *, field_name: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object.")
    return value


def _expect_bool(value: object, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean.")
    return value


def _expect_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer.")
    return value


def _expect_number(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a number.")
    return float(value)


def _expect_string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    return value


def _expect_string_allow_empty(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string.")
    return value


def load_template(path: Path) -> Template:
    """Load and validate a template JSON file."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"Could not read template file: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Template file is not valid JSON: {exc}") from exc

    root = _expect_mapping(payload, field_name="root")
    template_data = _expect_mapping(root.get("template"), field_name="template")
    name = _expect_string(template_data.get("name"), field_name="template.name")
    version = _expect_int(template_data.get("version"), field_name="template.version")
    if version != 1:
        raise ValueError(f"Unsupported template.version: {version}. Expected 1.")

    coordinate_mode = _expect_string(
        template_data.get("coordinate_mode"),
        field_name="template.coordinate_mode",
    )
    if coordinate_mode not in {"ratio", "pixel"}:
        raise ValueError("template.coordinate_mode must be 'ratio' or 'pixel'.")

    description = template_data.get("description")
    if description is not None and not isinstance(description, str):
        raise ValueError("template.description must be a string when provided.")

    defaults_data = root.get("defaults")
    default_is_vertical: bool | None = None
    if defaults_data is not None:
        defaults_mapping = _expect_mapping(defaults_data, field_name="defaults")
        if "is_vertical" in defaults_mapping:
            default_is_vertical = _expect_bool(
                defaults_mapping["is_vertical"],
                field_name="defaults.is_vertical",
            )

    preprocess_data = root.get("preprocess")
    preprocess_deskew: bool | None = None
    preprocess_remove_horizontal_lines: bool | None = None
    preprocess_remove_vertical_lines: bool | None = None
    preprocess_newline_handling: str | None = None
    if preprocess_data is not None:
        preprocess_mapping = _expect_mapping(preprocess_data, field_name="preprocess")
        if "deskew" in preprocess_mapping:
            preprocess_deskew = _expect_bool(
                preprocess_mapping["deskew"],
                field_name="preprocess.deskew",
            )
        if "remove_horizontal_lines" in preprocess_mapping:
            preprocess_remove_horizontal_lines = _expect_bool(
                preprocess_mapping["remove_horizontal_lines"],
                field_name="preprocess.remove_horizontal_lines",
            )
        if "remove_vertical_lines" in preprocess_mapping:
            preprocess_remove_vertical_lines = _expect_bool(
                preprocess_mapping["remove_vertical_lines"],
                field_name="preprocess.remove_vertical_lines",
            )
        if "newline_handling" in preprocess_mapping:
            preprocess_newline_handling = _expect_string(
                preprocess_mapping["newline_handling"],
                field_name="preprocess.newline_handling",
            )
            if preprocess_newline_handling not in NEWLINE_HANDLING_MODES:
                raise ValueError(
                    f"preprocess.newline_handling must be one of {', '.join(sorted(NEWLINE_HANDLING_MODES))}."
                )

    reference_width: int | None = None
    reference_height: int | None = None
    reference_size = template_data.get("reference_size")
    if reference_size is not None:
        reference_mapping = _expect_mapping(reference_size, field_name="template.reference_size")
        reference_width = _expect_int(reference_mapping.get("width"), field_name="template.reference_size.width")
        reference_height = _expect_int(reference_mapping.get("height"), field_name="template.reference_size.height")
        if reference_width <= 0 or reference_height <= 0:
            raise ValueError("template.reference_size.width/height must be positive integers.")
    elif coordinate_mode == "pixel":
        raise ValueError("template.reference_size is required when coordinate_mode='pixel'.")

    raw_boxes = root.get("boxes")
    if not isinstance(raw_boxes, list) or not raw_boxes:
        raise ValueError("boxes must be a non-empty array.")

    boxes: list[TemplateBox] = []
    seen_ids: set[int] = set()
    for index, raw_box in enumerate(raw_boxes):
        field_prefix = f"boxes[{index}]"
        box_data = _expect_mapping(raw_box, field_name=field_prefix)
        box_id = _expect_int(box_data.get("id"), field_name=f"{field_prefix}.id")
        if box_id in seen_ids:
            raise ValueError(f"Duplicate template box id: {box_id}.")
        seen_ids.add(box_id)

        label = _expect_string_allow_empty(box_data.get("label", ""), field_name=f"{field_prefix}.label")
        x = _expect_number(box_data.get("x"), field_name=f"{field_prefix}.x")
        y = _expect_number(box_data.get("y"), field_name=f"{field_prefix}.y")
        width = _expect_number(box_data.get("width"), field_name=f"{field_prefix}.width")
        height = _expect_number(box_data.get("height"), field_name=f"{field_prefix}.height")
        if width <= 0 or height <= 0:
            raise ValueError(f"{field_prefix}.width and {field_prefix}.height must be positive.")

        if coordinate_mode == "ratio":
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                raise ValueError(f"{field_prefix}.x and {field_prefix}.y must be within 0.0-1.0.")
            if x + width > 1.0 or y + height > 1.0:
                raise ValueError(f"{field_prefix} exceeds ratio bounds.")
        else:
            assert reference_width is not None and reference_height is not None
            if x < 0 or y < 0:
                raise ValueError(f"{field_prefix}.x and {field_prefix}.y must be non-negative.")
            if x + width > reference_width or y + height > reference_height:
                raise ValueError(f"{field_prefix} exceeds template.reference_size bounds.")

        raw_is_vertical = box_data.get("is_vertical")
        is_vertical = default_is_vertical
        if raw_is_vertical is not None:
            is_vertical = _expect_bool(raw_is_vertical, field_name=f"{field_prefix}.is_vertical")

        raw_hint = box_data.get("hint")
        if raw_hint is not None and not isinstance(raw_hint, str):
            raise ValueError(f"{field_prefix}.hint must be a string when provided.")

        boxes.append(
            TemplateBox(
                id=box_id,
                label=label,
                x=x,
                y=y,
                width=width,
                height=height,
                is_vertical=is_vertical,
                hint=raw_hint,
            )
        )

    return Template(
        name=name,
        version=version,
        coordinate_mode=coordinate_mode,
        boxes=boxes,
        description=description,
        reference_width=reference_width,
        reference_height=reference_height,
        default_is_vertical=default_is_vertical,
        preprocess_deskew=preprocess_deskew,
        preprocess_remove_horizontal_lines=preprocess_remove_horizontal_lines,
        preprocess_remove_vertical_lines=preprocess_remove_vertical_lines,
        preprocess_newline_handling=preprocess_newline_handling,
    )


def _select_boxes(template: Template, box_ids: tuple[int, ...] | None) -> list[TemplateBox]:
    boxes_by_id = {box.id: box for box in template.boxes}
    if box_ids is not None:
        requested_ids = set(box_ids)
        unknown_ids = sorted(requested_ids - boxes_by_id.keys())
        if unknown_ids:
            joined = ", ".join(str(box_id) for box_id in unknown_ids)
            raise ValueError(f"Unknown template box id(s): {joined}.")
        selected = [boxes_by_id[box_id] for box_id in sorted(requested_ids)]
    else:
        selected = sorted(template.boxes, key=lambda box: box.id)
    return selected


def template_to_page_result(
    template: Template,
    img_width: int,
    img_height: int,
    source: str,
    page: int | None,
    box_ids: tuple[int, ...] | None = None,
) -> PageResult:
    """Convert a template definition into a PageResult for one image."""
    boxes: list[TextBox] = []
    for box in _select_boxes(template, box_ids):
        if template.coordinate_mode == "ratio":
            px_x = box.x * img_width
            px_y = box.y * img_height
            px_width = box.width * img_width
            px_height = box.height * img_height
        else:
            assert template.reference_width is not None and template.reference_height is not None
            scale_x = img_width / template.reference_width
            scale_y = img_height / template.reference_height
            px_x = box.x * scale_x
            px_y = box.y * scale_y
            px_width = box.width * scale_x
            px_height = box.height * scale_y

        width = max(1, round(px_width))
        height = max(1, round(px_height))
        boxes.append(
            TextBox(
                text="",
                width=width,
                height=height,
                x=px_x,
                y=px_y,
                confidence=0.0,
                order=box.id,
                type=box.label,
                is_vertical=box.is_vertical,
                text_source="template",
                box_source="template",
            )
        )

    return PageResult(
        source=source,
        page=page,
        img_width=img_width,
        img_height=img_height,
        boxes=boxes,
    )


def build_template_prompt_contexts(
    template: Template,
    box_ids: tuple[int, ...] | None = None,
) -> dict[int, TemplatePromptContext]:
    """Build per-box prompt context keyed by PageResult box index."""
    contexts: dict[int, TemplatePromptContext] = {}
    for index, box in enumerate(_select_boxes(template, box_ids)):
        context: TemplatePromptContext = {}
        if box.label.strip():
            context["label"] = box.label
        if box.hint:
            context["hint"] = box.hint
        contexts[index] = context
    return contexts


def _reading_order_key(box: TextBox) -> tuple[int, int, float, float]:
    order = box.order if box.order is not None else 10**9
    return (0 if box.order is not None else 1, order, box.center_y, box.center_x)


_SUBSTANTIVE_TEXT_RE = re.compile(r"[0-9A-Za-z\u3040-\u30ff\u4e00-\u9fff¥￥$%]")
_OBVIOUS_NOISE_RE = re.compile(r"[-_=~.·・ー一|/\\]+")


def _has_substantive_ocr_text(text: str) -> bool:
    return bool(_SUBSTANTIVE_TEXT_RE.search(text))


def _normalize_ocr_candidate(text: str) -> str:
    return " ".join(text.split())


def _is_obvious_ocr_noise(text: str) -> bool:
    compact = re.sub(r"\s+", "", text)
    if not compact or _has_substantive_ocr_text(compact):
        return False
    return len(compact) >= 3 and bool(_OBVIOUS_NOISE_RE.fullmatch(compact))


def build_ocr_evidence(
    page_result: PageResult,
    ocr_results_by_index: dict[int, PageResult],
    *,
    low_ink_by_index: dict[int, bool] | None = None,
) -> PageResult:
    """Attach per-box OCR evidence to the target boxes in *page_result*."""
    boxes: list[TextBox] = []
    for index, box in enumerate(page_result.boxes):
        ocr_result = ocr_results_by_index.get(
            index,
            PageResult(
                source=page_result.source, page=page_result.page, img_width=box.width, img_height=box.height, boxes=[]
            ),
        )
        matched = sorted(ocr_result.boxes, key=_reading_order_key)
        seed_parts = [candidate.text.strip() for candidate in matched if candidate.text.strip()]
        seed_text = " ".join(seed_parts)
        seed_confidence = min((candidate.confidence for candidate in matched), default=None)
        substantive_candidates = [
            (candidate.text.strip(), candidate.confidence)
            for candidate in matched
            if candidate.text.strip() and _has_substantive_ocr_text(candidate.text.strip())
        ]
        consensus_text: str | None = None
        consensus_confidence: float | None = None
        if substantive_candidates:
            grouped: dict[str, list[tuple[str, float]]] = {}
            for text, confidence in substantive_candidates:
                grouped.setdefault(_normalize_ocr_candidate(text), []).append((text, confidence))
            if len(grouped) == 1:
                normalized = next(iter(grouped))
                entries = grouped[normalized]
                best_text, best_confidence = max(entries, key=lambda entry: entry[1])
                consensus_text = best_text
                consensus_confidence = best_confidence
        boxes.append(
            replace(
                box,
                text=seed_text,
                confidence=seed_confidence if seed_confidence is not None else 0.0,
                ocr_seed_text=seed_text,
                ocr_seed_confidence=seed_confidence,
                ocr_match_count=len(matched),
                ocr_consensus_text=consensus_text,
                ocr_consensus_confidence=consensus_confidence,
                low_ink=(low_ink_by_index or {}).get(index),
            )
        )

    return replace(page_result, boxes=boxes)


def decide_target_box_actions(
    page_result: PageResult,
    *,
    confidence_threshold: float | None = None,
) -> tuple[PageResult, tuple[int, ...]]:
    """Choose blank-skip / OCR / AI for each target box."""
    boxes: list[TextBox] = []
    refine_indices: list[int] = []
    for index, box in enumerate(page_result.boxes):
        match_count = box.ocr_match_count or 0
        seed_text = box.ocr_seed_text or ""
        consensus_text = box.ocr_consensus_text or ""
        consensus_confidence = box.ocr_consensus_confidence
        low_ink = bool(box.low_ink)
        if low_ink and (match_count == 0 or _is_obvious_ocr_noise(seed_text)):
            boxes.append(
                replace(
                    box,
                    text="",
                    confidence=0.0,
                    text_source="blank_skip",
                    decision="blank_skip",
                )
            )
            continue

        if (
            confidence_threshold is not None
            and consensus_text.strip()
            and consensus_confidence is not None
            and consensus_confidence >= confidence_threshold
        ):
            boxes.append(
                replace(
                    box,
                    text=consensus_text,
                    confidence=consensus_confidence,
                    text_source="ocr",
                    decision="ocr",
                )
            )
            continue

        refine_indices.append(index)
        boxes.append(replace(box, decision="ai"))

    return replace(page_result, boxes=boxes), tuple(refine_indices)
