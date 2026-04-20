"""Data models for OCR pipeline output."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass


def _round_float(value: float, digits: int) -> float:
    """Round float output to a stable, human-friendly precision."""
    return round(value, digits)


@dataclass(frozen=True)
class PageSerializationOptions:
    """Controls how PageResult objects are serialized to JSON-friendly dicts."""

    include_absolute_geometry: bool = True
    include_debug_fields: bool = True


@dataclass(frozen=True, init=False)
class TextBox:
    """A single detected text box with content and spatial information."""

    text: str
    width: int
    height: int
    x: float
    y: float
    confidence: float
    order: int | None = None
    type: str | None = None
    is_vertical: bool | None = None
    text_source: str | None = None
    box_source: str | None = None
    decision: str | None = None
    ocr_seed_text: str | None = None
    ocr_seed_confidence: float | None = None
    ocr_match_count: int | None = None
    ocr_consensus_text: str | None = None
    ocr_consensus_confidence: float | None = None
    low_ink: bool | None = None

    def __init__(
        self,
        *,
        text: str,
        width: int,
        height: int,
        confidence: float,
        x: float | None = None,
        y: float | None = None,
        center_x: float | None = None,
        center_y: float | None = None,
        order: int | None = None,
        type: str | None = None,
        is_vertical: bool | None = None,
        text_source: str | None = None,
        box_source: str | None = None,
        decision: str | None = None,
        ocr_seed_text: str | None = None,
        ocr_seed_confidence: float | None = None,
        ocr_match_count: int | None = None,
        ocr_consensus_text: str | None = None,
        ocr_consensus_confidence: float | None = None,
        low_ink: bool | None = None,
    ) -> None:
        if x is None:
            if center_x is None:
                raise TypeError("TextBox requires either x or center_x.")
            x = center_x - width / 2
        if y is None:
            if center_y is None:
                raise TypeError("TextBox requires either y or center_y.")
            y = center_y - height / 2

        object.__setattr__(self, "text", text)
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "height", height)
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "order", order)
        object.__setattr__(self, "type", type)
        object.__setattr__(self, "is_vertical", is_vertical)
        object.__setattr__(self, "text_source", text_source)
        object.__setattr__(self, "box_source", box_source)
        object.__setattr__(self, "decision", decision)
        object.__setattr__(self, "ocr_seed_text", ocr_seed_text)
        object.__setattr__(self, "ocr_seed_confidence", ocr_seed_confidence)
        object.__setattr__(self, "ocr_match_count", ocr_match_count)
        object.__setattr__(self, "ocr_consensus_text", ocr_consensus_text)
        object.__setattr__(self, "ocr_consensus_confidence", ocr_consensus_confidence)
        object.__setattr__(self, "low_ink", low_ink)

    @property
    def center_x(self) -> float:
        return self.x + self.width / 2

    @property
    def center_y(self) -> float:
        return self.y + self.height / 2

    def to_dict(self) -> dict:
        data = {k: v for k, v in asdict(self).items() if v is not None}
        data["x"] = _round_float(self.x, 1)
        data["y"] = _round_float(self.y, 1)
        data["confidence"] = _round_float(self.confidence, 3)
        if "ocr_seed_confidence" in data:
            data["ocr_seed_confidence"] = _round_float(self.ocr_seed_confidence or 0.0, 3)
        if "ocr_consensus_confidence" in data:
            data["ocr_consensus_confidence"] = _round_float(self.ocr_consensus_confidence or 0.0, 3)
        return data


def _box_value(box: TextBox | dict, key: str):
    if isinstance(box, dict):
        if key == "center_x" and "center_x" not in box:
            return float(box.get("x") or 0) + float(box.get("width") or 0) / 2
        if key == "center_y" and "center_y" not in box:
            return float(box.get("y") or 0) + float(box.get("height") or 0) / 2
        return box.get(key)
    return getattr(box, key)


def effective_is_vertical(box: TextBox | dict) -> bool:
    """Infer usable writing direction from geometry plus text content.

    OCR backends sometimes mark every box as vertical. For downstream LLM
    prompting and overlay rendering, geometry and character mix are more
    reliable than the raw flag alone.
    """
    width = float(_box_value(box, "width") or 0)
    height = float(_box_value(box, "height") or 0)
    raw_is_vertical = bool(_box_value(box, "is_vertical"))
    text = str(_box_value(box, "text") or "")
    compact = text.replace("\n", "").replace(" ", "")

    if width <= 0 or height <= 0:
        return raw_is_vertical
    if width >= height * 1.05:
        return False
    if height >= width * 2.2 and not _looks_horizontal_text(compact):
        return True
    if not compact:
        return raw_is_vertical if height > width else False
    if _looks_horizontal_text(compact):
        return False
    return raw_is_vertical and _looks_vertical_text(compact) and height > width * 1.2


def _looks_horizontal_text(text: str) -> bool:
    if not text:
        return False
    ascii_ratio = sum(ord(char) < 128 for char in text) / len(text)
    digit_ratio = sum(char.isdigit() for char in text) / len(text)
    if ascii_ratio >= 0.5 or digit_ratio >= 0.5:
        return True
    if len(text) <= 4 and any(ord(char) < 128 or char.isdigit() for char in text):
        return True
    if re.search(r"https?://|www\.|[@:/%¥$().,-]", text):
        return True
    return bool(re.search(r"\d+[./:-]\d+|\d{2,}", text))


def _looks_vertical_text(text: str) -> bool:
    if not text:
        return False
    if _looks_horizontal_text(text):
        return False
    cjk_count = sum("\u3040" <= char <= "\u30ff" or "\u4e00" <= char <= "\u9fff" for char in text)
    return cjk_count / len(text) >= 0.6


@dataclass(frozen=True)
class PageResult:
    """OCR result for a single page/image."""

    source: str
    page: int | None
    img_width: int
    img_height: int
    boxes: list[TextBox]

    def to_dict(self, options: PageSerializationOptions | None = None) -> dict:
        options = options or PageSerializationOptions()
        safe_width = max(1, self.img_width)
        safe_height = max(1, self.img_height)
        return {
            "source": self.source,
            "page": self.page,
            "img_width": self.img_width,
            "img_height": self.img_height,
            "box_count": len(self.boxes),
            "boxes": [
                _serialize_box(
                    box,
                    img_width=safe_width,
                    img_height=safe_height,
                    include_absolute_geometry=options.include_absolute_geometry,
                    include_debug_fields=options.include_debug_fields,
                )
                for box in self.boxes
            ],
        }


def _serialize_box(
    box: TextBox,
    *,
    img_width: int,
    img_height: int,
    include_absolute_geometry: bool,
    include_debug_fields: bool,
) -> dict:
    data: dict[str, object] = {}
    if box.order is not None:
        data["id"] = box.order
    data["text"] = box.text
    data["x"] = _round_float(box.x / img_width, 4)
    data["y"] = _round_float(box.y / img_height, 4)
    data["width"] = _round_float(box.width / img_width, 4)
    data["height"] = _round_float(box.height / img_height, 4)
    if box.decision is not None:
        data["decision"] = box.decision
    data["confidence"] = _round_float(box.confidence, 3)
    if include_absolute_geometry:
        data["pixel_x"] = _round_float(box.x, 1)
        data["pixel_y"] = _round_float(box.y, 1)
        data["pixel_width"] = box.width
        data["pixel_height"] = box.height
    if include_debug_fields:
        if box.type:
            data["type"] = box.type
        if box.is_vertical is not None:
            data["is_vertical"] = box.is_vertical
        if box.text_source is not None:
            data["text_source"] = box.text_source
        if box.box_source is not None:
            data["box_source"] = box.box_source
        if box.ocr_seed_text is not None:
            data["ocr_seed_text"] = box.ocr_seed_text
        if box.ocr_seed_confidence is not None:
            data["ocr_seed_confidence"] = _round_float(box.ocr_seed_confidence, 3)
        if box.ocr_match_count is not None:
            data["ocr_match_count"] = box.ocr_match_count
        if box.ocr_consensus_text is not None:
            data["ocr_consensus_text"] = box.ocr_consensus_text
        if box.ocr_consensus_confidence is not None:
            data["ocr_consensus_confidence"] = _round_float(box.ocr_consensus_confidence, 3)
        if box.low_ink is not None:
            data["low_ink"] = box.low_ink
    return data
