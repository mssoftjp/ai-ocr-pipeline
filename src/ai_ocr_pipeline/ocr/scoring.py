"""OCR result quality scoring for candidate selection."""

from __future__ import annotations

import re

from ai_ocr_pipeline.models import PageResult, TextBox


def score_result(result: PageResult) -> tuple[float, int, float]:
    """Score OCR results by estimated text quality, then confidence."""
    page_area = max(1, result.img_width * result.img_height)
    box_scores = [_score_box_quality(box, page_area=page_area) for box in result.boxes]
    quality_score = round(sum(box_scores), 3)
    usable_box_count = sum(1 for score in box_scores if score > 0)
    return (
        quality_score,
        usable_box_count,
        sum(box.confidence for box in result.boxes),
    )


def _score_box_quality(box: TextBox, *, page_area: int) -> float:
    text = box.text.strip()
    if not text:
        return -1.0

    chars = [char for char in text if not char.isspace()]
    if not chars:
        return -1.0

    signal_chars = sum(1 for char in chars if _is_text_signal_char(char))
    signal_ratio = signal_chars / len(chars)
    score = min(signal_chars, 24) * signal_ratio

    if len(chars) <= 1:
        score -= 1.5
    if signal_ratio < 0.45:
        score -= (0.45 - signal_ratio) * 10

    area_ratio = (box.width * box.height) / page_area
    if area_ratio > 0.12:
        score -= 8
    elif area_ratio > 0.06:
        score -= 3

    if re.search(r"(.)\1{5,}", text):
        score -= 4
    if re.search(r"(?:\bthe\b\s*){4,}|(?:\bto\b\s*){4,}", text, re.IGNORECASE):
        score -= 6
    if set(chars) <= set("-_,.()[]{}0123456789") and len(chars) > 8:
        score -= 4

    return score


def _is_text_signal_char(char: str) -> bool:
    code = ord(char)
    return (
        char.isalnum()
        or 0x3040 <= code <= 0x30FF
        or 0x3400 <= code <= 0x4DBF
        or 0x4E00 <= code <= 0x9FFF
        or 0xF900 <= code <= 0xFAFF
        or 0xFF01 <= code <= 0xFF5E
        or char in "々ー"
    )
