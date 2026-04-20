"""PDF helpers for rasterization and embedded text extraction."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pypdfium2 as pdfium

from ai_ocr_pipeline.models import PageResult, TextBox


@dataclass(frozen=True)
class _TextRect:
    text: str
    left: float
    top: float
    right: float
    bottom: float

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def height(self) -> float:
        return self.bottom - self.top

    @property
    def center_y(self) -> float:
        return self.top + self.height / 2


def _clean_text(text: str) -> str:
    text = text.replace("\r", "").replace("\n", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _convert_rect_to_pixels(
    rect: tuple[float, float, float, float],
    page_height_pt: float,
    scale: float,
) -> _TextRect:
    left, bottom, right, top = rect
    text_top = (page_height_pt - top) * scale
    text_bottom = (page_height_pt - bottom) * scale
    return _TextRect(
        text="",
        left=left * scale,
        top=text_top,
        right=right * scale,
        bottom=text_bottom,
    )


def _merge_text_rects(rects: list[_TextRect]) -> list[_TextRect]:
    if not rects:
        return []

    sorted_rects = sorted(rects, key=lambda rect: (rect.top, rect.left))
    lines: list[list[_TextRect]] = []
    for rect in sorted_rects:
        if not lines:
            lines.append([rect])
            continue

        last_line = lines[-1]
        line_center = sum(item.center_y for item in last_line) / len(last_line)
        line_height = max(item.height for item in last_line)
        if abs(rect.center_y - line_center) <= max(line_height, rect.height) * 0.7:
            last_line.append(rect)
        else:
            lines.append([rect])

    merged: list[_TextRect] = []
    for line in lines:
        line = sorted(line, key=lambda rect: rect.left)
        chunk = [line[0]]
        for rect in line[1:]:
            prev = chunk[-1]
            gap = rect.left - prev.right
            threshold = max(prev.height, rect.height) * 1.5
            if gap <= threshold:
                chunk.append(rect)
                continue
            merged.append(_merge_chunk(chunk))
            chunk = [rect]
        merged.append(_merge_chunk(chunk))

    return merged


def _merge_chunk(chunk: list[_TextRect]) -> _TextRect:
    parts: list[str] = []
    for index, item in enumerate(chunk):
        if index > 0 and _needs_space_between(chunk[index - 1], item):
            parts.append(" ")
        parts.append(item.text)

    return _TextRect(
        text="".join(parts),
        left=min(item.left for item in chunk),
        top=min(item.top for item in chunk),
        right=max(item.right for item in chunk),
        bottom=max(item.bottom for item in chunk),
    )


def _needs_space_between(left: _TextRect, right: _TextRect) -> bool:
    gap = right.left - left.right
    threshold = max(left.height, right.height) * 0.6
    if gap <= threshold:
        return False

    left_char = _last_visible_char(left.text)
    right_char = _first_visible_char(right.text)
    if left_char is None or right_char is None:
        return False

    if _is_cjk_char(left_char) or _is_cjk_char(right_char):
        return False
    if right_char in ",.;:!?)]}%":
        return False
    return left_char not in "([{"


def _first_visible_char(text: str) -> str | None:
    for char in text:
        if not char.isspace():
            return char
    return None


def _last_visible_char(text: str) -> str | None:
    for char in reversed(text):
        if not char.isspace():
            return char
    return None


def _is_cjk_char(char: str) -> bool:
    code = ord(char)
    return 0x3040 <= code <= 0x30FF or 0x3400 <= code <= 0x4DBF or 0x4E00 <= code <= 0x9FFF or 0xF900 <= code <= 0xFAFF


def _extract_page_text(
    page: pdfium.PdfPage,
    *,
    source: str,
    page_number: int,
    dpi: int,
    min_chars: int = 10,
) -> PageResult | None:
    textpage = page.get_textpage()
    char_count = textpage.count_chars()
    if char_count < min_chars:
        return None

    raw_text = _clean_text(textpage.get_text_bounded())
    if len(raw_text) < min_chars:
        return None

    scale = dpi / 72
    page_width_pt, page_height_pt = page.get_size()
    rect_count = textpage.count_rects()
    rects: list[_TextRect] = []

    for index in range(rect_count):
        rect = textpage.get_rect(index)
        text = _clean_text(textpage.get_text_bounded(*rect))
        if not text:
            continue
        converted = _convert_rect_to_pixels(rect, page_height_pt, scale)
        rects.append(
            _TextRect(
                text=text,
                left=converted.left,
                top=converted.top,
                right=converted.right,
                bottom=converted.bottom,
            )
        )

    merged_rects = _merge_text_rects(rects)
    if not merged_rects:
        return None

    boxes: list[TextBox] = []
    for order, rect in enumerate(merged_rects):
        width = round(rect.width)
        height = round(rect.height)
        boxes.append(
            TextBox(
                text=rect.text,
                width=width,
                height=height,
                x=rect.left,
                y=rect.top,
                confidence=1.0,
                order=order,
                type="text_layer",
                is_vertical=height > width,
                text_source="text_layer",
            )
        )

    return PageResult(
        source=source,
        page=page_number,
        img_width=round(page_width_pt * scale),
        img_height=round(page_height_pt * scale),
        boxes=boxes,
    )


def extract_pdf_text_layers(
    pdf_path: Path,
    *,
    dpi: int = 300,
    min_chars: int = 10,
) -> list[PageResult | None]:
    """Extract positioned text from a PDF text layer when available."""
    pdf = pdfium.PdfDocument(pdf_path)
    results: list[PageResult | None] = []
    for index in range(len(pdf)):
        page = pdf[index]
        results.append(
            _extract_page_text(
                page,
                source=pdf_path.name,
                page_number=index + 1,
                dpi=dpi,
                min_chars=min_chars,
            )
        )
    pdf.close()
    return results


def pdf_to_images(
    pdf_path: Path,
    output_dir: Path,
    dpi: int = 300,
    page_numbers: list[int] | None = None,
) -> list[Path]:
    """Convert each page of a PDF to a PNG image.

    Returns list of generated image paths.
    """
    pdf = pdfium.PdfDocument(pdf_path)
    image_paths: list[Path] = []
    target_pages = set(page_numbers) if page_numbers is not None else None

    for i in range(len(pdf)):
        if target_pages is not None and (i + 1) not in target_pages:
            continue
        page = pdf[i]
        bitmap = page.render(scale=dpi / 72)
        pil_image = bitmap.to_pil()

        out_path = output_dir / f"{pdf_path.stem}_page{i + 1:04d}.png"
        pil_image.save(out_path, format="PNG")
        image_paths.append(out_path)

    pdf.close()
    return image_paths
