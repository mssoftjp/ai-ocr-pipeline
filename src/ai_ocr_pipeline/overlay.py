"""Generate OCR overlay SVG/PNG artifacts from PageResult data."""

from __future__ import annotations

import base64
import html
import shutil
import subprocess
import tempfile
from io import BytesIO
from pathlib import Path

from PIL import Image

from ai_ocr_pipeline.models import PageResult, TextBox, effective_is_vertical


def _confidence_color(conf: float) -> str:
    if conf >= 0.85:
        return "#0B6E4F"
    if conf >= 0.75:
        return "#6BAF45"
    if conf >= 0.60:
        return "#E6B800"
    if conf >= 0.45:
        return "#FF8C00"
    return "#DC143C"


def should_render_vertical(box: TextBox | dict) -> bool:
    """Choose overlay direction using the same heuristic as LLM prompting."""
    return effective_is_vertical(box)


def _page_relative_font_sizes(page_width: int, page_height: int) -> tuple[int, int]:
    """Choose readable minimum font sizes based on page dimensions."""
    long_side = max(page_width, page_height)
    text_min = max(10, min(30, int(long_side * 0.0045)))
    index_min = max(10, min(18, int(long_side * 0.0024)))
    return text_min, index_min


def generate_svg(
    page_result: PageResult,
    image_path: Path,
    *,
    bg_opacity: float = 0.25,
    show_index: bool = True,
    max_text_len: int | None = None,
) -> str:
    with Image.open(image_path) as img:
        if img.mode != "RGB":
            img = img.convert("RGB")

        buffer = BytesIO()
        img.save(buffer, format="PNG")
        img_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")

    img_w = page_result.img_width
    img_h = page_result.img_height
    min_font_size, min_index_size = _page_relative_font_sizes(img_w, img_h)
    lines: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{img_w}" height="{img_h}" viewBox="0 0 {img_w} {img_h}">',
        "  <style>",
        "    text { font-family: 'Hiragino Kaku Gothic ProN', 'Noto Sans JP', sans-serif; }",
        "  </style>",
        f'  <image href="data:image/png;base64,{img_b64}" width="{img_w}" height="{img_h}" opacity="{bg_opacity}"/>',
    ]

    for i, box in enumerate(page_result.boxes):
        w, h = box.width, box.height
        x = box.x
        y = box.y
        color = _confidence_color(box.confidence)
        text = box.text.replace("\n", "↩")
        is_vertical = should_render_vertical(box)

        lines.append(
            f'  <rect x="{x:.0f}" y="{y:.0f}" width="{w}" height="{h}" '
            f'fill="{color}" fill-opacity="0.08" stroke="{color}" stroke-width="1.5" />'
        )

        display = text if max_text_len is None or max_text_len <= 0 else text[:max_text_len]
        escaped = html.escape(display)
        if is_vertical:
            font_size = min(max(22, min_font_size + 10), max(min_font_size, int(w * 0.9)))
            if show_index:
                lines.append(
                    f'  <text x="{x + 2:.0f}" y="{y + 10:.0f}" '
                    f'font-size="{min_index_size}" fill="{color}" opacity="0.9">[{i}]</text>'
                )
            lines.append(
                f'  <text x="{x + w - 2:.0f}" y="{y + 2:.0f}" '
                f'font-size="{font_size}" fill="{color}" opacity="0.9" '
                f'writing-mode="vertical-rl" text-orientation="upright" dominant-baseline="hanging">'
                f"{escaped}</text>"
            )
        else:
            font_size = min(max(22, min_font_size + 10), max(min_font_size, int(h * 0.75)))
            prefix = f"[{i}] " if show_index else ""
            lines.append(
                f'  <text x="{x + 2:.0f}" y="{y + h * 0.72:.0f}" '
                f'font-size="{font_size}" fill="{color}" opacity="0.9">'
                f"{prefix}{escaped}</text>"
            )

    lines.append("</svg>")
    return "\n".join(lines)


def _render_svg_to_png(svg_content: str, output_path: Path) -> bool:
    try:
        import cairosvg  # type: ignore[import-untyped]

        cairosvg.svg2png(bytestring=svg_content.encode("utf-8"), write_to=str(output_path))
        return True
    except ImportError:
        pass

    rsvg_convert = shutil.which("rsvg-convert")
    if not rsvg_convert:
        return False

    with tempfile.NamedTemporaryFile("w", suffix=".svg", encoding="utf-8", delete=False) as tmp:
        tmp.write(svg_content)
        tmp_svg = Path(tmp.name)

    try:
        subprocess.run(
            [rsvg_convert, str(tmp_svg), "-o", str(output_path)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except subprocess.CalledProcessError:
        return False
    finally:
        tmp_svg.unlink(missing_ok=True)


def svg_to_png(svg_content: str, output_path: Path) -> tuple[str, Path]:
    """Render raw SVG content to PNG when possible, else save an SVG fallback."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if _render_svg_to_png(svg_content, output_path):
        return "png", output_path

    svg_path = output_path.with_suffix(".svg")
    svg_path.write_text(svg_content, encoding="utf-8")
    return "svg", svg_path


def write_overlay_artifact(
    page_result: PageResult,
    image_path: Path,
    output_path: Path,
    *,
    bg_opacity: float = 0.25,
    show_index: bool = True,
    max_text_len: int | None = None,
) -> tuple[str, Path]:
    """Write a PNG overlay when possible, else save an SVG fallback."""
    svg_content = generate_svg(
        page_result,
        image_path,
        bg_opacity=bg_opacity,
        show_index=show_index,
        max_text_len=max_text_len,
    )
    return svg_to_png(svg_content, output_path)
