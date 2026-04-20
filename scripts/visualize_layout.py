#!/usr/bin/env python3
"""Visualize extracted layout boxes as SVG overlays."""

from __future__ import annotations

import argparse
import colorsys
import json
from pathlib import Path
from xml.sax.saxutils import escape

import pypdfium2 as pdfium
from PIL import Image, ImageDraw, ImageFont

DEFAULT_FONT = Path("/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render PDF pages and overlay extracted text boxes as SVG.")
    parser.add_argument("result_json", type=Path, help="Pipeline JSON output path.")
    parser.add_argument(
        "--pdf",
        type=Path,
        required=True,
        help="Source PDF used to generate the JSON results.",
    )
    parser.add_argument(
        "--pages",
        type=int,
        nargs="+",
        required=True,
        help="1-based page numbers to visualize.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tmp/layout_viz"),
        help="Directory for generated PNG/SVG files.",
    )
    parser.add_argument(
        "--font",
        type=Path,
        default=DEFAULT_FONT,
        help="Font file used for PNG label rendering.",
    )
    return parser.parse_args()


def _load_results(path: Path) -> dict[int, dict]:
    payload = json.loads(path.read_text())
    results = payload.get("results")
    if not isinstance(results, list):
        raise ValueError("Expected top-level 'results' list in JSON output.")

    pages: dict[int, dict] = {}
    for entry in results:
        page_number = entry.get("page")
        if isinstance(page_number, int):
            pages[page_number] = entry
    return pages


def _page_png_name(stem: str, page_number: int) -> str:
    return f"{stem}_page{page_number:04d}.png"


def _page_overlay_name(stem: str, page_number: int) -> str:
    return f"{stem}_page{page_number:04d}_overlay.svg"


def _page_boxes_name(stem: str, page_number: int) -> str:
    return f"{stem}_page{page_number:04d}_boxes.svg"


def _page_overlay_png_name(stem: str, page_number: int) -> str:
    return f"{stem}_page{page_number:04d}_overlay.png"


def _page_boxes_png_name(stem: str, page_number: int) -> str:
    return f"{stem}_page{page_number:04d}_boxes.png"


def _box_geometry(result: dict, box: dict) -> tuple[float, float, float, float]:
    if all(key in box for key in ("pixel_x", "pixel_y", "pixel_width", "pixel_height")):
        return (
            float(box["pixel_x"]),
            float(box["pixel_y"]),
            float(box["pixel_width"]),
            float(box["pixel_height"]),
        )

    page_width = float(result["img_width"])
    page_height = float(result["img_height"])
    return (
        float(box["x"]) * page_width,
        float(box["y"]) * page_height,
        float(box["width"]) * page_width,
        float(box["height"]) * page_height,
    )


def _color_for_order(order: int) -> str:
    hue = (order * 37) % 360
    return f"hsl({hue} 85% 45%)"


def _rgb_for_order(order: int) -> tuple[int, int, int]:
    hue = ((order * 37) % 360) / 360.0
    red, green, blue = colorsys.hls_to_rgb(hue, 0.45, 0.85)
    return int(red * 255), int(green * 255), int(blue * 255)


def _render_page_png(pdf: pdfium.PdfDocument, page_number: int, result: dict, out_path: Path) -> None:
    page = pdf[page_number - 1]
    page_width_pt, _ = page.get_size()
    target_width = int(result["img_width"])
    scale = target_width / page_width_pt
    bitmap = page.render(scale=scale)
    bitmap.to_pil().save(out_path, format="PNG")


def _build_overlay_svg(
    *,
    page_png_name: str,
    result: dict,
    show_image: bool,
) -> str:
    width = int(result["img_width"])
    height = int(result["img_height"])
    background = "#ffffff" if not show_image else "none"
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'xmlns:xlink="http://www.w3.org/1999/xlink" '
            f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        ),
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="{background}"/>',
    ]
    if show_image:
        parts.append(f'<image href="{escape(page_png_name)}" x="0" y="0" width="{width}" height="{height}"/>')

    for box in result["boxes"]:
        box_id = int(box.get("id", 0))
        left, top, box_width, box_height = _box_geometry(result, box)
        color = _color_for_order(box_id)
        label = escape(f"{box_id}: {box['text'][:24]}")
        label_y = max(14.0, top - 4.0)
        parts.append(
            f'<rect x="{left:.1f}" y="{top:.1f}" width="{box_width:.1f}" '
            f'height="{box_height:.1f}" fill="{color}" fill-opacity="0.10" '
            f'stroke="{color}" stroke-width="2"/>'
        )
        parts.append(f"<title>{escape(box['text'])}</title>")
        parts.append(
            f'<rect x="{left:.1f}" y="{label_y - 12:.1f}" '
            f'width="{max(56, min(len(label) * 7, 420))}" height="14" '
            f'fill="#ffffff" fill-opacity="0.85"/>'
        )
        parts.append(
            f'<text x="{left + 2:.1f}" y="{label_y:.1f}" '
            f'font-size="11" font-family="Helvetica, Arial, sans-serif" '
            f'fill="{color}">{label}</text>'
        )

    parts.append("</svg>")
    return "\n".join(parts)


def _validate_boxes(result: dict) -> list[str]:
    width = float(result["img_width"])
    height = float(result["img_height"])
    errors: list[str] = []
    for box in result["boxes"]:
        left, top, box_width, box_height = _box_geometry(result, box)
        right = left + box_width
        bottom = top + box_height
        box_id = box.get("id", "?")
        if left < -1 or top < -1 or right > width + 1 or bottom > height + 1:
            errors.append(
                f"page {result['page']} box {box_id} out of bounds: ({left:.1f}, {top:.1f}, {right:.1f}, {bottom:.1f})"
            )
    return errors


def _draw_overlay_png(
    *,
    result: dict,
    background_path: Path | None,
    out_path: Path,
    font_path: Path,
) -> None:
    if background_path is not None:
        base = Image.open(background_path).convert("RGBA")
    else:
        base = Image.new(
            "RGBA",
            (int(result["img_width"]), int(result["img_height"])),
            (255, 255, 255, 255),
        )

    overlay = Image.new("RGBA", base.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    font = _load_font(font_path, 14)

    for box in result["boxes"]:
        box_id = int(box.get("id", 0))
        left, top, box_width, box_height = _box_geometry(result, box)
        right = left + box_width
        bottom = top + box_height
        rgb = _rgb_for_order(box_id)
        draw.rectangle(
            (left, top, right, bottom),
            outline=(*rgb, 255),
            width=3,
            fill=(*rgb, 32),
        )
        label = f"{box_id}: {box['text'][:24]}"
        text_bbox = draw.textbbox((0, 0), label, font=font)
        label_width = text_bbox[2] - text_bbox[0] + 6
        label_height = text_bbox[3] - text_bbox[1] + 4
        label_top = max(0.0, top - label_height)
        draw.rectangle(
            (left, label_top, left + label_width, label_top + label_height),
            fill=(255, 255, 255, 220),
        )
        draw.text((left + 3, label_top + 2), label, fill=(*rgb, 255), font=font)

    merged = Image.alpha_composite(base, overlay).convert("RGB")
    merged.save(out_path, format="PNG")


def _load_font(font_path: Path, size: int) -> ImageFont.ImageFont:
    if font_path.exists():
        try:
            return ImageFont.truetype(str(font_path), size)
        except OSError:
            pass
    return ImageFont.load_default()


def main() -> int:
    args = _parse_args()
    page_results = _load_results(args.result_json)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pdf = pdfium.PdfDocument(args.pdf)
    try:
        for page_number in args.pages:
            result = page_results.get(page_number)
            if result is None:
                raise ValueError(f"Page {page_number} not found in JSON results.")

            png_name = _page_png_name(args.result_json.stem, page_number)
            png_path = args.output_dir / png_name
            overlay_path = args.output_dir / _page_overlay_name(args.result_json.stem, page_number)
            boxes_path = args.output_dir / _page_boxes_name(args.result_json.stem, page_number)
            overlay_png_path = args.output_dir / _page_overlay_png_name(args.result_json.stem, page_number)
            boxes_png_path = args.output_dir / _page_boxes_png_name(args.result_json.stem, page_number)

            _render_page_png(pdf, page_number, result, png_path)
            overlay_path.write_text(
                _build_overlay_svg(
                    page_png_name=png_name,
                    result=result,
                    show_image=True,
                )
            )
            boxes_path.write_text(
                _build_overlay_svg(
                    page_png_name=png_name,
                    result=result,
                    show_image=False,
                )
            )
            _draw_overlay_png(
                result=result,
                background_path=png_path,
                out_path=overlay_png_path,
                font_path=args.font,
            )
            _draw_overlay_png(
                result=result,
                background_path=None,
                out_path=boxes_png_path,
                font_path=args.font,
            )

            errors = _validate_boxes(result)
            summary = (
                f"page={page_number} boxes={len(result['boxes'])} "
                f"png={png_path.name} overlay={overlay_path.name} "
                f"boxes_only={boxes_path.name} overlay_png={overlay_png_path.name} "
                f"boxes_png={boxes_png_path.name}"
            )
            print(summary)
            if errors:
                for error in errors:
                    print(f"ERROR {error}")
                return 1
    finally:
        pdf.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
