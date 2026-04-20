#!/usr/bin/env python3
"""Generate SVG/PNG text map from OCR JSON results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ai_ocr_pipeline.models import PageResult, TextBox
from ai_ocr_pipeline.overlay import (
    _confidence_color,
    _page_relative_font_sizes,
    write_overlay_artifact,
)
from ai_ocr_pipeline.overlay import (
    generate_svg as _generate_svg_for_page_result,
)
from ai_ocr_pipeline.overlay import (
    should_render_vertical as _should_render_vertical,
)


def _box_from_json(box: dict, *, img_width: int, img_height: int) -> TextBox:
    if all(key in box for key in ("pixel_x", "pixel_y", "pixel_width", "pixel_height")):
        return TextBox(
            text=box.get("text", ""),
            x=box["pixel_x"],
            y=box["pixel_y"],
            width=box["pixel_width"],
            height=box["pixel_height"],
            confidence=box.get("confidence", 0.0),
            order=box.get("id"),
            type=box.get("type"),
            is_vertical=box.get("is_vertical"),
            text_source=box.get("text_source"),
            box_source=box.get("box_source"),
            decision=box.get("decision"),
        )

    if all(key in box for key in ("x", "y", "width", "height")):
        return TextBox(
            text=box.get("text", ""),
            x=float(box["x"]) * img_width,
            y=float(box["y"]) * img_height,
            width=max(1, round(float(box["width"]) * img_width)),
            height=max(1, round(float(box["height"]) * img_height)),
            confidence=box.get("confidence", 0.0),
            order=box.get("id"),
            type=box.get("type"),
            is_vertical=box.get("is_vertical"),
            text_source=box.get("text_source"),
            box_source=box.get("box_source"),
            decision=box.get("decision"),
        )

    raise ValueError("Unsupported box geometry. Expected ratio x/y/width/height or pixel_* fields.")


def _load_page_result(ocr_json_path: Path, *, page_index: int = 0) -> PageResult:
    data = json.loads(ocr_json_path.read_text(encoding="utf-8"))
    result = data["results"][page_index]
    img_width = result["img_width"]
    img_height = result["img_height"]
    return PageResult(
        source=result.get("source", ocr_json_path.name),
        page=result.get("page"),
        img_width=img_width,
        img_height=img_height,
        boxes=[_box_from_json(box, img_width=img_width, img_height=img_height) for box in result["boxes"]],
    )


def generate_svg(
    ocr_json_path: Path,
    image_path: Path,
    *,
    bg_opacity: float = 0.25,
    show_index: bool = True,
    max_text_len: int = 40,
) -> str:
    return _generate_svg_for_page_result(
        _load_page_result(ocr_json_path),
        image_path,
        bg_opacity=bg_opacity,
        show_index=show_index,
        max_text_len=max_text_len,
    )


def svg_to_png(svg_content: str, output_path: Path) -> tuple[str, Path]:
    from ai_ocr_pipeline.overlay import svg_to_png as _svg_to_png

    return _svg_to_png(svg_content, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text map overlay from OCR JSON")
    parser.add_argument("ocr_json", type=Path, help="Path to OCR result JSON")
    parser.add_argument("source_image", type=Path, help="Path to source image")
    parser.add_argument("--output", "-o", type=Path, default=None, help="Output path")
    parser.add_argument("--format", "-f", choices=["svg", "png"], default="svg")
    parser.add_argument("--bg-opacity", type=float, default=0.25)
    parser.add_argument("--no-index", action="store_true")
    args = parser.parse_args()

    if args.output is None:
        args.output = args.ocr_json.with_name(args.ocr_json.stem + f"_map.{args.format}")

    page_result = _load_page_result(args.ocr_json)
    if args.format == "svg":
        svg_content = _generate_svg_for_page_result(
            page_result,
            args.source_image,
            bg_opacity=args.bg_opacity,
            show_index=not args.no_index,
        )
        args.output.write_text(svg_content, encoding="utf-8")
        print(f"SVG saved: {args.output} ({args.output.stat().st_size // 1024}KB)")
        return

    actual_format, actual_path = write_overlay_artifact(
        page_result,
        args.source_image,
        args.output,
        bg_opacity=args.bg_opacity,
        show_index=not args.no_index,
    )
    if actual_format == "png":
        print(f"PNG saved: {actual_path}")
    else:
        print(f"PNG renderer unavailable. SVG saved instead: {actual_path}")


__all__ = [
    "_confidence_color",
    "_page_relative_font_sizes",
    "_should_render_vertical",
    "generate_svg",
    "main",
    "svg_to_png",
]


if __name__ == "__main__":
    main()
