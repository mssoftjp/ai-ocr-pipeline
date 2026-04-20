#!/usr/bin/env python3
"""Reconstruct a readable page image from positioned OCR boxes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

DEFAULT_FONT = Path("/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render OCR JSON boxes back onto a blank page.")
    parser.add_argument("result_json", type=Path, help="Pipeline JSON output path.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tmp/reconstructed"),
        help="Directory for generated images.",
    )
    parser.add_argument(
        "--font",
        type=Path,
        default=DEFAULT_FONT,
        help="Font file used to draw text.",
    )
    parser.add_argument(
        "--page",
        type=int,
        default=1,
        help="1-based page number within the JSON results.",
    )
    return parser.parse_args()


def _load_page(result_json: Path, page_number: int) -> dict:
    payload = json.loads(result_json.read_text())
    results = payload.get("results")
    if not isinstance(results, list):
        raise ValueError("Expected top-level 'results' list in JSON output.")

    for result in results:
        if result.get("page") == page_number:
            return result

    if page_number == 1 and results:
        return results[0]
    raise ValueError(f"Page {page_number} not found in {result_json}.")


def _font(font_path: Path, height: float) -> ImageFont.FreeTypeFont:
    font_size = max(12, min(int(height * 0.8), 48))
    if font_path.exists():
        try:
            return ImageFont.truetype(str(font_path), font_size)
        except OSError:
            pass
    return ImageFont.load_default()


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


def _render(result: dict, *, font_path: Path, boxed: bool) -> Image.Image:
    image = Image.new(
        "RGB",
        (int(result["img_width"]), int(result["img_height"])),
        "white",
    )
    draw = ImageDraw.Draw(image)

    def sort_key(item: dict) -> tuple[float, float, int]:
        left, top, width, height = _box_geometry(result, item)
        center_x = left + width / 2
        center_y = top + height / 2
        return (center_y, center_x, int(item.get("id", 0)))

    for box in sorted(result["boxes"], key=sort_key):
        left, top, width, height = _box_geometry(result, box)
        right = left + width
        bottom = top + height
        font = _font(font_path, height)

        if boxed:
            draw.rectangle((left, top, right, bottom), outline="#cc6666", width=2)
        draw.text((left, top), box["text"], fill="black", font=font)

    return image


def main() -> int:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = _load_page(args.result_json, args.page)
    stem = args.result_json.stem

    plain_path = args.output_dir / f"{stem}_reconstructed.png"
    boxed_path = args.output_dir / f"{stem}_reconstructed_boxed.png"

    _render(result, font_path=args.font, boxed=False).save(plain_path)
    _render(result, font_path=args.font, boxed=True).save(boxed_path)

    print(plain_path)
    print(boxed_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
