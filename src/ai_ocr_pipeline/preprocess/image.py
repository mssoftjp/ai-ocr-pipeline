"""Image preprocessing variants for OCR candidate selection."""

from __future__ import annotations

from pathlib import Path


def ensure_rgb(image_path: Path, work_dir: Path) -> Path:
    """Convert non-RGB images (1-bit, grayscale, palette) to RGB PNG.

    Returns the original path if already RGB, or a new converted path.
    """
    from PIL import Image

    with Image.open(image_path) as img:
        if img.mode in ("RGB", "RGBA"):
            return image_path
        converted = img.convert("RGB")
        out_path = work_dir / f"{image_path.stem}_rgb.png"
        converted.save(out_path, format="PNG")
        return out_path


def build_inverted_variant(image_path: Path, work_dir: Path) -> Path:
    """Build an inverted RGB variant for OCR comparison."""
    from PIL import Image, ImageOps

    with Image.open(image_path) as img:
        inverted = ImageOps.invert(img.convert("L")).convert("RGB")
        out_path = work_dir / f"{image_path.stem}_inverted.png"
        inverted.save(out_path, format="PNG")
        return out_path


def build_line_removed_variant(
    image_path: Path,
    work_dir: Path,
    *,
    remove_horizontal_lines: bool,
    remove_vertical_lines: bool,
    invert_output: bool,
) -> Path:
    """Build an RGB PNG variant with long table/form lines removed."""
    import cv2
    import numpy as np
    from PIL import Image

    with Image.open(image_path) as img:
        gray = np.array(img.convert("L"))

    _, binary_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    height, width = binary_inv.shape
    lines = np.zeros_like(binary_inv)

    if remove_horizontal_lines:
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, width // 25), 1))
        lines_h = cv2.morphologyEx(binary_inv, cv2.MORPH_OPEN, h_kernel)
        lines = cv2.bitwise_or(lines, lines_h)

    if remove_vertical_lines:
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, height // 25)))
        lines_v = cv2.morphologyEx(binary_inv, cv2.MORPH_OPEN, v_kernel)
        lines = cv2.bitwise_or(lines, lines_v)

    cleaned = cv2.subtract(binary_inv, lines)
    output = 255 - cleaned if invert_output else cleaned

    tags = []
    if remove_horizontal_lines:
        tags.append("h")
    if remove_vertical_lines:
        tags.append("v")
    tags.append("inv" if invert_output else "plain")
    out_path = work_dir / f"{image_path.stem}_{'_'.join(tags)}_line_removed.png"
    Image.fromarray(output).convert("RGB").save(out_path, format="PNG")
    return out_path
