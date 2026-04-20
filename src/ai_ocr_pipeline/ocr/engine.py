"""ndlocr-lite OCR engine wrapper.

Calls ndlocr-lite as a subprocess and parses the JSON output.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

from ai_ocr_pipeline.models import PageResult, TextBox


def _find_ndlocr_lite() -> str:
    """Locate the ndlocr-lite command."""
    # Check PATH first
    path = shutil.which("ndlocr-lite")
    if path:
        return path
    # Check alongside the current Python interpreter (same venv)
    venv_bin = Path(sys.executable).parent / "ndlocr-lite"
    if venv_bin.exists():
        return str(venv_bin)
    raise FileNotFoundError(
        "ndlocr-lite command not found. Install it with: pip install ndlocr-lite  or  uv tool install ndlocr-lite"
    )


def run_ocr(
    image_path: Path,
    output_dir: Path,
    *,
    device: str = "cpu",
) -> Path:
    """Run ndlocr-lite on a single image.

    Returns the path to the generated JSON file.
    """
    cmd = [
        _find_ndlocr_lite(),
        "--sourceimg",
        str(image_path),
        "--output",
        str(output_dir),
        "--device",
        device,
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ndlocr-lite failed (exit {result.returncode}):\n{result.stderr}")

    json_path = output_dir / f"{image_path.stem}.json"
    if not json_path.exists():
        raise FileNotFoundError(
            f"Expected output not found: {json_path}\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
    _compact_ocr_json(json_path)
    return json_path


def _compact_ocr_json(json_path: Path) -> None:
    """Rewrite ndlocr-lite JSON with compact boundingBox and readable key order."""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    for group in data.get("contents", []):
        for i, item in enumerate(group):
            group[i] = {
                "id": item.get("id"),
                "text": item.get("text", ""),
                "boundingBox": item.get("boundingBox", []),
                **{k: v for k, v in item.items() if k not in ("id", "text", "boundingBox")},
            }
    text = json.dumps(data, ensure_ascii=False, indent=2)
    # Collapse 4-point boundingBox arrays to single lines.
    text = re.sub(
        r'"boundingBox": \[\s*\[\s*(\d+),\s*(\d+)\s*\],\s*\[\s*(\d+),\s*(\d+)\s*\],'
        r"\s*\[\s*(\d+),\s*(\d+)\s*\],\s*\[\s*(\d+),\s*(\d+)\s*\]\s*\]",
        lambda m: f'"boundingBox": [[{m[1]}, {m[2]}], [{m[3]}, {m[4]}], [{m[5]}, {m[6]}], [{m[7]}, {m[8]}]]',
        text,
    )
    json_path.write_text(text, encoding="utf-8")


def parse_ocr_json(
    json_path: Path,
    *,
    source: str = "",
    page: int | None = None,
) -> PageResult:
    """Parse ndlocr-lite JSON output into a PageResult."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    imginfo = data.get("imginfo", {})
    img_width = imginfo.get("img_width", 0)
    img_height = imginfo.get("img_height", 0)

    boxes: list[TextBox] = []
    contents = data.get("contents", [[]])

    for group in contents:
        for item in group:
            bbox = item.get("boundingBox", [])
            if len(bbox) < 4:
                continue

            # boundingBox: [[xmin,ymin],[xmin,ymin+h],[xmin+w,ymin],[xmin+w,ymin+h]]
            # top-left, bottom-left, top-right, bottom-right
            x_min = bbox[0][0]
            y_min = bbox[0][1]
            x_max = bbox[3][0]
            y_max = bbox[3][1]

            width = x_max - x_min
            height = y_max - y_min
            is_vertical_str = item.get("isVertical", "false")
            is_vertical = is_vertical_str == "true" if isinstance(is_vertical_str, str) else bool(is_vertical_str)

            box = TextBox(
                text=item.get("text", ""),
                width=width,
                height=height,
                x=x_min,
                y=y_min,
                confidence=item.get("confidence", 0.0),
                order=item.get("id"),
                is_vertical=is_vertical,
                text_source="ocr",
            )
            boxes.append(box)

    return PageResult(
        source=source or imginfo.get("img_name", json_path.stem),
        page=page,
        img_width=img_width,
        img_height=img_height,
        boxes=boxes,
    )
