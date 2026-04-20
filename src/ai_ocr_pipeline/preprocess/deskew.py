"""Image deskew pre-processing using vendored deskew_HT."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ai_ocr_pipeline._vendored.deskew_ht import Deskew


def deskew_image(image_path: Path, output_path: Path) -> Path:
    """Deskew a single image and write the result.

    Returns the output path.
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    deskewed = deskew_in_memory(img)
    cv2.imwrite(str(output_path), deskewed)
    return output_path


def deskew_in_memory(img: np.ndarray) -> np.ndarray:
    """Deskew a BGR numpy array in memory. Returns deskewed BGR array."""
    d = Deskew(skew_max=4.0, acc_deg=0.5)
    return d.deskew_on_memory(img)
