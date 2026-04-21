"""OCR engine and quality scoring."""

from ai_ocr_pipeline.ocr.direct import (
    DEFAULT_GAP_SCORE_THRESHOLD,
    DEFAULT_MIN_GAP_HEIGHT_RATIO,
    DEFAULT_SPLIT_LEVEL,
    run_direct_ocr,
    split_params_for_legacy_sensitivity,
    split_params_for_level,
)
from ai_ocr_pipeline.ocr.engine import parse_ocr_json, run_ocr
from ai_ocr_pipeline.ocr.scoring import score_result

__all__ = [
    "DEFAULT_GAP_SCORE_THRESHOLD",
    "DEFAULT_MIN_GAP_HEIGHT_RATIO",
    "DEFAULT_SPLIT_LEVEL",
    "parse_ocr_json",
    "run_direct_ocr",
    "run_ocr",
    "score_result",
    "split_params_for_legacy_sensitivity",
    "split_params_for_level",
]
