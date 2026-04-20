"""LLM-based OCR refinement backends."""

from ai_ocr_pipeline.llm.lmstudio import (
    HINT_MODES,
    LLMRefinementStats,
    LMStudioConfig,
    refine_page_result,
    refine_page_result_with_stats,
    resolve_model,
)

__all__ = [
    "HINT_MODES",
    "LLMRefinementStats",
    "LMStudioConfig",
    "refine_page_result",
    "refine_page_result_with_stats",
    "resolve_model",
]
