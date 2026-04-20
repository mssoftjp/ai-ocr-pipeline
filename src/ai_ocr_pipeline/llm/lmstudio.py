"""LM Studio OpenAI-compatible helpers for hybrid OCR refinement."""

from __future__ import annotations

import base64
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from io import BytesIO
from pathlib import Path
from urllib import error, request

from PIL import Image

from ai_ocr_pipeline.models import PageResult, TextBox, effective_is_vertical
from ai_ocr_pipeline.template import TemplatePromptContext

HINT_MODES = ("full", "weak", "none")
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class LMStudioConfig:
    """Connection settings for the local LM Studio OpenAI-compatible API."""

    base_url: str = "http://127.0.0.1:1234/v1"
    model: str | None = None
    api_key: str | None = None
    timeout: float = 120.0
    max_image_side: int = 2048
    crop_padding_ratio: float = 0.25
    crop_padding_ratio_y: float | None = None
    max_tokens_per_request: int = 4096
    hint_mode: str = "full"
    box_indices: tuple[int, ...] | None = None
    save_crops_dir: str | None = None
    confidence_threshold: float | None = None
    context_confidence: float = 0.5
    max_workers: int = 4


@dataclass(frozen=True)
class BoxLLMResponse:
    """Normalized LM Studio response for a single OCR box request."""

    text: str
    finish_reason: str | None
    completion_tokens: int | None = None
    reasoning_tokens: int | None = None


@dataclass
class LLMRefinementStats:
    """Aggregated counters for one or more LM Studio refinement runs."""

    requests_total: int = 0
    finish_reason_stop: int = 0
    finish_reason_length: int = 0
    blank_responses: int = 0
    retried_on_length: int = 0
    fallback_count: int = 0

    def record_response(self, response: BoxLLMResponse) -> None:
        self.requests_total += 1
        if response.finish_reason == "stop":
            self.finish_reason_stop += 1
        elif response.finish_reason == "length":
            self.finish_reason_length += 1
        if not response.text.strip():
            self.blank_responses += 1

    def merge(self, other: LLMRefinementStats | None) -> None:
        if other is None:
            return
        self.requests_total += other.requests_total
        self.finish_reason_stop += other.finish_reason_stop
        self.finish_reason_length += other.finish_reason_length
        self.blank_responses += other.blank_responses
        self.retried_on_length += other.retried_on_length
        self.fallback_count += other.fallback_count

    def to_dict(self) -> dict[str, int]:
        return {
            "requests_total": self.requests_total,
            "finish_reason_stop": self.finish_reason_stop,
            "finish_reason_length": self.finish_reason_length,
            "blank_responses": self.blank_responses,
            "retried_on_length": self.retried_on_length,
            "fallback_count": self.fallback_count,
        }


def _find_neighbor_labels(
    target: TextBox,
    all_boxes: list[TextBox],
    *,
    max_gap: float = 80.0,
    max_labels: int = 2,
) -> list[str]:
    """Find text from boxes immediately to the left or above *target*.

    Returns short strings like ``"left: 金額"`` or ``"above: 日付"``.
    Only boxes whose text looks like a label (short, mostly non-numeric)
    are included.
    """
    candidates: list[tuple[float, str]] = []
    for box in all_boxes:
        if box is target or not box.text.strip():
            continue
        text = box.text.strip()
        # Skip long text — unlikely to be a label
        if len(text) > 20:
            continue
        # Skip mostly-numeric text — it's data, not a label
        non_space = text.replace(" ", "")
        if non_space and sum(c.isdigit() for c in non_space) / len(non_space) > 0.5:
            continue

        half_w = max(1.0, target.width / 2)
        half_h = max(1.0, target.height / 2)

        # Left neighbor: overlapping vertical band, box is to the left
        if (
            abs(box.center_y - target.center_y) < max(target.height, box.height) * 0.7
            and box.center_x + box.width / 2 <= target.center_x - half_w + max_gap
            and target.center_x - box.center_x < max_gap + box.width
        ):
            dist = target.center_x - box.center_x
            candidates.append((dist, f"left: {text}"))

        # Above neighbor: overlapping horizontal band, box is above
        elif (
            abs(box.center_x - target.center_x) < max(target.width, box.width) * 0.7
            and box.center_y + box.height / 2 <= target.center_y - half_h + max_gap
            and target.center_y - box.center_y < max_gap + box.height
        ):
            dist = target.center_y - box.center_y
            candidates.append((dist, f"above: {text}"))

    candidates.sort(key=lambda pair: pair[0])
    return [label for _, label in candidates[:max_labels]]


def normalize_base_url(base_url: str) -> str:
    """Normalize a user-supplied base URL to the OpenAI-compatible /v1 root."""
    normalized = base_url.rstrip("/")
    if normalized.endswith("/api/v1"):
        return normalized[: -len("/api/v1")] + "/v1"
    if normalized.endswith("/v1"):
        return normalized
    return normalized + "/v1"


def resolve_model(config: LMStudioConfig) -> str:
    """Resolve the target model id, defaulting to the first visible model."""
    if config.model:
        return config.model

    payload = _request_json(
        f"{normalize_base_url(config.base_url)}/models",
        timeout=config.timeout,
        api_key=config.api_key,
    )
    for model in payload.get("data", []):
        model_id = model.get("id")
        if isinstance(model_id, str) and model_id:
            return model_id

    raise RuntimeError("No model was returned by LM Studio at /v1/models.")


def refine_page_result_with_stats(
    image_path: Path,
    page_result: PageResult,
    config: LMStudioConfig,
    template_contexts: dict[int, TemplatePromptContext] | None = None,
    selected_box_indices: tuple[int, ...] | None = None,
) -> tuple[PageResult, LLMRefinementStats]:
    """Refine OCR text box-by-box using cropped images while preserving geometry."""
    stats = LLMRefinementStats()
    if not page_result.boxes:
        return page_result, stats

    threshold = config.confidence_threshold
    if template_contexts is not None:
        refinable_boxes = [
            {"index": index, "box": box}
            for index, box in enumerate(page_result.boxes)
            if box.type != "text_layer"
            and index in template_contexts
            and (selected_box_indices is None or index in selected_box_indices)
        ]
    else:
        refinable_boxes = [
            {"index": index, "box": box}
            for index, box in enumerate(page_result.boxes)
            if box.type != "text_layer"
            and (config.box_indices is None or index in config.box_indices)
            and (threshold is None or box.confidence < threshold)
        ]
    if not refinable_boxes:
        return page_result, stats

    skipped_count = len(page_result.boxes) - len(refinable_boxes)
    if skipped_count > 0:
        LOGGER.debug(
            "Skipping %d/%d boxes (text_layer, box_indices filter, or confidence >= %s)",
            skipped_count,
            len(page_result.boxes),
            threshold,
        )

    resolved = replace(
        config,
        base_url=normalize_base_url(config.base_url),
        model=config.model or resolve_model(config),
    )
    replacements: dict[int, str] = {}
    retry_items: list[dict] = []

    crops_dir: Path | None = None
    if resolved.save_crops_dir:
        crops_dir = Path(resolved.save_crops_dir)
        crops_dir.mkdir(parents=True, exist_ok=True)

    all_boxes = page_result.boxes

    # Phase 1: prepare crops and neighbor labels on the main thread
    # (PIL Image is not thread-safe, so crop extraction must be serial).
    prepared: list[dict] = []
    with Image.open(image_path) as image:
        for item in refinable_boxes:
            index = item["index"]
            box = item["box"]
            try:
                crop_url = _crop_box_data_url(
                    image,
                    box,
                    max_image_side=resolved.max_image_side,
                    padding_ratio=resolved.crop_padding_ratio,
                    padding_ratio_y=resolved.crop_padding_ratio_y,
                )
                if crops_dir is not None:
                    _save_crop_image(crop_url, crops_dir / f"box_{index:04d}.png")
                # Only add neighbor context for low-confidence boxes to avoid
                # misleading the model on boxes where the OCR is likely correct.
                neighbor_labels = (
                    _find_neighbor_labels(box, all_boxes) if box.confidence < resolved.context_confidence else []
                )
                prepared.append(
                    {
                        "index": index,
                        "box": box,
                        "crop_url": crop_url,
                        "neighbor_labels": neighbor_labels,
                        "template_context": template_contexts.get(index) if template_contexts is not None else None,
                    }
                )
            except Exception as exc:
                LOGGER.debug("LM Studio crop preparation skipped for box %s (%r): %s", index, box.text, exc)

    # Phase 2: send HTTP requests in parallel.
    def _refine_one(item: dict) -> tuple[dict, BoxLLMResponse]:
        response = _request_box_response(
            item["box"],
            item["crop_url"],
            resolved,
            neighbor_labels=item["neighbor_labels"],
            template_context=item["template_context"],
        )
        return item, response

    workers = min(resolved.max_workers, len(prepared)) if prepared else 1
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_refine_one, item): item for item in prepared}
        for future in as_completed(futures):
            item = futures[future]
            try:
                item, response = future.result()
                stats.record_response(response)
                if response.finish_reason == "length":
                    retry_items.append(item)
                    LOGGER.debug(
                        "LM Studio hit max_tokens for box %s (%r); retrying sequentially with a larger token budget.",
                        item["index"],
                        item["box"].text,
                    )
                    continue
                replacements[item["index"]] = _validate_box_text(
                    response.text,
                    item["box"],
                    neighbor_labels=item["neighbor_labels"],
                    template_context=item["template_context"],
                )
            except Exception as exc:
                LOGGER.debug("LM Studio refinement skipped for box %s (%r): %s", item["index"], item["box"].text, exc)
                stats.fallback_count += 1

    if retry_items:
        retry_config = replace(
            resolved,
            max_tokens_per_request=max(resolved.max_tokens_per_request * 4, resolved.max_tokens_per_request),
        )
        for item in retry_items:
            stats.retried_on_length += 1
            try:
                response = _request_box_response(
                    item["box"],
                    item["crop_url"],
                    retry_config,
                    neighbor_labels=item["neighbor_labels"],
                    template_context=item["template_context"],
                )
                stats.record_response(response)
                if response.finish_reason == "length":
                    raise RuntimeError("LM Studio response hit max_tokens even after retry.")
                replacements[item["index"]] = _validate_box_text(
                    response.text,
                    item["box"],
                    neighbor_labels=item["neighbor_labels"],
                    template_context=item["template_context"],
                )
            except Exception as exc:
                LOGGER.debug(
                    "LM Studio retry skipped for box %s (%r): %s",
                    item["index"],
                    item["box"].text,
                    exc,
                )
                stats.fallback_count += 1

    boxes = [
        replace(
            box,
            text=replacements.get(index, box.text) or box.text,
            text_source="llm" if index in replacements else (box.text_source or "ocr"),
        )
        for index, box in enumerate(page_result.boxes)
    ]
    return replace(page_result, boxes=boxes), stats


def refine_page_result(
    image_path: Path,
    page_result: PageResult,
    config: LMStudioConfig,
) -> PageResult:
    """Backward-compatible wrapper returning only the refined page result."""
    refined, _ = refine_page_result_with_stats(image_path, page_result, config)
    return refined


def _box_bounds(
    box: TextBox,
    image_size: tuple[int, int],
    *,
    padding_ratio: float,
    padding_ratio_y: float | None = None,
) -> tuple[int, int, int, int]:
    img_width, img_height = image_size
    pad_x, pad_y = _padding_amounts(
        box,
        padding_ratio=padding_ratio,
        padding_ratio_y=padding_ratio_y,
    )

    left = max(0, round(box.x - pad_x))
    top = max(0, round(box.y - pad_y))
    right = min(img_width, round(box.x + box.width + pad_x))
    bottom = min(img_height, round(box.y + box.height + pad_y))

    if right <= left:
        right = min(img_width, left + 1)
    if bottom <= top:
        bottom = min(img_height, top + 1)
    return left, top, right, bottom


def _crop_box_data_url(
    image: Image.Image,
    box: TextBox,
    *,
    max_image_side: int,
    padding_ratio: float,
    padding_ratio_y: float | None = None,
) -> str:
    crop = image.crop(
        _box_bounds(
            box,
            image.size,
            padding_ratio=padding_ratio,
            padding_ratio_y=padding_ratio_y,
        )
    )
    if max(crop.size) > max_image_side:
        crop.thumbnail((max_image_side, max_image_side))

    buffer = BytesIO()
    crop.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _request_box_response(
    box: TextBox,
    image_url: str,
    config: LMStudioConfig,
    *,
    neighbor_labels: list[str] | None = None,
    template_context: TemplatePromptContext | None = None,
) -> BoxLLMResponse:
    effective_hint_mode = "none" if template_context is not None else config.hint_mode
    payload = {
        "model": config.model,
        "temperature": 0,
        "max_tokens": config.max_tokens_per_request,
        "messages": [
            {
                "role": "system",
                "content": _build_system_prompt(effective_hint_mode),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": _build_refine_prompt(
                            box,
                            hint_mode=effective_hint_mode,
                            neighbor_labels=neighbor_labels,
                            template_context=template_context,
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                    },
                ],
            },
        ],
    }
    response = _request_json(
        f"{config.base_url}/chat/completions",
        payload=payload,
        timeout=config.timeout,
        api_key=config.api_key,
    )
    choice = _extract_primary_choice(response)
    usage = response.get("usage", {})
    completion_tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
    completion_details = usage.get("completion_tokens_details", {}) if isinstance(usage, dict) else {}
    reasoning_tokens = None
    if isinstance(completion_details, dict):
        raw_reasoning_tokens = completion_details.get("reasoning_tokens")
        if isinstance(raw_reasoning_tokens, int):
            reasoning_tokens = raw_reasoning_tokens

    return BoxLLMResponse(
        text=_normalize_box_text(_extract_message_text(response), fallback=""),
        finish_reason=choice.get("finish_reason") if isinstance(choice.get("finish_reason"), str) else None,
        completion_tokens=completion_tokens if isinstance(completion_tokens, int) else None,
        reasoning_tokens=reasoning_tokens,
    )


def _build_system_prompt(hint_mode: str) -> str:
    # System prompt carries only the output format constraints.
    # Task instructions live in the user prompt where the model attends
    # to them more strongly — especially important for vision tasks.
    if hint_mode == "none":
        return (
            "You transcribe text from cropped image regions. "
            "Return only the exact text. "
            "No explanations, JSON, markdown, code blocks, or quotes."
        )
    if hint_mode == "weak":
        return (
            "You transcribe text from cropped image regions. "
            "Return only the text you see. "
            "No explanations, JSON, markdown, code blocks, or quotes."
        )
    return (
        "You correct OCR text from cropped image regions. "
        "Return only the corrected text. "
        "No explanations, JSON, markdown, code blocks, or quotes."
    )


# Labels that strongly suggest the adjacent value is a date or numeric field.
_DATE_LABEL_KEYWORDS = re.compile(
    r"(?:年月日|日付|期日|期間|生年月日|年度|発行日|届出日|届出年月日|交付日|有効期限)",
)
_NUMERIC_LABEL_KEYWORDS = re.compile(
    r"(?:金額|数量|単価|合計|小計|税額|残高|番号|コード|No\.|NO\.|個数|回数|件数|面積|重量)",
)


def _detect_field_type(
    text: str,
    *,
    neighbor_labels: list[str] | None = None,
) -> str:
    """Classify OCR text as 'numeric', 'date', or 'text' for prompt tuning.

    Uses the OCR text itself as the primary signal, then uses neighboring
    label text (if available) to override or reinforce the classification
    when the OCR text alone is ambiguous.
    """
    stripped = text.strip()

    # --- primary detection from OCR text ---
    text_guess = "text"
    if stripped:
        # Japanese dates: may include era names (平成, 令和, etc.) plus 年月日
        if re.search(r"[年月日]", stripped) and re.match(
            r"^(?:(?:平成|令和|昭和|大正|明治)|[\d\s\-.,/年月日])+$",
            stripped,
        ):
            text_guess = "date"
        else:
            non_space = stripped.replace(" ", "")
            if non_space and sum(c.isdigit() for c in non_space) / len(non_space) > 0.6:
                text_guess = "numeric"

    if text_guess != "text":
        return text_guess

    # --- secondary detection from neighbor labels ---
    if neighbor_labels:
        label_text = " ".join(neighbor_labels)
        if _DATE_LABEL_KEYWORDS.search(label_text):
            return "date"
        if _NUMERIC_LABEL_KEYWORDS.search(label_text):
            return "numeric"

    return "text"


def _build_refine_prompt(
    box: TextBox,
    *,
    hint_mode: str = "full",
    neighbor_labels: list[str] | None = None,
    template_context: TemplatePromptContext | None = None,
) -> str:
    if template_context is not None:
        hint_mode = "none"
    orientation = "vertical" if effective_is_vertical(box) else "horizontal"
    field_type = None if template_context is not None else _detect_field_type(box.text, neighbor_labels=neighbor_labels)

    lines: list[str] = []

    hint_has_newline = "\n" in box.text

    if hint_mode == "none":
        lines.append("Read the text in this cropped region.")
        lines.append("Return only the exact text you see — keep the full text, do not truncate.")
        lines.append("Keep punctuation and spacing as shown.")
    elif hint_mode == "weak":
        lines.append("Read the text in this cropped region.")
        lines.append("Return only the text you actually see — keep the full text, do not truncate.")
        lines.append("Prioritize what you see over the OCR suggestion below.")
        lines.append(f"Previous OCR (may have errors): {box.text}")
    else:
        lines.append("Read the text in this cropped OCR box.")
        lines.append("Return only the corrected text — keep the full text, do not truncate.")
        lines.append("Keep punctuation and spacing natural.")
        lines.append("If the image is ambiguous, return the OCR hint unchanged.")
        lines.append(f"OCR hint: {box.text}")

    if template_context is not None:
        if template_context.get("label"):
            lines.append(f"Field: {template_context['label']}")
        if template_context.get("hint"):
            lines.append(f"Expected content: {template_context['hint']}")

    # Scope and shape constraints — prevent picking up adjacent rows and
    # emitting multi-line transcriptions that echo bleed-through text.
    lines.append(
        "The crop is padded and may include partial text from rows above or below; "
        "transcribe only the single line of text that is vertically centered in the crop "
        "and ignore any partial characters cut off at the top or bottom edges."
    )
    if hint_mode == "none":
        lines.append(
            "Return the transcription as a single line. Do not emit newline characters "
            "unless the region is clearly a natural multi-line block such as an address."
        )
    elif hint_has_newline:
        lines.append(
            "The OCR hint contains a line break, so you may keep at most that many lines; "
            "do not introduce additional newlines."
        )
    else:
        lines.append(
            "Return the transcription as a single line. Do not introduce newline characters — "
            "if you see two stacked rows of text, one of them is bleed from an adjacent box."
        )

    if field_type == "numeric":
        lines.append("This region likely contains numbers or codes. Transcribe each digit exactly.")
    elif field_type == "date":
        lines.append("This region likely contains a date. Transcribe it exactly as shown.")

    lines.append(f"Orientation: {orientation}")

    if neighbor_labels:
        lines.append(f"Context: {'; '.join(neighbor_labels)}")

    return "\n".join(lines)


def _normalize_box_text(text: str, *, fallback: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if len(lines) >= 3 and lines[-1].startswith("```"):
            cleaned = "\n".join(lines[1:-1]).strip()
    if cleaned.startswith('"') and cleaned.endswith('"') and len(cleaned) >= 2:
        cleaned = cleaned[1:-1].strip()
    # Strip obvious border pipes and digit separators without damaging normal text uses of '|'.
    if "|" in cleaned:
        if cleaned.startswith("|") and cleaned.endswith("|") and cleaned.count("|") >= 2:
            cleaned = cleaned.strip("|").strip()
        parts = [part.strip() for part in cleaned.split("|") if part.strip()]
        if len(parts) >= 3 and sum(len(part) <= 2 for part in parts) / len(parts) >= 0.75:
            cleaned = cleaned.replace("|", "").strip()
        else:
            cleaned = re.sub(r"(?<=\d)\|(?=\d)", "", cleaned)
        cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned or fallback


def _padding_amounts(
    box: TextBox,
    *,
    padding_ratio: float,
    padding_ratio_y: float | None,
) -> tuple[float, float]:
    """Compute LLM crop padding from the box short side.

    By default we add the same pixel padding on all four sides so that very
    wide boxes do not receive disproportionately large horizontal context.
    When ``padding_ratio_y`` is provided, only the vertical amount differs,
    but both directions still scale from the same short-side baseline.
    """
    short_side = max(1.0, min(float(box.width), float(box.height)))
    ratio_y = padding_ratio if padding_ratio_y is None else padding_ratio_y
    pad_x = max(4.0, short_side * padding_ratio)
    pad_y = max(4.0, short_side * ratio_y)
    return pad_x, pad_y


def _validate_box_text(
    text: str,
    box: TextBox,
    *,
    neighbor_labels: list[str] | None = None,
    template_context: TemplatePromptContext | None = None,
) -> str:
    cleaned = text.strip()
    if not cleaned:
        raise RuntimeError("LM Studio returned blank text.")

    non_empty_lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if len(non_empty_lines) >= 3:
        raise RuntimeError(f"LM Studio returned too many lines for one box ({len(non_empty_lines)} lines).")

    hint_length = len(box.text.strip())
    if template_context is not None:
        area_based_limit = round((box.width * box.height) / 80)
        max_reasonable_length = max(24, min(120, area_based_limit))
    elif hint_length:
        max_reasonable_length = max(24, hint_length * 4 + 8)
    else:
        area_based_limit = round((box.width * box.height) / 80)
        max_reasonable_length = max(24, min(120, area_based_limit))
    if len(cleaned) > max_reasonable_length:
        raise RuntimeError(
            "LM Studio returned text that is too long for one box "
            f"({len(cleaned)} chars > {max_reasonable_length} chars)."
        )

    field_type = None if template_context is not None else _detect_field_type(box.text, neighbor_labels=neighbor_labels)
    compact = cleaned.replace(" ", "")
    if field_type == "numeric" and compact:
        allowed_chars = set("0123456789,.-%¥\\()+-[]|/")
        allowed_ratio = sum(1 for char in compact if char in allowed_chars) / len(compact)
        if allowed_ratio < 0.6:
            raise RuntimeError(
                f"Numeric field but AI returned non-numeric content (allowed ratio {allowed_ratio:.2f}): {cleaned!r}"
            )
        digit_ratio = sum(1 for char in compact if char.isdigit()) / len(compact)
        if digit_ratio < 0.4:
            raise RuntimeError(
                f"Numeric field but AI returned too few digits (digit ratio {digit_ratio:.2f}): {cleaned!r}"
            )
    elif field_type == "date" and compact:
        allowed_chars = set("0123456789年月日/-.平成令和昭和大正明治元")
        allowed_ratio = sum(1 for char in compact if char in allowed_chars) / len(compact)
        if allowed_ratio < 0.6:
            raise RuntimeError(
                f"Date field but AI returned non-date content (allowed ratio {allowed_ratio:.2f}): {cleaned!r}"
            )
        if re.search(r"[A-Za-zァ-ヶぁ-ん]", cleaned):
            raise RuntimeError(f"Date field but AI returned alphabetic or kana content: {cleaned!r}")

    return cleaned


def _save_crop_image(data_url: str, dest: Path) -> None:
    """Decode a data:image/png;base64,... URL and write it to *dest*."""
    prefix = "data:image/png;base64,"
    if data_url.startswith(prefix):
        raw = base64.b64decode(data_url[len(prefix) :])
        dest.write_bytes(raw)


def _extract_message_text(payload: dict) -> str:
    choice = _extract_primary_choice(payload)
    message = choice.get("message", {})
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts)
    raise RuntimeError("LM Studio response content was not a supported type.")


def _extract_primary_choice(payload: dict) -> dict:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("LM Studio response did not contain any choices.")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise RuntimeError("LM Studio response choice was not an object.")
    return choice


def _request_json(
    url: str,
    *,
    payload: dict | None = None,
    timeout: float,
    api_key: str | None,
) -> dict:
    headers = {"Accept": "application/json"}
    data = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = request.Request(url, data=data, headers=headers, method="POST" if data is not None else "GET")
    try:
        with request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LM Studio request failed ({exc.code}): {body}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Could not reach LM Studio at {url}: {exc.reason}") from exc
