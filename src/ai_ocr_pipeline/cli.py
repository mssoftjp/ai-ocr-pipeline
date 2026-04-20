"""CLI entry point for ai-ocr-pipeline."""

from __future__ import annotations

import json
import shlex
import sys
import tempfile
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import click
import typer
from PIL import Image
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from ai_ocr_pipeline import __version__
from ai_ocr_pipeline.llm import (
    HINT_MODES,
    LLMRefinementStats,
    LMStudioConfig,
    refine_page_result_with_stats,
    resolve_model,
)
from ai_ocr_pipeline.models import PageResult, PageSerializationOptions, TextBox
from ai_ocr_pipeline.ocr import parse_ocr_json, run_direct_ocr, run_ocr, score_result
from ai_ocr_pipeline.overlay import write_overlay_artifact
from ai_ocr_pipeline.pdf import extract_pdf_text_layers, pdf_to_images
from ai_ocr_pipeline.preprocess import build_inverted_variant, build_line_removed_variant, ensure_rgb
from ai_ocr_pipeline.template import (
    NEWLINE_HANDLING_MODES,
    Template,
    build_ocr_evidence,
    build_template_prompt_contexts,
    decide_target_box_actions,
    load_template,
    template_to_page_result,
)

app = typer.Typer(
    name="ai-ocr-pipeline",
    help=(
        "Extract text boxes from images and PDFs as JSON with box geometry.\n\n"
        "AI refinement is on by default (local LM Studio). Use --ndl for OCR only."
    ),
    no_args_is_help=True,
    add_completion=False,
)
console = Console(stderr=True)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".jp2", ".bmp"}
PDF_EXTENSIONS = {".pdf"}
ENGINE_CHOICES = {"ndlocr-lite", "lmstudio-hybrid"}
LLM_BACKENDS = {"lmstudio"}
OCR_BACKENDS = {"direct", "subprocess"}


def _apply_newline_handling_to_text(text: str, newline_handling: str | None) -> str:
    if newline_handling is None or newline_handling == "preserve":
        return text

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    if newline_handling == "first_line":
        return normalized.split("\n", 1)[0]
    if newline_handling == "join":
        return normalized.replace("\n", "")
    raise ValueError(
        f"Unsupported newline handling: {newline_handling!r}. "
        f"Expected one of {', '.join(sorted(NEWLINE_HANDLING_MODES))}."
    )


def _apply_newline_handling(page_result: PageResult, newline_handling: str | None) -> PageResult:
    if newline_handling is None or newline_handling == "preserve":
        return page_result
    return replace(
        page_result,
        boxes=[
            replace(box, text=_apply_newline_handling_to_text(box.text, newline_handling)) for box in page_result.boxes
        ],
    )


def _effective_template_newline_handling(template_obj: Template | None) -> str | None:
    if template_obj is None:
        return None
    return template_obj.preprocess_newline_handling or "preserve"


@dataclass(frozen=True)
class _CandidateResult:
    image_path: Path
    result: PageResult
    variant_name: str
    llm_stats: LLMRefinementStats | None = None


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def _is_pdf(path: Path) -> bool:
    return path.suffix.lower() in PDF_EXTENSIONS


def _overlay_key(source: str, page: int | None) -> tuple[str, int | None]:
    return source, page


def _source_stem_and_ext(source: str) -> tuple[str, str]:
    path = Path(source)
    stem = path.stem or path.name
    ext = path.suffix.lower().lstrip(".") or "file"
    return stem, ext


def _bundle_stem(input_path: Path) -> str:
    return input_path.stem or input_path.name


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _mask_cli_args(argv: list[str]) -> list[str]:
    masked: list[str] = []
    skip_next = False
    for arg in argv:
        if skip_next:
            masked.append("***")
            skip_next = False
            continue
        if arg == "--llm-api-key":
            masked.append(arg)
            skip_next = True
            continue
        if arg.startswith("--llm-api-key="):
            masked.append("--llm-api-key=***")
            continue
        masked.append(arg)
    return masked


def _source_meta(input_path: Path) -> dict:
    try:
        stat = input_path.stat()
        file_size_bytes = stat.st_size
        modified_at = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).astimezone().isoformat(timespec="seconds")
    except OSError:
        file_size_bytes = None
        modified_at = None

    source_type = "directory" if input_path.is_dir() else ("pdf" if _is_pdf(input_path) else "image")
    return {
        "path": str(input_path),
        "type": source_type,
        "file_size_bytes": file_size_bytes,
        "modified_at": modified_at,
    }


def _image_meta(*, input_path: Path, dpi: int) -> dict:
    return {"dpi": dpi if _is_pdf(input_path) else None}


def _json_output_path(
    input_path: Path,
    *,
    output: Path | None,
    output_dir: Path | None,
) -> Path | None:
    if output is not None:
        return output
    if output_dir is not None:
        return output_dir / f"{_bundle_stem(input_path)}.json"
    return None


def _timestamped_run_dir(run_root: Path, input_path: Path, started_at: str) -> Path:
    parent = run_root / _bundle_stem(input_path)
    started = datetime.fromisoformat(started_at)
    base_name = started.strftime("%Y%m%d-%H%M%S")
    candidate = parent / base_name
    suffix = 1
    while candidate.exists():
        candidate = parent / f"{base_name}_{suffix:02d}"
        suffix += 1
    return candidate


def _overlay_output_root(
    input_path: Path,
    *,
    output: Path | None,
    output_dir: Path | None,
) -> Path:
    if output_dir is not None:
        return output_dir
    if output is not None:
        return output.parent
    if input_path.is_dir():
        return input_path / "ocr_overlays"
    return input_path.parent


def _overlay_output_path(
    result: PageResult,
    *,
    input_path: Path,
    output: Path | None,
    output_dir: Path | None,
) -> Path:
    root = _overlay_output_root(
        input_path,
        output=output,
        output_dir=output_dir,
    )
    prefix = output.stem if output is not None else _bundle_stem(input_path)
    source_stem, source_ext = _source_stem_and_ext(result.source)

    if result.page is not None:
        filename = f"{prefix}_p{result.page:04d}_overlay.png"
    elif input_path.is_dir():
        if output is not None:
            filename = f"{prefix}_{source_stem}_{source_ext}_overlay.png"
        else:
            filename = f"{source_stem}_{source_ext}_overlay.png"
    else:
        filename = f"{prefix}_overlay.png"
    return root / filename


def _write_overlay_outputs(
    results: list[PageResult],
    *,
    input_path: Path,
    output: Path | None,
    output_dir: Path | None,
    overlay_sources: dict[tuple[str, int | None], Path],
) -> None:
    _overlay_output_root(
        input_path,
        output=output,
        output_dir=output_dir,
    ).mkdir(
        parents=True,
        exist_ok=True,
    )
    for result in results:
        image_path = overlay_sources.get(_overlay_key(result.source, result.page))
        if image_path is None:
            console.print(
                f"[yellow]Overlay skipped for {result.source}"
                f"{'' if result.page is None else f' p.{result.page}'}: source image not found.[/yellow]"
            )
            continue

        overlay_path = _overlay_output_path(
            result,
            input_path=input_path,
            output=output,
            output_dir=output_dir,
        )
        actual_format, actual_path = write_overlay_artifact(result, image_path, overlay_path)
        if actual_format == "png":
            console.print(f"[green]Overlay written to {actual_path}[/green]")
        else:
            console.print(f"[yellow]PNG renderer unavailable. SVG overlay written to {actual_path}[/yellow]")


def _resolve_overlay_setting(
    overlay: bool | None,
    *,
    output: Path | None,
    output_dir: Path | None,
) -> bool:
    if overlay is not None:
        return overlay
    return output_dir is not None


def _resolve_pretty_setting(
    pretty: bool | None,
    *,
    output: Path | None,
    output_dir: Path | None,
) -> bool:
    if pretty is not None:
        return pretty
    if output is not None or output_dir is not None:
        return True
    return sys.stdout.isatty()


def _resolve_include_absolute_geometry(
    include_absolute_geometry: bool | None,
    *,
    template_obj: Template | None,
) -> bool:
    if include_absolute_geometry is not None:
        return include_absolute_geometry
    return template_obj is None


def _resolve_include_debug_fields(
    include_debug_fields: bool | None,
    *,
    template_obj: Template | None,
) -> bool:
    if include_debug_fields is not None:
        return include_debug_fields
    return template_obj is None


def _commandline_parameter_names(*names: str) -> set[str]:
    context = click.get_current_context(silent=True)
    if context is None:
        return set()

    commandline = click.core.ParameterSource.COMMANDLINE
    return {name for name in names if context.get_parameter_source(name) is commandline}


def _parse_int_csv_option(value: str, *, option_name: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise ValueError(f"Invalid --{option_name}: {value!r}. Use comma-separated integers.") from exc
    if not parsed:
        raise ValueError(f"Invalid --{option_name}: {value!r}. Specify at least one integer.")
    return parsed


def _prepare_single_image(
    image_path: Path,
    work_dir: Path,
    *,
    deskew: bool = False,
    remove_horizontal_lines: bool = False,
    remove_vertical_lines: bool = False,
) -> Path:
    """Prepare a single image path for template-mode processing."""
    prepared_path = ensure_rgb(image_path, work_dir)
    if remove_horizontal_lines or remove_vertical_lines:
        prepared_path = build_line_removed_variant(
            prepared_path,
            work_dir,
            remove_horizontal_lines=remove_horizontal_lines,
            remove_vertical_lines=remove_vertical_lines,
            invert_output=False,
        )
    if deskew:
        from ai_ocr_pipeline.preprocess.deskew import deskew_image

        prepared_path = deskew_image(prepared_path, work_dir / f"deskewed_{prepared_path.stem}.png")
    return prepared_path


def _template_box_crop_bounds(box: TextBox) -> tuple[int, int, int, int]:
    left = max(0, round(box.x))
    top = max(0, round(box.y))
    right = max(left + 1, left + box.width)
    bottom = max(top + 1, top + box.height)
    return left, top, right, bottom


def _is_low_ink(crop_image: Image.Image, *, threshold: float = 0.005) -> bool:
    import cv2
    import numpy as np

    gray = np.array(crop_image.convert("L"))
    _thr, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _thr, binary_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink_ratio = min(
        float(np.count_nonzero(binary)) / float(binary.size),
        float(np.count_nonzero(binary_inv)) / float(binary_inv.size),
    )
    return ink_ratio < threshold


def _build_template_crop_ocr_evidence(
    image_path: Path,
    work_dir: Path,
    page_result: PageResult,
    *,
    device: str,
    source_name: str,
    page: int | None,
    ocr_backend: str,
    filter_container_fallbacks: bool,
    split_wide_lines: bool,
) -> PageResult:
    """Run OCR on strict template-owned crops and attach the results as evidence."""
    ocr_results_by_index: dict[int, PageResult] = {}
    low_ink_by_index: dict[int, bool] = {}
    crops_dir = work_dir / "template_ocr_crops"
    crops_dir.mkdir(exist_ok=True)

    with Image.open(image_path) as image:
        img_width, img_height = image.size
        for index, box in enumerate(page_result.boxes):
            left, top, right, bottom = _template_box_crop_bounds(box)
            right = min(img_width, right)
            bottom = min(img_height, bottom)
            crop = image.crop((left, top, right, bottom))
            crop_path = crops_dir / f"box_{index:04d}.png"
            crop.save(crop_path, format="PNG")
            low_ink_by_index[index] = _is_low_ink(crop)
            box_work_dir = work_dir / f"template_box_{index:04d}"
            box_work_dir.mkdir(exist_ok=True)
            ocr_results_by_index[index] = _run_ocr_for_image(
                crop_path,
                box_work_dir,
                device=device,
                source_name=source_name,
                page=page,
                ocr_backend=ocr_backend,
                filter_container_fallbacks=filter_container_fallbacks,
                split_wide_lines=split_wide_lines,
            )

    return build_ocr_evidence(page_result, ocr_results_by_index, low_ink_by_index=low_ink_by_index)


def _run_ocr_for_image(
    image_path: Path,
    work_dir: Path,
    *,
    device: str,
    source_name: str,
    page: int | None,
    recognition_image_path: Path | None = None,
    ocr_backend: str = "direct",
    filter_container_fallbacks: bool = True,
    split_wide_lines: bool = True,
) -> PageResult:
    """Run OCR for a prepared image file and parse the result."""
    if ocr_backend == "direct":
        return run_direct_ocr(
            image_path,
            recognition_image_path=recognition_image_path,
            source=source_name or image_path.name,
            page=page,
            device=device,
            filter_container_fallbacks=filter_container_fallbacks,
            split_wide_lines=split_wide_lines,
        )

    ocr_output_dir = work_dir / f"ocr_output_{image_path.stem}"
    ocr_output_dir.mkdir(exist_ok=True)
    json_path = run_ocr(image_path, ocr_output_dir, device=device)
    return parse_ocr_json(
        json_path,
        source=source_name or image_path.name,
        page=page,
    )


def _process_template_image(
    image_path: Path,
    work_dir: Path,
    *,
    template_obj: Template,
    deskew: bool = False,
    remove_horizontal_lines: bool = False,
    remove_vertical_lines: bool = False,
    device: str = "cpu",
    source_name: str = "",
    page: int | None = None,
    box_ids: tuple[int, ...] | None = None,
    lmstudio_config: LMStudioConfig,
    ocr_backend: str = "direct",
    filter_container_fallbacks: bool = True,
    split_wide_lines: bool = True,
) -> _CandidateResult:
    """Process one image through template-mode refinement."""
    prepared_path = _prepare_single_image(
        image_path,
        work_dir,
        deskew=deskew,
        remove_horizontal_lines=remove_horizontal_lines,
        remove_vertical_lines=remove_vertical_lines,
    )
    with Image.open(prepared_path) as prepared_image:
        img_width, img_height = prepared_image.size

    page_result = template_to_page_result(
        template_obj,
        img_width,
        img_height,
        source_name or image_path.name,
        page,
        box_ids,
    )
    page_result = _build_template_crop_ocr_evidence(
        prepared_path,
        work_dir,
        page_result,
        device=device,
        source_name=source_name or image_path.name,
        page=page,
        ocr_backend=ocr_backend,
        filter_container_fallbacks=filter_container_fallbacks,
        split_wide_lines=split_wide_lines,
    )
    page_result, refine_indices = decide_target_box_actions(
        page_result,
        confidence_threshold=lmstudio_config.confidence_threshold,
    )
    prompt_contexts = build_template_prompt_contexts(template_obj, box_ids)
    refined_result, llm_stats = refine_page_result_with_stats(
        prepared_path,
        page_result,
        lmstudio_config,
        template_contexts=prompt_contexts,
        selected_box_indices=refine_indices,
    )
    refined_result = _apply_newline_handling(refined_result, template_obj.preprocess_newline_handling)
    unresolved_boxes = [box for box in refined_result.boxes if box.text_source not in {"llm", "blank_skip", "ocr"}]
    if unresolved_boxes:
        unresolved_orders = ", ".join(str(box.order) for box in unresolved_boxes if box.order is not None)
        raise RuntimeError(
            "Template refinement did not produce text for all boxes"
            + ("" if not unresolved_orders else f" (box ids: {unresolved_orders})")
            + "."
        )
    return _CandidateResult(
        image_path=prepared_path,
        variant_name="template",
        result=refined_result,
        llm_stats=llm_stats,
    )


def _process_image(
    image_path: Path,
    work_dir: Path,
    *,
    deskew: bool = False,
    remove_horizontal_lines: bool = False,
    remove_vertical_lines: bool = False,
    device: str = "cpu",
    source_name: str = "",
    page: int | None = None,
    engine: str = "ndlocr-lite",
    lmstudio_config: LMStudioConfig | None = None,
    ocr_backend: str = "direct",
    filter_container_fallbacks: bool = True,
    split_wide_lines: bool = True,
) -> _CandidateResult:
    """Process a single image through the pipeline."""
    actual_path = ensure_rgb(image_path, work_dir)
    inverted_path = build_inverted_variant(image_path, work_dir)
    natural_image_path = actual_path
    candidate_specs: list[tuple[str, Path, Path]] = [
        ("natural", actual_path, actual_path),
        ("inverted", inverted_path, actual_path),
    ]

    if remove_horizontal_lines or remove_vertical_lines:
        line_removed_path = build_line_removed_variant(
            image_path,
            work_dir,
            remove_horizontal_lines=remove_horizontal_lines,
            remove_vertical_lines=remove_vertical_lines,
            invert_output=False,
        )
        line_removed_inverted_path = build_line_removed_variant(
            image_path,
            work_dir,
            remove_horizontal_lines=remove_horizontal_lines,
            remove_vertical_lines=remove_vertical_lines,
            invert_output=True,
        )
        candidate_specs.append(
            (
                "line_removed",
                line_removed_path,
                line_removed_path,
            )
        )
        candidate_specs.append(
            (
                "line_removed_inverted",
                line_removed_inverted_path,
                line_removed_path,
            )
        )
    if deskew:
        from ai_ocr_pipeline.preprocess.deskew import deskew_image

        deskew_cache: dict[Path, Path] = {}

        def _deskew_once(path: Path) -> Path:
            cached = deskew_cache.get(path)
            if cached is None:
                cached = deskew_image(path, work_dir / f"deskewed_{path.stem}.png")
                deskew_cache[path] = cached
            return cached

        candidate_specs = [
            (name, _deskew_once(path), _deskew_once(recognition_path))
            for name, path, recognition_path in candidate_specs
        ]
        natural_image_path = candidate_specs[0][2]

    candidates = [
        _CandidateResult(
            image_path=path,
            variant_name=variant_name,
            result=_run_ocr_for_image(
                path,
                work_dir,
                device=device,
                source_name=source_name,
                page=page,
                recognition_image_path=recognition_path,
                ocr_backend=ocr_backend,
                filter_container_fallbacks=filter_container_fallbacks,
                split_wide_lines=split_wide_lines,
            ),
        )
        for variant_name, path, recognition_path in candidate_specs
    ]
    best = max(candidates, key=lambda candidate: score_result(candidate.result))
    if engine == "lmstudio-hybrid":
        if lmstudio_config is None:
            raise ValueError("lmstudio_config is required when engine='lmstudio-hybrid'.")
        try:
            refined_result, llm_stats = refine_page_result_with_stats(natural_image_path, best.result, lmstudio_config)
            return _CandidateResult(
                image_path=best.image_path,
                variant_name=best.variant_name,
                result=refined_result,
                llm_stats=llm_stats,
            )
        except Exception as exc:
            console.print(f"[yellow]LM Studio refinement failed, using original OCR text: {exc}[/yellow]")
            return best
    return best


@app.command()
def run(
    input_path: Path = typer.Argument(
        ...,
        help="Image, PDF, or directory of images.",
        exists=True,
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Write JSON to file.",
        rich_help_panel="Output",
    ),
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        "-d",
        help="Bundle JSON + overlays in a directory.",
        rich_help_panel="Output",
    ),
    run_root: Path | None = typer.Option(
        None,
        "--run-root",
        help="Create a timestamped bundle at ROOT/<input_name>/<YYYYMMDD-HHMMSS>/.",
        rich_help_panel="Output",
    ),
    overlay: bool | None = typer.Option(
        None,
        "--overlay/--no-overlay",
        help="Write overlay images.",
        rich_help_panel="Output",
    ),
    deskew: bool = typer.Option(
        False,
        "--deskew/--no-deskew",
        help="Straighten skewed scans.",
        rich_help_panel="Preprocessing",
    ),
    remove_horizontal_lines: bool = typer.Option(
        False,
        "--remove-horizontal-lines/--no-remove-horizontal-lines",
        help="Remove long horizontal form/table lines.",
        rich_help_panel="Preprocessing",
    ),
    remove_vertical_lines: bool = typer.Option(
        False,
        "--remove-vertical-lines/--no-remove-vertical-lines",
        help="Remove long vertical form/table lines.",
        rich_help_panel="Preprocessing",
    ),
    dpi: int = typer.Option(
        600,
        "--dpi",
        help="PDF rasterization DPI.",
        rich_help_panel="Preprocessing",
    ),
    prefer_text_layer: bool = typer.Option(
        False,
        "--prefer-text-layer/--no-prefer-text-layer",
        help="Use PDF embedded text when explicitly requested.",
        rich_help_panel="Preprocessing",
    ),
    device: str = typer.Option(
        "cpu",
        "--device",
        help="Inference device: cpu or cuda.",
        rich_help_panel="Preprocessing",
    ),
    ocr_backend: str = typer.Option(
        "direct",
        "--ocr-backend",
        click_type=click.Choice(sorted(OCR_BACKENDS), case_sensitive=False),
        help="'direct' imports ndlocr-lite in-process (faster, allows post-fixes); 'subprocess' launches the CLI.",
        rich_help_panel="Preprocessing",
    ),
    filter_container_fallbacks: bool = typer.Option(
        True,
        "--ocr-filter-container-fallbacks/--no-ocr-filter-container-fallbacks",
        help="Drop oversized LINEs emitted when a text_block/block_table has no detected child lines. Direct backend only.",
        rich_help_panel="Preprocessing",
    ),
    split_wide_lines: bool = typer.Option(
        True,
        "--ocr-split-wide-lines/--no-ocr-split-wide-lines",
        help="Split oversized horizontal LINEs at whitespace gaps so table rows become per-cell boxes. Direct backend only.",
        rich_help_panel="Preprocessing",
    ),
    template: Path | None = typer.Option(
        None,
        "--template",
        help="Template JSON defining box positions to read.",
        rich_help_panel="Template",
    ),
    template_boxes: str | None = typer.Option(
        None,
        "--template-boxes",
        help="Process only these template box IDs. E.g. '1,3,5'.",
        rich_help_panel="Template",
    ),
    use_lmstudio: bool = typer.Option(
        False,
        "--lmstudio",
        help="Use local LM Studio for AI refinement (error on connection failure).",
        rich_help_panel="AI Backend",
    ),
    use_openai: bool = typer.Option(
        False,
        "--openai",
        help="Use OpenAI API for AI refinement (not yet implemented).",
        rich_help_panel="AI Backend",
    ),
    use_gemini: bool = typer.Option(
        False,
        "--gemini",
        help="Use Gemini API for AI refinement (not yet implemented).",
        rich_help_panel="AI Backend",
    ),
    use_ndl: bool = typer.Option(
        False,
        "--ndl",
        help="OCR only — skip AI refinement.",
        rich_help_panel="AI Backend",
    ),
    engine: str | None = typer.Option(
        None,
        "--engine",
        help="Deprecated. Use --lmstudio instead.",
        hidden=True,
    ),
    llm: str | None = typer.Option(
        None,
        "--llm",
        click_type=click.Choice(sorted(LLM_BACKENDS), case_sensitive=False),
        help="Deprecated. Use --lmstudio instead.",
        hidden=True,
    ),
    llm_base_url: str = typer.Option(
        "http://127.0.0.1:1234/v1",
        "--llm-base-url",
        envvar="LMSTUDIO_BASE_URL",
        help="Endpoint URL.",
        rich_help_panel="LLM Refinement",
    ),
    llm_model: str | None = typer.Option(
        None,
        "--llm-model",
        envvar="LMSTUDIO_MODEL",
        help="Model ID.",
        rich_help_panel="LLM Refinement",
    ),
    llm_api_key: str | None = typer.Option(
        None,
        "--llm-api-key",
        envvar="LMSTUDIO_API_KEY",
        help="API key.",
        rich_help_panel="LLM Refinement",
    ),
    llm_timeout: float = typer.Option(
        120.0,
        "--llm-timeout",
        help="Request timeout in seconds.",
        rich_help_panel="LLM Refinement",
    ),
    llm_max_tokens: int = typer.Option(
        4096,
        "--llm-max-tokens",
        help="Max completion tokens per LLM request.",
        rich_help_panel="LLM Refinement",
    ),
    llm_hint_mode: str = typer.Option(
        "full",
        "--llm-hint-mode",
        help="OCR hint: full, weak, or none.",
        rich_help_panel="LLM Refinement",
    ),
    llm_crop_padding: float = typer.Option(
        0.25,
        "--llm-crop-padding",
        help="Box crop padding ratio, applied from the box short side.",
        rich_help_panel="Debug",
    ),
    llm_confidence_threshold: float | None = typer.Option(
        None,
        "--llm-confidence-threshold",
        help="Skip LLM refinement for boxes with confidence >= this value (e.g. 0.9).",
        rich_help_panel="LLM Refinement",
    ),
    llm_context_confidence: float = typer.Option(
        0.5,
        "--llm-context-confidence",
        help="Add neighbor label context only for boxes with confidence below this value.",
        rich_help_panel="LLM Refinement",
    ),
    llm_max_workers: int = typer.Option(
        16,
        "--llm-max-workers",
        help="Max parallel LLM requests per page.",
        rich_help_panel="LLM Refinement",
    ),
    llm_box_indices: str | None = typer.Option(
        None,
        "--llm-box-indices",
        help="Refine specific boxes. E.g. '0,3,7'.",
        rich_help_panel="Debug",
    ),
    llm_save_crops: Path | None = typer.Option(
        None,
        "--llm-save-crops",
        help="Save crop images for inspection.",
        rich_help_panel="Debug",
    ),
    include_absolute_geometry: bool | None = typer.Option(
        None,
        "--include-absolute-geometry/--no-include-absolute-geometry",
        help="Include pixel geometry in JSON output. Template mode defaults to off.",
        rich_help_panel="Output",
    ),
    include_debug_fields: bool | None = typer.Option(
        None,
        "--include-debug-fields/--no-include-debug-fields",
        help="Include OCR evidence and other debug-oriented fields. Template mode defaults to off.",
        rich_help_panel="Output",
    ),
    pretty: bool | None = typer.Option(
        None,
        "--pretty/--no-pretty",
        help="Pretty-print JSON with indentation, or force compact single-line JSON.",
        rich_help_panel="Output",
    ),
) -> None:
    """Extract text boxes from images and PDFs as JSON with box geometry.

    AI refinement is on by default (local LM Studio). Use --ndl for OCR only.
    """
    run_started_at = _now_iso()
    run_started_monotonic = time.monotonic()

    if sum(value is not None for value in (output, output_dir, run_root)) > 1:
        console.print(
            "[red]--output, --output-dir, and --run-root are mutually exclusive.[/red]\n"
            "Use -o for JSON-only output, -d for an explicit bundle directory, or --run-root for timestamped bundles."
        )
        raise typer.Exit(1)

    if run_root is not None:
        output_dir = _timestamped_run_dir(run_root, input_path, run_started_at)

    template_obj: Template | None = None
    parsed_template_boxes: tuple[int, ...] | None = None
    if template is not None:
        try:
            template_obj = load_template(template)
        except ValueError as exc:
            console.print(f"[red]Invalid --template: {exc}[/red]")
            raise typer.Exit(1) from exc
        if template_boxes is not None:
            try:
                parsed_template_boxes = _parse_int_csv_option(template_boxes, option_name="template-boxes")
            except ValueError as exc:
                console.print(f"[red]{exc}[/red]")
                raise typer.Exit(1) from exc
        try:
            build_template_prompt_contexts(template_obj, parsed_template_boxes)
        except ValueError as exc:
            console.print(f"[red]Invalid --template-boxes: {exc}[/red]")
            raise typer.Exit(1) from exc

    explicit_preprocess_params = _commandline_parameter_names(
        "deskew",
        "remove_horizontal_lines",
        "remove_vertical_lines",
    )
    if template_obj is not None:
        if "deskew" not in explicit_preprocess_params and template_obj.preprocess_deskew is not None:
            deskew = template_obj.preprocess_deskew
        if (
            "remove_horizontal_lines" not in explicit_preprocess_params
            and template_obj.preprocess_remove_horizontal_lines is not None
        ):
            remove_horizontal_lines = template_obj.preprocess_remove_horizontal_lines
        if (
            "remove_vertical_lines" not in explicit_preprocess_params
            and template_obj.preprocess_remove_vertical_lines is not None
        ):
            remove_vertical_lines = template_obj.preprocess_remove_vertical_lines

    # --- Resolve AI mode from flags ---
    new_flags = sum([use_lmstudio, use_openai, use_gemini, use_ndl])
    if new_flags > 1:
        console.print("[red]--lmstudio, --openai, --gemini, and --ndl are mutually exclusive.[/red]")
        raise typer.Exit(1)

    deprecated_flag = engine is not None or llm is not None
    if new_flags and deprecated_flag:
        console.print("[red]Cannot combine --lmstudio/--openai/--gemini/--ndl with deprecated --llm or --engine.[/red]")
        raise typer.Exit(1)

    ai_mode: str  # "lmstudio", "openai", "gemini", "ndl", or "auto"
    if use_lmstudio:
        ai_mode = "lmstudio"
    elif use_openai:
        ai_mode = "openai"
    elif use_gemini:
        ai_mode = "gemini"
    elif use_ndl:
        ai_mode = "ndl"
    elif engine is not None or llm is not None:
        if engine is not None and llm is not None:
            console.print("[red]--engine is deprecated. Use --lmstudio instead. Do not combine both.[/red]")
            raise typer.Exit(1)
        if engine == "lmstudio-hybrid":
            console.print("[yellow]--engine lmstudio-hybrid is deprecated. Use --lmstudio instead.[/yellow]")
            ai_mode = "lmstudio"
        elif engine is not None:
            console.print(f"[red]Unsupported engine: {engine}[/red]")
            raise typer.Exit(1)
        else:
            console.print("[yellow]--llm lmstudio is deprecated. Use --lmstudio instead.[/yellow]")
            ai_mode = "lmstudio"
    else:
        ai_mode = "auto"

    if template_obj is not None:
        if ai_mode == "ndl":
            console.print("[red]Template mode requires AI refinement. Remove --ndl or add --lmstudio.[/red]")
            raise typer.Exit(1)
        if ai_mode == "auto":
            ai_mode = "lmstudio"

    overlay_enabled = _resolve_overlay_setting(
        overlay,
        output=output,
        output_dir=output_dir,
    )
    pretty_enabled = _resolve_pretty_setting(
        pretty,
        output=output,
        output_dir=output_dir,
    )
    include_absolute_geometry_enabled = _resolve_include_absolute_geometry(
        include_absolute_geometry,
        template_obj=template_obj,
    )
    include_debug_fields_enabled = _resolve_include_debug_fields(
        include_debug_fields,
        template_obj=template_obj,
    )

    if ai_mode in ("openai", "gemini"):
        console.print(f"[red]--{ai_mode} backend is not yet implemented.[/red]")
        raise typer.Exit(1)

    if template_obj is not None and llm_box_indices is not None:
        console.print("[red]--llm-box-indices cannot be combined with --template-boxes/--template.[/red]")
        raise typer.Exit(1)

    explicit_llm_params = _commandline_parameter_names(
        "llm_base_url",
        "llm_model",
        "llm_api_key",
        "llm_timeout",
        "llm_max_tokens",
        "llm_hint_mode",
        "llm_crop_padding",
        "llm_confidence_threshold",
        "llm_context_confidence",
        "llm_max_workers",
        "llm_box_indices",
        "llm_save_crops",
    )
    lmstudio_config: LMStudioConfig | None = None
    ai_backend_used: str | None = None

    if ai_mode == "ndl":
        if explicit_llm_params:
            console.print(
                "[yellow]--llm-* options ignored in --ndl mode: "
                + ", ".join(f"--{name.replace('_', '-')}" for name in sorted(explicit_llm_params))
                + "[/yellow]"
            )
    elif ai_mode in ("lmstudio", "auto"):
        if llm_hint_mode not in HINT_MODES:
            console.print(f"[red]Unsupported hint mode: {llm_hint_mode}[/red]\nSupported: {', '.join(HINT_MODES)}")
            raise typer.Exit(1)

        parsed_box_indices: tuple[int, ...] | None = None
        if llm_box_indices is not None:
            try:
                parsed_box_indices = _parse_int_csv_option(llm_box_indices, option_name="llm-box-indices")
            except ValueError as exc:
                console.print(f"[red]{exc}[/red]")
                raise typer.Exit(1) from exc

        try:
            lmstudio_config = LMStudioConfig(
                base_url=llm_base_url,
                model=llm_model,
                api_key=llm_api_key,
                timeout=llm_timeout if ai_mode == "lmstudio" else min(llm_timeout, 5.0),
                max_tokens_per_request=llm_max_tokens,
                hint_mode=llm_hint_mode,
                crop_padding_ratio=llm_crop_padding,
                confidence_threshold=llm_confidence_threshold,
                context_confidence=llm_context_confidence,
                max_workers=llm_max_workers,
                box_indices=parsed_box_indices,
                save_crops_dir=str(llm_save_crops) if llm_save_crops else None,
            )
            lmstudio_config = replace(
                lmstudio_config,
                model=resolve_model(lmstudio_config),
                timeout=llm_timeout,
            )
            ai_backend_used = "lmstudio"
            console.print(f"AI refinement: lmstudio (model: {lmstudio_config.model}, url: {lmstudio_config.base_url})")
        except Exception as exc:
            if ai_mode == "lmstudio":
                console.print(f"[red]Failed to connect to LM Studio: {exc}[/red]")
                raise typer.Exit(1) from exc
            console.print(f"[yellow]LM Studio not available ({exc}). Falling back to OCR-only mode.[/yellow]")
            lmstudio_config = None

    results: list[PageResult] = []
    processed_candidates: list[_CandidateResult] = []
    overlay_sources: dict[tuple[str, int | None], Path] = {}

    with tempfile.TemporaryDirectory(prefix="ai_ocr_pipeline_") as tmpdir:
        work_dir = Path(tmpdir)

        # Collect input files
        if input_path.is_dir():
            image_files = sorted(f for f in input_path.iterdir() if _is_image(f))
            if not image_files:
                console.print(f"[red]No image files found in {input_path}[/red]")
                raise typer.Exit(1)
            file_list: list[tuple[Path, str, int | None]] = [(f, f.name, None) for f in image_files]
            if overlay_enabled:
                overlay_sources.update(
                    {_overlay_key(source_name, page_num): img_path for img_path, source_name, page_num in file_list}
                )

        elif _is_pdf(input_path):
            page_results: dict[int, PageResult] = {}
            missing_pages: list[int] = []

            if template_obj is not None:
                import pypdfium2 as pdfium

                pdf = pdfium.PdfDocument(input_path)
                missing_pages = list(range(1, len(pdf) + 1))
                pdf.close()
            elif prefer_text_layer:
                text_layer_pages = extract_pdf_text_layers(input_path, dpi=dpi)
                text_layer_count = sum(1 for page in text_layer_pages if page is not None)
                if text_layer_count:
                    console.print(
                        f"Using embedded PDF text layer for {text_layer_count}/{len(text_layer_pages)} page(s)."
                    )
                for page_number, page_result in enumerate(text_layer_pages, start=1):
                    if page_result is None:
                        missing_pages.append(page_number)
                    else:
                        page_results[page_number] = page_result
            else:
                import pypdfium2 as pdfium

                pdf = pdfium.PdfDocument(input_path)
                missing_pages = list(range(1, len(pdf) + 1))
                pdf.close()

            if missing_pages:
                console.print(f"Converting {len(missing_pages)} PDF page(s) to images (DPI={dpi})...")
                pdf_img_dir = work_dir / "pdf_images"
                pdf_img_dir.mkdir()
                image_files = pdf_to_images(
                    input_path,
                    pdf_img_dir,
                    dpi=dpi,
                    page_numbers=missing_pages,
                )
                if not image_files and not page_results:
                    console.print("[red]No pages found in PDF.[/red]")
                    raise typer.Exit(1)
                console.print(
                    f"  {len(image_files)} page(s) extracted for {'template processing' if template_obj is not None else 'OCR'}."
                )
                file_list = [
                    (f, input_path.name, page_num) for f, page_num in zip(image_files, missing_pages, strict=False)
                ]
            else:
                file_list = []

            if page_results:
                results.extend(page_results[page_num] for page_num in sorted(page_results))

        elif _is_image(input_path):
            file_list = [(input_path, input_path.name, None)]
            if overlay_enabled:
                overlay_sources[_overlay_key(input_path.name, None)] = input_path

        else:
            console.print(
                f"[red]Unsupported file type: {input_path.suffix}[/red]\n"
                f"Supported: {', '.join(sorted(IMAGE_EXTENSIONS | PDF_EXTENSIONS))}"
            )
            raise typer.Exit(1)

        # Process each file
        if file_list:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Processing...", total=len(file_list))
                for img_path, source_name, page_num in file_list:
                    desc = f"{'Template' if template_obj is not None else 'OCR'}: {source_name}"
                    if page_num is not None:
                        desc += f" (p.{page_num})"
                    progress.update(task, description=desc)

                    page_work_dir = work_dir / f"page_{page_num or 0}"
                    page_work_dir.mkdir(exist_ok=True)

                    if template_obj is not None:
                        assert lmstudio_config is not None
                        result = _process_template_image(
                            img_path,
                            page_work_dir,
                            template_obj=template_obj,
                            deskew=deskew,
                            remove_horizontal_lines=remove_horizontal_lines,
                            remove_vertical_lines=remove_vertical_lines,
                            device=device,
                            source_name=source_name,
                            page=page_num,
                            box_ids=parsed_template_boxes,
                            lmstudio_config=lmstudio_config,
                            ocr_backend=ocr_backend,
                            filter_container_fallbacks=filter_container_fallbacks,
                            split_wide_lines=split_wide_lines,
                        )
                    else:
                        result = _process_image(
                            img_path,
                            page_work_dir,
                            deskew=deskew,
                            remove_horizontal_lines=remove_horizontal_lines,
                            remove_vertical_lines=remove_vertical_lines,
                            device=device,
                            source_name=source_name,
                            page=page_num,
                            engine="lmstudio-hybrid" if lmstudio_config is not None else "ndlocr-lite",
                            lmstudio_config=lmstudio_config,
                            ocr_backend=ocr_backend,
                            filter_container_fallbacks=filter_container_fallbacks,
                            split_wide_lines=split_wide_lines,
                        )
                    processed_candidates.append(result)
                    results.append(result.result)
                    if overlay_enabled:
                        overlay_sources[_overlay_key(source_name, page_num)] = result.image_path
                    progress.advance(task)

        results.sort(key=lambda result: (result.page is None, result.page or 0, result.source))

        if overlay_enabled and results:
            if _is_pdf(input_path):
                missing_overlay_pages: list[int] = []
                for result in results:
                    key = _overlay_key(result.source, result.page)
                    if key in overlay_sources:
                        continue
                    if result.page is None:
                        console.print(f"[yellow]Overlay skipped for {result.source}: page number is missing.[/yellow]")
                        continue
                    missing_overlay_pages.append(result.page)

                if missing_overlay_pages:
                    overlay_pdf_dir = work_dir / "overlay_pdf_images"
                    overlay_pdf_dir.mkdir(exist_ok=True)
                    page_numbers = sorted(set(missing_overlay_pages))
                    page_images = pdf_to_images(
                        input_path,
                        overlay_pdf_dir,
                        dpi=dpi,
                        page_numbers=page_numbers,
                    )
                    overlay_sources.update(
                        {
                            _overlay_key(input_path.name, page_num): image_path
                            for image_path, page_num in zip(page_images, page_numbers, strict=False)
                        }
                    )
            _write_overlay_outputs(
                results,
                input_path=input_path,
                output=output,
                output_dir=output_dir,
                overlay_sources=overlay_sources,
            )

    run_finished_at = _now_iso()
    duration_ms = round((time.monotonic() - run_started_monotonic) * 1000)
    masked_arguments = _mask_cli_args(sys.argv[1:])
    variant_by_result = {
        _overlay_key(candidate.result.source, candidate.result.page): candidate.variant_name
        for candidate in processed_candidates
    }
    serialization_options = PageSerializationOptions(
        include_absolute_geometry=include_absolute_geometry_enabled,
        include_debug_fields=include_debug_fields_enabled,
    )
    serialized_results = []
    for result in results:
        serialized = result.to_dict(serialization_options)
        if template_obj is None or include_debug_fields_enabled:
            serialized["ocr_image_variant"] = variant_by_result.get(_overlay_key(result.source, result.page))
        serialized_results.append(serialized)
    llm_stats = LLMRefinementStats()
    for candidate in processed_candidates:
        llm_stats.merge(candidate.llm_stats)

    # Output
    run_meta = {
        "started_at": run_started_at,
        "finished_at": run_finished_at,
        "duration_ms": duration_ms,
        "command": shlex.join([Path(sys.argv[0]).name, *masked_arguments]),
    }
    if template_obj is None or include_debug_fields_enabled:
        run_meta["arguments"] = masked_arguments

    preprocess_meta = {
        "deskew": deskew,
        "remove_horizontal_lines": remove_horizontal_lines,
        "remove_vertical_lines": remove_vertical_lines,
        "newline_handling": _effective_template_newline_handling(template_obj),
    }
    if template_obj is None or include_debug_fields_enabled:
        preprocess_meta.update(
            {
                "prefer_text_layer": False if template_obj is not None else prefer_text_layer,
                "llm_image_variant": "template"
                if template_obj is not None
                else ("natural" if lmstudio_config is not None else None),
                "used_inverted_for_llm": False if lmstudio_config is not None else None,
            }
        )

    ocr_meta = {
        "engine": "template" if template_obj is not None else "ndlocr-lite",
        "backend": ocr_backend,
    }
    if template_obj is None or include_debug_fields_enabled:
        ocr_meta.update(
            {
                "engine_version": None,
                "device": device,
                "filter_container_fallbacks": (filter_container_fallbacks if ocr_backend == "direct" else None),
                "split_wide_lines": split_wide_lines if ocr_backend == "direct" else None,
            }
        )

    template_meta = (
        {
            "name": template_obj.name,
            "path": str(template),
            "coordinate_mode": template_obj.coordinate_mode,
            "box_count": len(template_obj.boxes),
            "box_ids_requested": list(parsed_template_boxes) if parsed_template_boxes is not None else None,
        }
        if template_obj is not None
        else None
    )
    if template_meta is not None and include_debug_fields_enabled:
        template_meta["preprocess"] = {
            "deskew": template_obj.preprocess_deskew,
            "remove_horizontal_lines": template_obj.preprocess_remove_horizontal_lines,
            "remove_vertical_lines": template_obj.preprocess_remove_vertical_lines,
            "newline_handling": template_obj.preprocess_newline_handling,
        }

    output_data = {
        "meta": {
            "schema_version": "2",
            "tool_version": __version__,
            "run": run_meta,
            "source": _source_meta(input_path),
            "image": _image_meta(input_path=input_path, dpi=dpi),
            "preprocess": preprocess_meta,
            "ocr": ocr_meta,
            "template": template_meta,
            "llm": (
                {
                    "backend": ai_backend_used,
                    "model": lmstudio_config.model,
                    "base_url": lmstudio_config.base_url,
                    "hint_mode": "none" if template_obj is not None else lmstudio_config.hint_mode,
                    "max_tokens": lmstudio_config.max_tokens_per_request,
                    "timeout_seconds": lmstudio_config.timeout,
                    "crop_padding_ratio": lmstudio_config.crop_padding_ratio,
                    "crop_padding_ratio_y": lmstudio_config.crop_padding_ratio_y,
                    "confidence_threshold": lmstudio_config.confidence_threshold,
                    "context_confidence": lmstudio_config.context_confidence,
                    "max_workers": lmstudio_config.max_workers,
                    "box_indices": list(lmstudio_config.box_indices)
                    if lmstudio_config.box_indices is not None
                    else None,
                    "stats": llm_stats.to_dict(),
                }
                if lmstudio_config is not None
                else None
            ),
        },
        "results": serialized_results,
    }
    json_str = json.dumps(output_data, ensure_ascii=False, indent=2 if pretty_enabled else None)

    output_path = _json_output_path(
        input_path,
        output=output,
        output_dir=output_dir,
    )
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json_str, encoding="utf-8")
    else:
        print(json_str)

    ai_status = f"AI: {ai_backend_used}" if ai_backend_used else "AI: off"
    console.print(
        f"Processed {len(results)} result(s); {ai_status}."
        + ("" if output_path is None else f" JSON saved to {output_path}")
    )
