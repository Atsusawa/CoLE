#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import gc
import importlib
import json
import math
import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cole.common.io import read_json, read_jsonl, write_json, write_jsonl
from cole.core.program import digest


VBENCH16_DIMENSIONS: tuple[str, ...] = (
    "subject_consistency",
    "background_consistency",
    "aesthetic_quality",
    "imaging_quality",
    "object_class",
    "multiple_objects",
    "color",
    "spatial_relationship",
    "scene",
    "temporal_style",
    "overall_consistency",
    "human_action",
    "temporal_flickering",
    "motion_smoothness",
    "dynamic_degree",
    "appearance_style",
)

QUALITY_DIMENSIONS: tuple[str, ...] = (
    "subject_consistency",
    "background_consistency",
    "temporal_flickering",
    "motion_smoothness",
    "aesthetic_quality",
    "imaging_quality",
    "dynamic_degree",
)

SEMANTIC_DIMENSIONS: tuple[str, ...] = (
    "object_class",
    "multiple_objects",
    "human_action",
    "color",
    "spatial_relationship",
    "scene",
    "appearance_style",
    "temporal_style",
    "overall_consistency",
)

OFFICIAL_DIM_WEIGHTS: dict[str, float] = {
    "subject_consistency": 1.0,
    "background_consistency": 1.0,
    "temporal_flickering": 1.0,
    "motion_smoothness": 1.0,
    "aesthetic_quality": 1.0,
    "imaging_quality": 1.0,
    "dynamic_degree": 0.5,
    "object_class": 1.0,
    "multiple_objects": 1.0,
    "human_action": 1.0,
    "color": 1.0,
    "spatial_relationship": 1.0,
    "scene": 1.0,
    "appearance_style": 1.0,
    "temporal_style": 1.0,
    "overall_consistency": 1.0,
}

OFFICIAL_NORMALIZE_RANGES: dict[str, tuple[float, float]] = {
    "subject_consistency": (0.1462, 1.0),
    "background_consistency": (0.2615, 1.0),
    "temporal_flickering": (0.6293, 1.0),
    "motion_smoothness": (0.7060, 0.9975),
    "dynamic_degree": (0.0, 1.0),
    "aesthetic_quality": (0.0, 1.0),
    # VBench reports video-level imaging_quality on a 0-100 scale.
    # Normalize to 0-1 before computing quality scores.
    "imaging_quality": (0.0, 100.0),
    "object_class": (0.0, 1.0),
    "multiple_objects": (0.0, 1.0),
    "human_action": (0.0, 1.0),
    "color": (0.0, 1.0),
    "spatial_relationship": (0.0, 1.0),
    "scene": (0.0, 0.8222),
    "appearance_style": (0.0009, 0.2855),
    "temporal_style": (0.0, 0.364),
    "overall_consistency": (0.0, 0.364),
}

QUALITY_WEIGHT = 4.0
SEMANTIC_WEIGHT = 1.0
TIMING_FIELDS = ("end_to_end_sec", "denoise_loop_sec", "decode_post_sec", "prompt_inclusive_generation_sec")


@dataclass(frozen=True)
class PromptInfo:
    prompt_id: str
    split_prompt_text: str
    canonical_prompt_text: str
    official_info: dict[str, Any]

    @property
    def dimensions(self) -> tuple[str, ...]:
        return tuple(str(item) for item in self.official_info["dimension"])


@dataclass(frozen=True)
class StrategySpec:
    name: str
    run_root: Path


@dataclass(frozen=True)
class StrategyVideoRecord:
    strategy_name: str
    prompt_id: str
    prompt_en: str
    dimensions: tuple[str, ...]
    source_video_path: Path
    flat_video_path: Path
    run_metadata_path: Path


def _parse_strategy_specs(raw_values: Sequence[str]) -> list[StrategySpec]:
    specs: list[StrategySpec] = []
    names: set[str] = set()
    for raw_value in raw_values:
        if "=" in raw_value:
            raw_name, raw_path = raw_value.split("=", 1)
            name = raw_name.strip()
            path = Path(raw_path.strip()).expanduser().resolve()
        else:
            path = Path(raw_value).expanduser().resolve()
            name = path.name
        if not name or name in (".", "..") or Path(name).name != name:
            raise ValueError(f"Invalid strategy name in --strategy_run: {raw_value}")
        if name in names:
            raise ValueError(f"Duplicate strategy name: {name}")
        if not path.is_dir():
            raise FileNotFoundError(f"Strategy run root not found: {path}")
        names.add(name)
        specs.append(StrategySpec(name=name, run_root=path))
    if not specs:
        raise ValueError("At least one --strategy_run is required.")
    return specs


def _safe_mean(values: Iterable[float | None]) -> float | None:
    kept = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not kept:
        return None
    return float(sum(kept) / len(kept))


def _normalize_dimension(
    dimension: str,
    raw_value: float | bool | int | None,
    *,
    clamp: bool = False,
    pre_normalized: bool = False,
) -> float | None:
    if raw_value is None:
        return None
    if bool(pre_normalized):
        normalized = float(raw_value)
    else:
        min_value, max_value = OFFICIAL_NORMALIZE_RANGES[dimension]
        normalized = (float(raw_value) - min_value) / (max_value - min_value)
    weighted = normalized * OFFICIAL_DIM_WEIGHTS[dimension]
    if clamp:
        weighted = max(0.0, min(float(OFFICIAL_DIM_WEIGHTS[dimension]), weighted))
    return float(weighted)


def _score_dimension_subset(row: Mapping[str, Any], dimensions: Sequence[str]) -> float | None:
    total = 0.0
    weight_sum = 0.0
    for dimension in dimensions:
        normalized = row.get(f"{dimension}_normalized_official")
        if normalized is None:
            return None
        total += float(normalized)
        weight_sum += OFFICIAL_DIM_WEIGHTS[dimension]
    if weight_sum <= 0.0:
        return None
    return float(total / weight_sum)


def _is_vbench_result_payload(value: Any) -> bool:
    return isinstance(value, (list, tuple)) and len(value) >= 2


def _add_official_vbench16_scores(
    row: dict[str, Any],
    *,
    clamp_normalized: bool = False,
    pre_normalized_dimensions: set[str] | None = None,
) -> None:
    pre_normalized = (
        set()
        if pre_normalized_dimensions is None
        else {str(item) for item in pre_normalized_dimensions}
    )
    for dimension in VBENCH16_DIMENSIONS:
        row[f"{dimension}_normalized_official"] = (
            _normalize_dimension(
                dimension,
                row.get(dimension),
                clamp=clamp_normalized,
                pre_normalized=(dimension in pre_normalized),
            )
            if row.get(dimension) is not None
            else None
        )
    row["official_quality_score"] = _score_dimension_subset(row, QUALITY_DIMENSIONS)
    row["official_semantic_score"] = _score_dimension_subset(row, SEMANTIC_DIMENSIONS)
    quality = row.get("official_quality_score")
    semantic = row.get("official_semantic_score")
    row["official_total_score"] = (
        None
        if quality is None or semantic is None
        else float((float(quality) * QUALITY_WEIGHT + float(semantic) * SEMANTIC_WEIGHT) / (QUALITY_WEIGHT + SEMANTIC_WEIGHT))
    )


def _load_prompt_infos(prompt_split_path: Path, official_full_info_path: Path) -> dict[str, PromptInfo]:
    split_rows = read_jsonl(prompt_split_path)
    official_rows = read_json(official_full_info_path)
    if not isinstance(official_rows, list):
        raise ValueError(f"Official full info must be a JSON list: {official_full_info_path}")

    infos: dict[str, PromptInfo] = {}
    for row in split_rows:
        prompt_id = str(row.get("prompt_id") or "").strip()
        if not prompt_id:
            raise ValueError(f"Missing prompt_id in split row: {row}")
        if prompt_id in infos:
            raise ValueError(f"Duplicate prompt ID in split: {prompt_id}")
        source_line = int(row["source_line"])
        official_index = source_line - 1
        if official_index < 0 or official_index >= len(official_rows):
            raise IndexError(f"source_line out of range for {prompt_id}: {source_line}")
        official_info = copy.deepcopy(official_rows[official_index])
        canonical_prompt_text = str(row.get("canonical_prompt_text") or "").strip()
        official_prompt = str(official_info.get("prompt_en") or "").strip()
        if official_prompt != canonical_prompt_text:
            raise ValueError(
                f"Prompt mismatch for {prompt_id}: source_line={source_line}, "
                f"official={official_prompt!r}, split={canonical_prompt_text!r}"
            )
        official_dimensions = tuple(str(item) for item in official_info.get("dimension") or [])
        split_dimensions = tuple(str(item) for item in row.get("dimensions") or [])
        if set(official_dimensions) != set(split_dimensions):
            raise ValueError(
                f"Dimension mismatch for {prompt_id}: official={official_dimensions}, split={split_dimensions}"
            )
        infos[prompt_id] = PromptInfo(
            prompt_id=prompt_id,
            split_prompt_text=str(row.get("prompt_text") or "").strip(),
            canonical_prompt_text=canonical_prompt_text,
            official_info=official_info,
        )
    if not infos:
        raise ValueError(f"Prompt split is empty: {prompt_split_path}")
    return infos


def _find_video_for_prompt(run_root: Path, prompt_id: str) -> tuple[Path, Path]:
    prompt_dir = run_root / "strategies" / "rollout_r00" / prompt_id
    video_path = prompt_dir / "output.mp4"
    metadata_path = prompt_dir / "run_metadata.json"
    if not video_path.is_file():
        raise FileNotFoundError(f"Missing video for {prompt_id}: {video_path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing run metadata for {prompt_id}: {metadata_path}")
    return video_path, metadata_path


def _sanitize_filename_component(value: str) -> str:
    out = str(value).replace("/", "_").replace("\x00", "").strip()
    while "  " in out:
        out = out.replace("  ", " ")
    return out or "prompt"


def _flat_video_name(prompt_info: PromptInfo) -> str:
    if "human_action" in prompt_info.dimensions:
        return f"{_sanitize_filename_component(prompt_info.canonical_prompt_text)}-0.mp4"
    return f"{prompt_info.prompt_id}.mp4"


def _build_strategy_records(
    *,
    strategy: StrategySpec,
    prompt_infos: Mapping[str, PromptInfo],
    flat_root: Path,
) -> list[StrategyVideoRecord]:
    records: list[StrategyVideoRecord] = []
    flat_dir = flat_root / strategy.name
    seen_paths: set[Path] = set()
    for prompt_id, prompt_info in sorted(prompt_infos.items()):
        source_video_path, metadata_path = _find_video_for_prompt(strategy.run_root, prompt_id)
        flat_video_path = flat_dir / _flat_video_name(prompt_info)
        if flat_video_path in seen_paths:
            raise ValueError(f"Multiple prompts map to the VBench filename {flat_video_path.name}")
        seen_paths.add(flat_video_path)
        records.append(
            StrategyVideoRecord(
                strategy_name=strategy.name,
                prompt_id=prompt_id,
                prompt_en=prompt_info.canonical_prompt_text,
                dimensions=prompt_info.dimensions,
                source_video_path=source_video_path,
                flat_video_path=flat_video_path,
                run_metadata_path=metadata_path,
            )
        )
    return records


def _link_strategy_videos(records: Sequence[StrategyVideoRecord]) -> None:
    records[0].flat_video_path.parent.mkdir(parents=True, exist_ok=True)
    for record in records:
        flat_video = record.flat_video_path
        if flat_video.is_symlink():
            if flat_video.readlink() == record.source_video_path:
                continue
            flat_video.unlink()
        flat_video.symlink_to(record.source_video_path)


def _build_full_info(
    *,
    prompt_infos: Mapping[str, PromptInfo],
    records: Sequence[StrategyVideoRecord],
) -> list[dict[str, Any]]:
    record_by_prompt = {record.prompt_id: record for record in records}
    rows: list[dict[str, Any]] = []
    for prompt_id, prompt_info in sorted(prompt_infos.items()):
        record = record_by_prompt[prompt_id]
        info = copy.deepcopy(prompt_info.official_info)
        info["prompt_id"] = prompt_id
        info["dimension"] = list(record.dimensions)
        info["video_list"] = [str(record.flat_video_path)]
        rows.append(info)
    return rows


def _prepare_result_cache(
    *,
    result_path: Path,
    repo_path: Path,
    full_info_rows: Sequence[Mapping[str, Any]],
    records: Sequence[StrategyVideoRecord],
    recompute: bool,
) -> None:
    videos = []
    for record in records:
        video = record.source_video_path.resolve(strict=True)
        stat = video.stat()
        videos.append((str(video), stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))
    signature = digest({"vbench_repo": str(repo_path), "full_info": full_info_rows, "videos": videos})
    signature_path = result_path.with_name("evaluation_inputs.json")
    if result_path.is_file():
        if not recompute:
            if signature_path.is_file() and read_json(signature_path) == signature:
                return
            raise ValueError(
                f"Cannot verify cached scores in {result_path} for these inputs. "
                "Choose a new --output_root or pass --recompute_existing."
            )
        print(f"[vbench16] Starting a fresh evaluation in {result_path.parent}", flush=True)
        write_json(result_path, {})
    write_json(signature_path, signature)


def _ensure_vbench_sys_path(repo_path: Path) -> None:
    resolved = str(repo_path.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


def _patch_tag2text_scene_bert_init() -> None:
    import vbench.third_party.tag2Text.tag2text as tag2text_module

    bert_model_cls = tag2text_module.BertModel
    if getattr(bert_model_cls, "_cole_scene_offline_patch", False):
        return
    original_from_pretrained = bert_model_cls.from_pretrained

    @classmethod
    def _from_pretrained_or_config(cls: type[Any], pretrained_model_name_or_path: Any, *args: Any, **kwargs: Any) -> Any:
        if str(pretrained_model_name_or_path) == "bert-base-uncased":
            config = kwargs.get("config")
            if config is None:
                raise ValueError("Tag2Text scene offline patch requires an explicit BertConfig.")
            add_pooling_layer = bool(kwargs.get("add_pooling_layer", True))
            return cls(config=config, add_pooling_layer=add_pooling_layer)
        return original_from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

    bert_model_cls.from_pretrained = _from_pretrained_or_config  # type: ignore[method-assign]
    setattr(bert_model_cls, "_cole_scene_offline_patch", True)


def _patch_bert_tokenizer_offline() -> None:
    """Route VBench's hard-coded BERT tokenizer request to the shared cache."""
    local_tokenizer = os.environ.get("COLE_BERT_BASE_UNCASED_DIR")
    if not local_tokenizer:
        return
    tokenizer_dir = Path(local_tokenizer).expanduser()
    if not tokenizer_dir.is_dir():
        return
    from transformers import BertTokenizer

    if getattr(BertTokenizer, "_cole_offline_patch", False):
        return
    original_from_pretrained = BertTokenizer.from_pretrained

    @classmethod
    def _from_pretrained_or_local_cache(
        cls: type[Any], pretrained_model_name_or_path: Any, *args: Any, **kwargs: Any
    ) -> Any:
        if str(pretrained_model_name_or_path) == "bert-base-uncased":
            local_kwargs = dict(kwargs)
            local_kwargs.setdefault("local_files_only", True)
            return original_from_pretrained(str(tokenizer_dir), *args, **local_kwargs)
        return original_from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

    BertTokenizer.from_pretrained = _from_pretrained_or_local_cache  # type: ignore[method-assign]
    setattr(BertTokenizer, "_cole_offline_patch", True)


def _run_vbench16(
    *,
    repo_path: Path,
    full_info_path: Path,
    result_path: Path,
    device: str,
    dimensions: Sequence[str],
    local: bool,
    read_frame: bool,
) -> dict[str, Any]:
    existing_results: dict[str, Any] = {}
    if result_path.is_file():
        loaded = read_json(result_path)
        if isinstance(loaded, dict):
            existing_results = dict(loaded)
        if all(dimension in existing_results for dimension in dimensions):
            return existing_results

    _ensure_vbench_sys_path(repo_path)
    _patch_bert_tokenizer_offline()
    from vbench.utils import init_submodules

    pending = [dimension for dimension in dimensions if dimension not in existing_results]
    submodules_dict = init_submodules(pending, local=local, read_frame=read_frame)
    results_dict: dict[str, Any] = existing_results
    try:
        import torch

        eval_device: Any = torch.device(device)
    except Exception:
        eval_device = device
    for index, dimension in enumerate(pending, start=1):
        print(f"[vbench16] {index}/{len(pending)} {dimension}", flush=True)
        if dimension == "scene":
            _patch_tag2text_scene_bert_init()
        dimension_module = importlib.import_module(f"vbench.{dimension}")
        evaluate_func = getattr(dimension_module, f"compute_{dimension}")
        results_dict[dimension] = evaluate_func(str(full_info_path), eval_device, submodules_dict[dimension])
        write_json(result_path, results_dict)
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    return results_dict


def _result_video_maps(result_payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    maps: dict[str, dict[str, Any]] = {}
    for dimension in VBENCH16_DIMENSIONS:
        payload = result_payload.get(dimension)
        if not _is_vbench_result_payload(payload):
            continue
        per_video: dict[str, Any] = {}
        for item in payload[1] or []:
            if not isinstance(item, Mapping):
                continue
            video_path = str(item.get("video_path") or "")
            if video_path:
                per_video[str(Path(video_path).resolve())] = item.get("video_results")
                per_video[str(Path(video_path))] = item.get("video_results")
        maps[dimension] = per_video
    return maps


def _load_config_summary(run_root: Path) -> dict[str, Any]:
    rows_path = run_root / "config_summary_rows.jsonl"
    if not rows_path.is_file():
        return {}
    rows = read_jsonl(rows_path)
    return dict(rows[0]) if rows else {}


def _build_strategy_row(
    *,
    strategy: StrategySpec,
    result_payload: Mapping[str, Any],
    records: Sequence[StrategyVideoRecord],
    result_path: Path,
    video_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    row = _load_config_summary(strategy.run_root)
    row.update(
        {
            "strategy_name": strategy.name,
            "run_root": str(strategy.run_root),
            "result_path": str(result_path),
            "video_count": len(records),
        }
    )
    for key in TIMING_FIELDS:
        mean = _safe_mean(video.get(key) for video in video_rows)
        if mean is not None:
            row[f"mean_{key}"] = mean
    dimension_counts = Counter(dimension for record in records for dimension in record.dimensions)
    for dimension in VBENCH16_DIMENSIONS:
        payload = result_payload.get(dimension)
        if not _is_vbench_result_payload(payload):
            row[dimension] = None
            row[f"{dimension}_num_videos"] = 0
            continue
        value = payload[0]
        row[dimension] = float(value) if value is not None else None
        row[f"{dimension}_num_videos"] = len(payload[1] or [])
        row[f"{dimension}_expected_num_videos"] = int(dimension_counts.get(dimension, 0))
    # VBench's imaging-quality evaluator returns a 0-1 aggregate while its
    # per-video values are on a 0-100 scale.
    _add_official_vbench16_scores(row, pre_normalized_dimensions={"imaging_quality"})
    return row


def _build_video_rows(
    *,
    strategy: StrategySpec,
    records: Sequence[StrategyVideoRecord],
    result_payload: Mapping[str, Any],
) -> list[dict[str, Any]]:
    dimension_video_maps = _result_video_maps(result_payload)
    rows: list[dict[str, Any]] = []
    for record in records:
        metadata = read_json(record.run_metadata_path)
        row: dict[str, Any] = {
            "strategy_name": strategy.name,
            "prompt_id": record.prompt_id,
            "prompt_en": record.prompt_en,
            "generated_prompt": metadata.get("prompt") or metadata.get("prompt_text"),
            "dimensions": list(record.dimensions),
            "flat_video_path": str(record.flat_video_path),
            "source_video_path": str(record.source_video_path),
        }
        row.update({key: metadata.get(key) for key in TIMING_FIELDS})
        resolved_path = str(record.flat_video_path.resolve())
        direct_path = str(record.flat_video_path)
        for dimension in VBENCH16_DIMENSIONS:
            if dimension not in record.dimensions:
                row[dimension] = None
                continue
            row[dimension] = dimension_video_maps.get(dimension, {}).get(
                direct_path,
                dimension_video_maps.get(dimension, {}).get(resolved_path),
            )
            if isinstance(row[dimension], bool):
                row[dimension] = float(row[dimension])
            elif row[dimension] is not None:
                row[dimension] = float(row[dimension])
        _add_official_vbench16_scores(row)
        rows.append(row)
    return rows


def _format_float(value: Any, digits: int = 6) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not math.isfinite(number):
        return "NA"
    return f"{number:.{digits}f}"


def _write_summary_md(
    *,
    path: Path,
    strategy_rows: Sequence[Mapping[str, Any]],
    prompt_infos: Mapping[str, PromptInfo],
    dimension_counts: Mapping[str, int] | None = None,
) -> None:
    counts = Counter(dimension for info in prompt_infos.values() for dimension in info.dimensions)
    if dimension_counts is not None:
        counts = Counter({str(key): int(value) for key, value in dimension_counts.items()})
    lines = [
        "# VBench16 Strategy Evaluation",
        "",
        "Prompt dimensions follow the official VBench metadata.",
        "",
        "## Prompt Distribution",
        "",
        "| dimension | n |",
        "|---|---:|",
    ]
    for dimension in VBENCH16_DIMENSIONS:
        lines.append(f"| {dimension} | {counts.get(dimension, 0)} |")
    lines.extend(
        [
            "",
            "## Official Scores",
            "",
            "| strategy | total | quality | semantic | generation (s) | E2E incl. text encoding (s) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in strategy_rows:
        seconds = row.get("mean_end_to_end_sec", row.get("mean_elapsed_sec", row.get("mean_inference_sec")))
        lines.append(
            "| "
            f"{row.get('strategy_name')} | "
            f"{_format_float(row.get('official_total_score'))} | "
            f"{_format_float(row.get('official_quality_score'))} | "
            f"{_format_float(row.get('official_semantic_score'))} | "
            f"{_format_float(seconds, digits=3)} | "
            f"{_format_float(row.get('mean_prompt_inclusive_generation_sec'), digits=3)} |"
        )
    lines.extend(
        [
            "",
            "## Raw Dimension Scores",
            "",
            "| dimension | " + " | ".join(str(row.get("strategy_name")) for row in strategy_rows) + " |",
            "|---|" + "|".join("---:" for _ in strategy_rows) + "|",
        ]
    )
    for dimension in VBENCH16_DIMENSIONS:
        values = " | ".join(_format_float(row.get(dimension)) for row in strategy_rows)
        lines.append(f"| {dimension} | {values} |")
    if len(strategy_rows) == 2:
        a, b = strategy_rows[0], strategy_rows[1]
        lines.extend(
            [
                "",
                f"## Delta: {a.get('strategy_name')} - {b.get('strategy_name')}",
                "",
                "| metric | delta |",
                "|---|---:|",
            ]
        )
        for metric in ("official_total_score", "official_quality_score", "official_semantic_score"):
            left = a.get(metric)
            right = b.get(metric)
            delta = None if left is None or right is None else float(left) - float(right)
            lines.append(f"| {metric} | {_format_float(delta)} |")
        for dimension in VBENCH16_DIMENSIONS:
            left = a.get(dimension)
            right = b.get(dimension)
            delta = None if left is None or right is None else float(left) - float(right)
            lines.append(f"| {dimension} | {_format_float(delta)} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _evaluate(args: argparse.Namespace) -> None:
    repo_path = Path(args.vbench_repo).expanduser().resolve()
    official_full_info_path = (Path(args.official_full_info).expanduser().resolve() if args.official_full_info else repo_path / "vbench/VBench_full_info.json")
    prompt_split_path = Path(args.prompt_split).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    dimensions = list(dict.fromkeys(item.strip() for item in args.dimensions.split(",") if item.strip()))
    if not dimensions:
        raise ValueError("Select at least one VBench dimension.")
    unknown = sorted(set(dimensions) - set(VBENCH16_DIMENSIONS))
    if unknown:
        raise ValueError(f"Unsupported dimensions: {unknown}")

    os.environ["VBENCH_CACHE_DIR"] = str(Path(args.vbench_cache_dir).expanduser().resolve())
    output_root.mkdir(parents=True, exist_ok=True)
    flat_root = output_root / "flat_videos"
    flat_root.mkdir(parents=True, exist_ok=True)

    strategies = _parse_strategy_specs(args.strategy_run)
    prompt_infos = _load_prompt_infos(prompt_split_path, official_full_info_path)

    all_strategy_rows: list[dict[str, Any]] = []
    all_video_rows: list[dict[str, Any]] = []
    summary_dimension_counts: Counter[str] | None = None
    for strategy in strategies:
        print(f"[vbench16] strategy={strategy.name}", flush=True)
        strategy_out = output_root / strategy.name
        strategy_out.mkdir(parents=True, exist_ok=True)
        records = _build_strategy_records(
            strategy=strategy,
            prompt_infos=prompt_infos,
            flat_root=flat_root,
        )
        full_info_rows = _build_full_info(prompt_infos=prompt_infos, records=records)
        result_path = strategy_out / f"{strategy.name}_eval_results.json"
        _prepare_result_cache(
            result_path=result_path, repo_path=repo_path, full_info_rows=full_info_rows,
            records=records, recompute=args.recompute_existing,
        )
        _link_strategy_videos(records)
        if summary_dimension_counts is None:
            summary_dimension_counts = Counter(dimension for record in records for dimension in record.dimensions)
        manifest_rows = [
            {
                "strategy_name": record.strategy_name,
                "prompt_id": record.prompt_id,
                "prompt_en": record.prompt_en,
                "dimensions": list(record.dimensions),
                "flat_video_path": str(record.flat_video_path),
                "source_video_path": str(record.source_video_path),
            }
            for record in records
        ]
        write_jsonl(strategy_out / f"{strategy.name}_manifest.jsonl", manifest_rows)
        full_info_path = strategy_out / f"{strategy.name}_full_info.json"
        write_json(full_info_path, full_info_rows)
        write_json(
            strategy_out / "dimension_counts.json",
            {
                "from_split": Counter(dimension for info in prompt_infos.values() for dimension in info.dimensions),
                "from_full_info": Counter(dimension for row in full_info_rows for dimension in row["dimension"]),
            },
        )
        result_payload = _run_vbench16(
            repo_path=repo_path,
            full_info_path=full_info_path,
            result_path=result_path,
            device=args.device,
            dimensions=dimensions,
            local=True,
            read_frame=False,
        )
        video_rows = _build_video_rows(strategy=strategy, records=records, result_payload=result_payload)
        all_strategy_rows.append(
            _build_strategy_row(
                strategy=strategy, result_payload=result_payload, records=records,
                result_path=result_path, video_rows=video_rows,
            )
        )
        all_video_rows.extend(video_rows)

    all_strategy_rows = sorted(all_strategy_rows, key=lambda row: str(row.get("strategy_name") or ""))
    all_video_rows = sorted(all_video_rows, key=lambda row: (str(row.get("strategy_name") or ""), str(row.get("prompt_id") or "")))
    write_jsonl(output_root / "strategy_summary_with_vbench16.jsonl", all_strategy_rows)
    write_jsonl(output_root / "video_summary_with_vbench16.jsonl", all_video_rows)
    write_json(
        output_root / "vbench16_metadata.json",
        {
            "prompt_split": str(prompt_split_path),
            "official_full_info": str(official_full_info_path),
            "vbench_repo": str(repo_path),
            "vbench_cache_dir": os.environ.get("VBENCH_CACHE_DIR"),
            "dimensions": dimensions,
            "official_quality_dimensions": list(QUALITY_DIMENSIONS),
            "official_semantic_dimensions": list(SEMANTIC_DIMENSIONS),
            "official_dimension_weights": OFFICIAL_DIM_WEIGHTS,
            "official_normalize_ranges": {
                key: {"min": value[0], "max": value[1]} for key, value in OFFICIAL_NORMALIZE_RANGES.items()
            },
        },
    )
    _write_summary_md(
        path=output_root / "summary.md",
        strategy_rows=all_strategy_rows,
        prompt_infos=prompt_infos,
        dimension_counts=summary_dimension_counts,
    )
    print(output_root / "summary.md", flush=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run full VBench-16 evaluation over existing strategy outputs.")
    parser.add_argument(
        "--strategy_run",
        action="append",
        required=True,
        help="Strategy spec as name=/path/to/run. May be repeated.",
    )
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--prompt_split", default=str(PROJECT_ROOT / "configs/vbench946.jsonl"))
    parser.add_argument("--vbench_repo", default=os.environ.get("VBENCH_REPO", "VBench"))
    parser.add_argument("--official_full_info", default=None)
    parser.add_argument("--vbench_cache_dir", default=os.environ.get("VBENCH_CACHE_DIR", str(Path.home() / ".cache/vbench")))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dimensions", default=",".join(VBENCH16_DIMENSIONS))
    parser.add_argument("--recompute_existing", action="store_true")
    return parser


def main() -> None:
    parser = build_arg_parser()
    try:
        _evaluate(parser.parse_args())
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
