from pathlib import Path

import pytest

from cole.common.io import read_json, write_json
from scripts.score_vbench import (
    StrategySpec,
    StrategyVideoRecord,
    _build_strategy_row,
    _build_video_rows,
    _prepare_result_cache,
    _write_summary_md,
)


def test_result_cache_does_not_reuse_scores_for_replaced_video(tmp_path: Path) -> None:
    video = tmp_path / "output.mp4"
    video.write_bytes(b"first video")
    record = StrategyVideoRecord(
        "CoLE", "sample", "A sailboat", ("temporal_flickering",), video,
        tmp_path / "flat.mp4", tmp_path / "run_metadata.json",
    )
    result = tmp_path / "CoLE_eval_results.json"
    inputs = {
        "result_path": result,
        "repo_path": tmp_path / "vbench",
        "full_info_rows": [{"prompt_id": "sample", "video_list": [str(record.flat_video_path)]}],
        "records": [record],
    }
    scores = {"temporal_flickering": [0.9, []]}

    write_json(result, scores)
    with pytest.raises(ValueError, match="Cannot verify cached scores"):
        _prepare_result_cache(**inputs, recompute=False)
    assert read_json(result) == scores

    _prepare_result_cache(**inputs, recompute=True)
    write_json(result, scores)
    _prepare_result_cache(**inputs, recompute=False)
    assert read_json(result) == scores

    video.write_bytes(b"replacement video with the same output path")
    with pytest.raises(ValueError, match="Cannot verify cached scores"):
        _prepare_result_cache(**inputs, recompute=False)
    assert read_json(result) == scores
    _prepare_result_cache(**inputs, recompute=True)
    assert read_json(result) == {}


@pytest.mark.parametrize("prompt_inclusive_sec, expected_cell", [(30.6, "30.600"), (None, "NA")])
def test_summary_reports_prompt_inclusive_timing(
    tmp_path: Path, prompt_inclusive_sec: float | None, expected_cell: str,
) -> None:
    strategy = StrategySpec(name="CoLE", run_root=tmp_path)
    record = StrategyVideoRecord(
        "CoLE", "sample", "A sailboat", ("temporal_flickering",),
        tmp_path / "output.mp4", tmp_path / "flat.mp4", tmp_path / "run_metadata.json",
    )
    metadata = {"end_to_end_sec": 29.0}
    if prompt_inclusive_sec is not None:
        metadata["prompt_inclusive_generation_sec"] = prompt_inclusive_sec
    write_json(record.run_metadata_path, metadata)

    video_rows = _build_video_rows(strategy=strategy, records=[record], result_payload={})
    row = _build_strategy_row(
        strategy=strategy, result_payload={}, records=[record],
        result_path=tmp_path / "scores.json", video_rows=video_rows,
    )
    report = tmp_path / "summary.md"
    _write_summary_md(path=report, strategy_rows=[row], prompt_infos={})

    assert video_rows[0]["prompt_inclusive_generation_sec"] == prompt_inclusive_sec
    assert row.get("mean_prompt_inclusive_generation_sec") == prompt_inclusive_sec
    assert f"| 29.000 | {expected_cell} |" in report.read_text()
