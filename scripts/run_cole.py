#!/usr/bin/env python3
"""Generate Wan2.1 video with CoLE or CoLE-L."""
from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cole.core.program import StrategyProgram, file_sha256

# These two checkpoint files carry identical controller tensors and inference settings.
_REPACKAGED_CHECKPOINTS = {
    "657827cdf290de2b78c4d61aefef784297e1d39731e7b286368dbf450689c82d":
        "e159127418de79d8de2ab718d545dac734449dc473df454858a6ff614d497a16",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompt")
    source.add_argument("--vbench", action="store_true", help="Generate the 946 paper prompts for one seed.")
    parser.add_argument("--prompt-id", help="Generate one prompt ID from the VBench manifest.")
    parser.add_argument("--model-dir", type=Path, required=True, help="Local Wan2.1-T2V-1.3B-Diffusers files.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--variant", choices=("cole", "cole-l", "cole-l-no-reuse"), default="cole",
        help="Use the evolved program, learned readouts, or the learned controller with reuse disabled.",
    )
    parser.add_argument("--reuse-checkpoint", type=Path, help="Six-action shared controller checkpoint for CoLE-L.")
    parser.add_argument("--seed", type=int, default=20260406)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dry-run", action="store_true", help="Write input manifests and per-prompt commands.")
    args = parser.parse_args()
    if args.prompt_id and not args.vbench:
        parser.error("--prompt-id requires --vbench.")

    if args.variant == "cole-l" and args.reuse_checkpoint is None:
        parser.error("--variant cole-l requires --reuse-checkpoint.")
    if args.reuse_checkpoint is not None and args.variant != "cole-l":
        parser.error("--reuse-checkpoint requires --variant cole-l.")
    reuse_checkpoint = args.reuse_checkpoint.expanduser().resolve() if args.reuse_checkpoint else None
    if reuse_checkpoint is not None and not reuse_checkpoint.is_file():
        parser.error(f"Missing CoLE-L checkpoint at {reuse_checkpoint}.")

    model_dir = args.model_dir.expanduser().resolve()
    if not args.dry_run and not (model_dir / "model_index.json").is_file():
        parser.error(f"Missing Wan Diffusers model_index.json in {model_dir}.")

    program = StrategyProgram.load(ROOT / "configs/wan21_program.json")
    checkpoint = ROOT / program.checkpoint_path
    if file_sha256(checkpoint) != program.checkpoint_sha256:
        parser.error("Controller checkpoint SHA-256 differs from the bundled program.")
    if args.vbench:
        rows = [
            json.loads(line) for line in
            (ROOT / "configs/vbench946.jsonl").read_text(encoding="utf-8").splitlines()
        ]
    else:
        rows = [{"prompt_id": "demo", "prompt": args.prompt, "switch_step": 21, "rollout_index": 0}]
    for row in rows:
        row["seed"] = args.seed
    selected_rows = [row for row in rows if not args.prompt_id or row["prompt_id"] == args.prompt_id]
    if not selected_rows:
        parser.error(f"VBench prompt ID {args.prompt_id!r} was not found.")

    output_dir = args.output_dir.expanduser().resolve()
    inputs = output_dir / "inputs"
    program_payload = program.to_dict()
    program_payload["incumbent"]["checkpoint_path"] = str(checkpoint)
    manifest = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    method_path = inputs / "method.json"
    program_path = inputs / "program.json"
    if (args.variant == "cole" and method_path.exists()) or (args.variant != "cole" and program_path.exists()):
        parser.error("Existing inputs use a different variant. Use a separate output directory.")
    method = {"variant": args.variant, "base_checkpoint_sha256": program.checkpoint_sha256}
    if reuse_checkpoint is not None:
        method["reuse_checkpoint_sha256"] = file_sha256(reuse_checkpoint)
    selection = (
        (program_path, json.dumps(program_payload, indent=2) + "\n")
        if args.variant == "cole" else (method_path, json.dumps(method, indent=2) + "\n")
    )
    input_files = (
        (inputs / "prompts.jsonl", manifest), selection,
        (inputs / "model.json", json.dumps({"model_dir": str(model_dir)}) + "\n"),
    )
    for path, contents in input_files:
        if path.exists():
            compatible = path.read_text(encoding="utf-8") == contents
            if not compatible and path.name == "program.json":
                recorded = StrategyProgram.load(path).to_dict()
                # A release checkout can move while the weights and ordered rules stay identical.
                incumbent = recorded["incumbent"]
                incumbent["checkpoint_path"] = str(checkpoint)
                recorded_hash = incumbent["checkpoint_sha256"]
                incumbent["checkpoint_sha256"] = _REPACKAGED_CHECKPOINTS.get(recorded_hash, recorded_hash)
                compatible = recorded == program_payload
            if not compatible and path.name == "method.json":
                recorded = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(recorded, dict) and isinstance(recorded.get("base_checkpoint_sha256"), str):
                    recorded_hash = recorded["base_checkpoint_sha256"]
                    recorded["base_checkpoint_sha256"] = _REPACKAGED_CHECKPOINTS.get(recorded_hash, recorded_hash)
                    compatible = recorded == method
            if not compatible:
                parser.error(f"Existing inputs differ in {path}. Use a separate output directory.")

    def write_inputs() -> None:
        inputs.mkdir(parents=True, exist_ok=True)
        for path, contents in input_files:
            if not path.exists():
                path.write_text(contents, encoding="utf-8")

    if args.dry_run:
        write_inputs()
        logs = output_dir / "logs"
        logs.mkdir(exist_ok=True)
        for row in selected_rows:
            prompt_id = row["prompt_id"]
            command = [
                sys.executable, str(Path(__file__).resolve()), "--model-dir", str(model_dir),
                "--output-dir", str(output_dir), "--seed", str(args.seed), "--device", args.device,
            ]
            if args.variant != "cole":
                command.extend(["--variant", args.variant])
            if reuse_checkpoint is not None:
                command.extend(["--reuse-checkpoint", str(reuse_checkpoint)])
            if args.vbench:
                command.extend(["--vbench", "--prompt-id", prompt_id])
            else:
                command.extend(["--prompt", row["prompt"]])
            (logs / f"{prompt_id}_r00.log").write_text(shlex.join(command) + "\n", encoding="utf-8")
        print(f"Prepared {len(selected_rows)} generation commands in {logs}.")
        return

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    from cole.common.seed import apply_visible_devices, parse_device_arg

    visible_devices, device_name = parse_device_arg(args.device)
    apply_visible_devices(visible_devices)
    import torch

    from cole.runtime.generate import WanGenerator

    device = torch.device(device_name)
    generator = None
    for row in selected_rows:
        prompt_id = row["prompt_id"]
        case_dir = output_dir / "strategies" / "rollout_r00" / prompt_id
        case_files = (case_dir / name for name in ("run_metadata.json", "decision_trace.json", "output.mp4"))
        if all(path.is_file() and path.stat().st_size > 0 for path in case_files):
            print(f"{prompt_id} already generated.")
            continue
        if generator is None:
            try:
                generator = WanGenerator(
                    model_dir, program=program, checkpoint_path=checkpoint, device=device,
                    variant=args.variant, reuse_checkpoint=reuse_checkpoint,
                )
            except ValueError as exc:
                parser.error(str(exc))
            write_inputs()
        generator.generate(prompt=row["prompt"], seed=row["seed"], output_dir=case_dir)
        print(f"{prompt_id} saved to {case_dir / 'output.mp4'}", flush=True)
    if generator is None:
        write_inputs()


if __name__ == "__main__":
    main()
