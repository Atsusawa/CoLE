import json
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from cole.core.feedback import ExecFeedback
from cole.core.sparse_windows import build_sparse_windows
from cole.runtime.wan_adapter import record_runtime_feedback
from scripts import run_cole


class MockScheduler:
    def __init__(self):
        self.config = SimpleNamespace(flow_shift=7.0, prediction_type="flow_prediction")
        self.sigmas = None

    def register_to_config(self, **kwargs):
        self.config.__dict__.update(kwargs)

    def set_timesteps(self, num_inference_steps, device):
        shift = self.config.flow_shift
        fraction = torch.linspace(1.0, 0.1, num_inference_steps, device=device)
        self.sigmas = shift * fraction / (1 + (shift - 1) * fraction)
        self.timesteps = (self.sigmas * 999).round().int()

    def step(self, noise_pred, timestep, latents, return_dict=False):
        return (latents - noise_pred * 0.02,)

    def add_noise(self, latents, noise, timestep):
        return latents * 0.2 + noise * 0.8


class MockTransformer:
    dtype = torch.bfloat16
    config = SimpleNamespace(in_channels=16, patch_size=(1, 2, 2))

    def __init__(self, pipe):
        self.pipe = pipe

    def __call__(self, *, hidden_states, timestep, encoder_hidden_states, attention_kwargs=None, return_dict=False):
        t = int(timestep.flatten()[0])
        name = self.pipe.cole_mask_plan.backend_name
        sampled = os.environ.get("COLE_ATTENTION_SAMPLE_TIMESTEPS", "").split(",")
        patch = {0: torch.full((1, 2, 1, 2, 512), 1 / 512)} if str(t) in sampled and attention_kwargs else None
        record_runtime_feedback(self.pipe, ExecFeedback(
            actual_density=0.61, backend_name=name, backend_is_sparse=name == "sparse",
            attention_prob_patches=patch,
        ))
        return (hidden_states * 0.125 + 0.05,)


class MockVAE(torch.nn.Module):
    dtype = torch.float32
    config = SimpleNamespace(z_dim=16, latents_mean=[0.0] * 16, latents_std=[1.0] * 16)

    def decode(self, latents, return_dict=False):
        return (latents[:, :3, :, :32, :32],)


class MockPipe:
    _execution_device = torch.device("cpu")
    vae_scale_factor_temporal = 40  # Three latent frames in this CPU fixture.
    vae_scale_factor_spatial = 8
    do_classifier_free_guidance = True

    def __init__(self):
        self.transformer = MockTransformer(self)
        self.vae = MockVAE()
        self.scheduler = MockScheduler()
        self.video_processor = SimpleNamespace(postprocess_video=lambda t, **kw: t.permute(0, 2, 1, 3, 4).sigmoid())

    def to(self, device):
        assert device == self._execution_device
        return self

    def encode_prompt(self, **kwargs):
        return torch.full((1, 4, 4096), 0.25), torch.zeros(1, 4, 4096)

    def progress_bar(self, total):
        return nullcontext(SimpleNamespace(update=lambda n: None))

    def maybe_free_model_hooks(self):
        pass


@pytest.fixture
def prepare_cli(monkeypatch, tmp_path):
    from cole.runtime import generate, install_wan_processor

    loads = []
    # Generation updates sampling configuration for the installed attention processor.
    for name in (
        "COLE_ATTENTION_PROXY_ENABLE", "COLE_ATTENTION_PROB_PATCHES_ENABLE",
        "COLE_ATTENTION_PROXY_STEP_INTERVAL", "COLE_ATTENTION_PROXY_LAYERS",
        "COLE_ATTENTION_PROXY_HEADS", "COLE_ATTENTION_PROXY_QUERIES",
        "COLE_ATTENTION_PROXY_KEYS", "COLE_ATTENTION_SAMPLE_TIMESTEPS",
    ):
        monkeypatch.delenv(name, raising=False)

    def load_pipe(model_dir, *, dtype):
        loads.append(model_dir)
        return MockPipe()

    monkeypatch.setattr(generate, "load_wan_pipeline", load_pipe)
    monkeypatch.setattr(generate, "load_sparse_kernel", lambda device: None)
    monkeypatch.setattr(install_wan_processor, "install_cole_wan_processor", lambda *a, **k: None)
    model = tmp_path / "model"
    model.mkdir()
    (model / "model_index.json").write_text("{}")

    def setup(variant=None, checkpoint=None):
        run = tmp_path / "run"
        args = [
            "run_cole.py", "--model-dir", str(model), "--prompt", "A sailboat.", "--seed", "20260406",
            "--device", "cpu", "--output-dir", str(run),
        ]
        if variant is not None:
            args.extend(["--variant", variant])
        if checkpoint is not None:
            args.extend(["--reuse-checkpoint", str(checkpoint)])
        monkeypatch.setattr(sys, "argv", args)
        return run, loads

    return setup


def test_frozen_program_generates_and_resumes(prepare_cli):
    from cole.runtime.generate import WARMUP_PROFILES

    run, loads = prepare_cli()
    run_cole.main()
    program_path = run / "inputs/program.json"
    previous_hash, repackaged_hash = next(iter(run_cole._REPACKAGED_CHECKPOINTS.items()))
    program_path.write_text(program_path.read_text().replace(repackaged_hash, previous_hash))
    run_cole.main()

    assert len(loads) == 1
    case = run / "strategies/rollout_r00/demo"
    assert (case / "output.mp4").stat().st_size > 0
    metadata = json.loads((case / "run_metadata.json").read_text())
    trace = json.loads((case / "decision_trace.json").read_text())
    assert len(trace) == 50
    assert metadata["switch_step"] == 24
    assert trace[24]["transition"]["applies_to_step"] == 25
    assert trace[24]["transition"]["scheduler_flow_shift_after"] == 9.0
    assert metadata["prediction_reuse_action"] == "ultra"
    assert [row["profile_name"] for row in metadata["profile_warmup_rows"]] == list(WARMUP_PROFILES)
    stats = metadata["prediction_reuse"]["stats"]
    assert set(metadata["attention_sample_steps"]) <= set(stats["compute_step_indices"])
    assert 25 in stats["compute_step_indices"]
    assert stats["cache_steps"] > 0
    assert stats["policy_update_count"] == 1
    assert metadata["decode_backend"] == "cpu_full"

    (case / "decision_trace.json").unlink()
    run_cole.main()
    assert len(loads) == 2
    assert (case / "decision_trace.json").is_file()


def test_cole_l_no_reuse_keeps_learned_switch_and_sparsity(prepare_cli):
    run, loads = prepare_cli("cole-l-no-reuse")
    run_cole.main()
    method_path = run / "inputs/method.json"
    previous_hash, repackaged_hash = next(iter(run_cole._REPACKAGED_CHECKPOINTS.items()))
    method_path.write_text(method_path.read_text().replace(repackaged_hash, previous_hash))
    run_cole.main()
    assert len(loads) == 1

    case = run / "strategies/rollout_r00/demo"
    metadata = json.loads((case / "run_metadata.json").read_text())
    trace = json.loads((case / "decision_trace.json").read_text())
    assert metadata["variant"] == "cole-l-no-reuse"
    assert metadata["program_execution"] is None
    assert metadata["prediction_reuse_action"] == "off"
    assert metadata["prediction_reuse"] is None
    switch_probabilities = metadata["switch_probabilities"]
    assert metadata["switch_step"] == int(max(switch_probabilities, key=switch_probabilities.get))
    for row in trace[3:]:
        action = row["sparsity"]["action"]
        probabilities = row["sparsity"]["action_probabilities"]
        assert action == max(probabilities, key=probabilities.get)


@pytest.fixture
def six_action_checkpoint(tmp_path):
    from cole.runtime.reuse_readout import REUSE_ACTIONS, ReuseReadout

    path = tmp_path / "six_action_checkpoint.pt"
    checkpoint = torch.load(run_cole.ROOT / "checkpoints/wan21_controller.pt", weights_only=True)
    width = checkpoint["model_config"]["adapter_width"]
    readout = ReuseReadout(shared_width=5 * checkpoint["model_config"]["backbone"]["width"], width=width, dropout=0)
    with torch.no_grad():
        for param in readout.parameters():
            param.zero_()
        # Score the test candidate using its time-reduction input.
        head = readout.reuse_head
        head.candidate_proj[0].weight.fill_(1)
        head.candidate_proj[1].weight[0, 7] = 1
        head.scorer[0].weight.fill_(1)
        head.scorer[1].weight[0, width] = 1
        head.scorer[4].weight[0, 0] = 1
    checkpoint["model_state_dict"].update(readout.state_dict())
    checkpoint["model_config"].update(reuse_decision_feature_dim=9, reuse_candidate_feature_dim=8)
    checkpoint["prediction_reuse"] = {
        "probe_step": 14,
        "candidates": [
            {
                "action": action,
                "expected_reuse_fraction": 0.0,
                "expected_time_reduction_fraction": float(action == "ultra"),
            }
            for action in REUSE_ACTIONS
        ],
    }
    torch.save(checkpoint, path)
    return path


@pytest.mark.parametrize("legacy", [False, True])
def test_cole_l_scores_six_presets_and_protects_switch(prepare_cli, six_action_checkpoint, legacy):
    from cole.runtime.reuse_readout import REUSE_ACTIONS

    if legacy:
        checkpoint = torch.load(six_action_checkpoint, weights_only=True)
        policy = checkpoint["velocity_cache_policy"] = checkpoint.pop("prediction_reuse")
        for candidate in policy["candidates"]:
            candidate["expected_cache_fraction"] = candidate.pop("expected_reuse_fraction")
        config = checkpoint["model_config"]
        for feature in ("decision_feature_dim", "candidate_feature_dim"):
            config[f"cache_{feature}"] = config.pop(f"reuse_{feature}")
        checkpoint["model_state_dict"] = {
            key.replace("reuse_", "cache_", 1) if key.startswith("reuse_") else key: tensor
            for key, tensor in checkpoint["model_state_dict"].items()
        }
        torch.save(checkpoint, six_action_checkpoint)

    run, _ = prepare_cli("cole-l", six_action_checkpoint)
    run_cole.main()

    case = run / "strategies/rollout_r00/demo"
    metadata = json.loads((case / "run_metadata.json").read_text())
    assert metadata["variant"] == "cole-l"
    assert metadata["program_execution"] is None
    assert tuple(metadata["prediction_reuse_probabilities"]) == REUSE_ACTIONS
    assert metadata["prediction_reuse_action"] == "ultra"
    switch = metadata["switch_step"]
    stats = metadata["prediction_reuse"]["stats"]
    full = set(stats["compute_step_indices"])
    window_starts = {window.start_step for window in build_sparse_windows(total_steps=50, switch_step=switch)}
    protected = {*range(15), *metadata["attention_sample_steps"], switch - 1, switch, switch + 1, 49, *window_starts}
    assert protected <= full
    assert stats["cache_steps"] > 0
    assert stats["policy_update_count"] == 1

    prepare_cli("cole")
    with pytest.raises(SystemExit) as error:
        run_cole.main()
    assert error.value.code == 2


@pytest.mark.parametrize("problem, message", [
    ("four actions", "six ordered reuse actions"),
    ("missing statistics", "needs reuse and time reduction statistics"),
    ("missing readout", "incompatible prediction reuse readout weights"),
])
def test_cole_l_rejects_invalid_checkpoint(prepare_cli, six_action_checkpoint, capsys, problem, message):
    checkpoint = torch.load(six_action_checkpoint, weights_only=True)
    candidates = checkpoint["prediction_reuse"]["candidates"]
    if problem == "four actions":
        del candidates[4:]
    elif problem == "missing statistics":
        del candidates[-1]["expected_reuse_fraction"]
    else:
        del checkpoint["model_state_dict"]["reuse_decision_proj.0.weight"]
    torch.save(checkpoint, six_action_checkpoint)
    run, loads = prepare_cli("cole-l", six_action_checkpoint)
    with pytest.raises(SystemExit) as error:
        run_cole.main()
    assert error.value.code == 2
    assert message in capsys.readouterr().err
    assert not loads
    assert not (run / "inputs").exists()
