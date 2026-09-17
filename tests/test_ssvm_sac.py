from __future__ import annotations

import importlib.util
import importlib
import csv
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from torch import nn


def _load_ssvm_module(name):
    path = Path(__file__).parents[1] / "algorithm" / "SSVM_SAC" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_test_ssvm_{name}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_ssvm_package_module(name):
    package_name = "_test_algorithm"
    if package_name not in sys.modules:
        package = type(sys)(package_name)
        package.__path__ = [str(Path(__file__).parents[1] / "algorithm")]
        sys.modules[package_name] = package
    return importlib.import_module(f"{package_name}.SSVM_SAC.{name}")


def test_episode_archive_round_trip_and_sequence_windows_do_not_cross_episodes(tmp_path):
    module = _load_ssvm_module("dataset")
    EpisodeArchive = module.EpisodeArchive
    EpisodeSequenceDataset = module.EpisodeSequenceDataset

    for episode_id, length in enumerate((5, 3)):
        depth = np.arange(length, dtype=np.float32).reshape(length, 1, 1, 1)
        EpisodeArchive.save(
            tmp_path,
            episode_id=episode_id,
            depth=depth,
            base=np.full((length, 2), episode_id, dtype=np.float32),
            action=np.zeros((length, 1), dtype=np.float32),
            reward=np.zeros((length,), dtype=np.float32),
            terminated=np.asarray([False] * (length - 1) + [True]),
            truncated=np.zeros((length,), dtype=bool),
        )

    dataset = EpisodeSequenceDataset(
        tmp_path,
        sequence_length=4,
        split="all",
        stride=4,
    )

    assert len(dataset) == 3
    first = dataset[0]
    second_episode = dataset[2]
    assert torch.equal(first["depth"][:, 0, 0, 0], torch.tensor([0.0, 1.0, 2.0, 3.0]))
    assert torch.equal(first["valid"], torch.tensor([True, True, True, True]))
    assert torch.equal(second_episode["base"][:, 0], torch.tensor([1.0, 1.0, 1.0, 0.0]))
    assert torch.equal(second_episode["valid"], torch.tensor([True, True, True, False]))


def test_reconstruction_targets_mask_missing_past_and_future_frames():
    build_reconstruction_targets = _load_ssvm_module("dataset").build_reconstruction_targets

    depth = torch.arange(5, dtype=torch.float32).view(1, 5, 1, 1, 1)
    valid = torch.ones((1, 5), dtype=torch.bool)

    targets, masks = build_reconstruction_targets(depth, valid, offsets=(-2, -1, 1))

    assert targets[-2][0, 2, 0, 0, 0].item() == 0.0
    assert targets[-1][0, 2, 0, 0, 0].item() == 1.0
    assert targets[1][0, 2, 0, 0, 0].item() == 3.0
    assert masks[-2].tolist() == [[False, False, True, True, True]]
    assert masks[-1].tolist() == [[False, True, True, True, True]]
    assert masks[1].tolist() == [[True, True, True, True, False]]


def test_default_reconstruction_targets_are_t_minus_5_t_minus_10_and_t_plus_5():
    build_reconstruction_targets = _load_ssvm_module(
        "dataset"
    ).build_reconstruction_targets
    depth = torch.arange(16, dtype=torch.float32).view(1, 16, 1, 1, 1)
    valid = torch.ones((1, 16), dtype=torch.bool)

    targets, masks = build_reconstruction_targets(depth, valid)

    assert tuple(targets) == (-5, -10, 5)
    assert targets[-5][0, 10].item() == 5.0
    assert targets[-10][0, 10].item() == 0.0
    assert targets[5][0, 10].item() == 15.0
    assert masks[-5].sum().item() == 11
    assert masks[-10].sum().item() == 6
    assert masks[5].sum().item() == 11


def test_complete_episodes_are_padded_only_at_batch_collation(tmp_path):
    module = _load_ssvm_module("dataset")
    for episode_id, length in enumerate((3, 5)):
        module.EpisodeArchive.save(
            tmp_path,
            episode_id=episode_id,
            depth=np.zeros((length, 1, 2, 2), dtype=np.float32),
            base=np.zeros((length, 2), dtype=np.float32),
            action=np.zeros((length, 1), dtype=np.float32),
            reward=np.zeros(length, dtype=np.float32),
            terminated=np.asarray([False] * (length - 1) + [True]),
            truncated=np.zeros(length, dtype=bool),
        )
    dataset = module.EpisodeSequenceDataset(
        tmp_path, sequence_length=None, split="all"
    )

    batch = module.pad_episode_batch([dataset[0], dataset[1]])

    assert batch["depth"].shape == (2, 5, 1, 2, 2)
    assert batch["valid"].tolist() == [
        [True, True, True, False, False],
        [True, True, True, True, True],
    ]
    assert batch["episode_start"].tolist() == [
        [True, False, False, False, False],
        [True, False, False, False, False],
    ]


def test_depth_frame_dataset_opens_each_episode_only_during_preload(
    tmp_path, monkeypatch
):
    module = _load_ssvm_module("dataset")
    for episode_id in range(2):
        length = 3
        depth = np.full((length, 1, 2, 2), episode_id, dtype=np.uint8)
        module.EpisodeArchive.save(
            tmp_path,
            episode_id=episode_id,
            depth=depth,
            clean_depth=depth + 1,
            base=np.zeros((length, 2), dtype=np.float32),
            action=np.zeros((length, 1), dtype=np.float32),
            reward=np.zeros(length, dtype=np.float32),
            terminated=np.asarray([False, False, True]),
            truncated=np.zeros(length, dtype=bool),
        )

    original_load = module.np.load
    opened_paths = []

    def counted_load(path, *args, **kwargs):
        opened_paths.append(Path(path))
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(module.np, "load", counted_load)
    dataset = module.DepthFrameDataset(tmp_path, split="all")

    assert len(opened_paths) == 2
    assert len(dataset) == 6
    assert dataset[0]["depth"].flatten()[0].item() == 0
    assert dataset[0]["target_depth"].flatten()[0].item() == 1
    assert dataset[5]["depth"].flatten()[0].item() == 1
    assert len(opened_paths) == 2


def test_episode_sequence_dataset_does_not_reopen_preloaded_archives(
    tmp_path, monkeypatch
):
    module = _load_ssvm_module("dataset")
    for episode_id in range(2):
        length = 3
        module.EpisodeArchive.save(
            tmp_path,
            episode_id=episode_id,
            depth=np.full((length, 1, 2, 2), episode_id, dtype=np.uint8),
            base=np.zeros((length, 2), dtype=np.float32),
            action=np.zeros((length, 1), dtype=np.float32),
            reward=np.zeros(length, dtype=np.float32),
            terminated=np.asarray([False, False, True]),
            truncated=np.zeros(length, dtype=bool),
        )

    original_load = module.np.load
    opened_paths = []

    def counted_load(path, *args, **kwargs):
        opened_paths.append(Path(path))
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(module.np, "load", counted_load)
    dataset = module.EpisodeSequenceDataset(
        tmp_path, sequence_length=None, split="all"
    )

    assert len(opened_paths) == 2
    assert dataset[0]["depth"].shape == (3, 1, 2, 2)
    assert dataset[1]["depth"].flatten()[0].item() == 1
    assert len(opened_paths) == 2


def test_config_projects_one_reconstruction_code_per_offset():
    config_module = _load_ssvm_package_module("config")
    config_path = Path(__file__).parents[1] / "algorithm" / "SSVM_SAC" / "params.yaml"

    config = config_module.SSVMConfig.load(config_path)

    assert config.memory_dim == 128
    assert config.reconstruction_offsets == (-20, -10, 0, 10)
    assert config.reconstruction_loss == "mse"
    assert config.charbonnier_epsilon == pytest.approx(1e-3)
    assert config.reconstruction_latent_dim > 0
    assert config.reconstruction_projection_dim == (
        len(config.reconstruction_offsets) * config.reconstruction_latent_dim
    )
    assert config.vision_batch_size > 0
    assert config.memory_batch_size > 0
    assert config.batch_size > 0


def test_final_stage_curve_logger_writes_vssm_sac_compatible_csv(tmp_path):
    metrics_module = _load_ssvm_package_module("metrics")
    logger = metrics_module.FinalStageCurveLogger(
        algorithm_name="SSVM-SAC",
        seed=25,
        results_root=tmp_path,
    )

    logger.record(
        episode=3,
        total_timesteps=1200,
        reward=18.5,
        episode_length=97,
        success_rate=0.625,
    )

    with logger.csv_path.open(newline="") as stream:
        rows = list(csv.reader(stream))
    assert logger.run_dir == tmp_path / "SSVM-SAC" / "seed25"
    assert rows == [
        ["episode", "total_timesteps", "reward", "episode_length", "success_rate"],
        ["3", "1200", "18.5", "97", "0.625"],
    ]


def test_final_stage_tensorboard_uses_the_formal_results_run_directory(tmp_path):
    train_module = _load_ssvm_package_module("train")
    metrics_module = _load_ssvm_package_module("metrics")
    logger = metrics_module.FinalStageCurveLogger(
        algorithm_name="CL-SSVM-SAC",
        seed=25,
        results_root=tmp_path,
    )

    assert train_module._event_log_dir(tmp_path / "output", logger) == logger.run_dir
    assert train_module._event_log_dir(tmp_path / "output", None) == (
        tmp_path / "output" / "tensorboard"
    )


def test_shared_curve_plotter_recognizes_ssvm_sac_name():
    from algo_name_utils import expand_algorithm_spec, to_output_algorithm_name

    assert expand_algorithm_spec("SSVM-SAC") == ["SSVM_SAC"]
    assert to_output_algorithm_name("CL-SSVM_SAC") == "CL-SSVM-SAC"


def test_final_sac_stage_uses_curriculum_learning_by_default():
    train_module = _load_ssvm_package_module("train")

    args = train_module.build_parser().parse_args(
        ["sac", "--perception-checkpoint", "memory.pt"]
    )

    assert args.curriculum is True
    assert args.max_steps == 150_000
    assert args.steps_per_update == 100
    assert args.gradient_steps == 0.5


def test_offline_stage_batch_overrides_default_to_stage_specific_config():
    train_module = _load_ssvm_package_module("train")

    vision_args = train_module.build_parser().parse_args(
        ["vision", "--dataset", "dataset"]
    )
    memory_args = train_module.build_parser().parse_args(
        ["memory", "--dataset", "dataset", "--vision-checkpoint", "vision.pt"]
    )

    assert vision_args.batch_size is None
    assert memory_args.batch_size is None


def test_bootstrap_and_collection_default_to_level_2():
    train_module = _load_ssvm_package_module("train")

    bootstrap_args = train_module.build_parser().parse_args(["bootstrap"])
    collect_args = train_module.build_parser().parse_args(
        ["collect", "--policy-checkpoint", "bootstrap.pt", "--dataset", "dataset"]
    )

    assert bootstrap_args.level == 2
    assert collect_args.level == 2


def test_memory_stage_uses_new_offsets_with_an_existing_vision_checkpoint():
    train_module = _load_ssvm_package_module("train")
    config_module = _load_ssvm_package_module("config")
    vision_checkpoint_config = config_module.SSVMConfig(
        reconstruction_offsets=(-2, -1, 1)
    ).to_dict()
    requested_config = config_module.SSVMConfig(
        reconstruction_offsets=(-5, -10, 5),
        memory_dim=192,
        reconstruction_loss="charbonnier",
        charbonnier_epsilon=2e-3,
    )

    memory_config = train_module._memory_stage_config(
        vision_checkpoint_config, requested_config
    )

    assert memory_config.reconstruction_offsets == (-5, -10, 5)
    assert memory_config.reconstruction_loss == "charbonnier"
    assert memory_config.charbonnier_epsilon == pytest.approx(2e-3)


def test_charbonnier_reconstruction_loss_matches_definition():
    train_module = _load_ssvm_package_module("train")
    prediction = torch.tensor([0.0, 0.25, 1.0], requires_grad=True)
    target = torch.tensor([0.0, 0.5, 0.0])
    epsilon = 0.1

    loss = train_module._reconstruction_loss(
        prediction,
        target,
        "charbonnier",
        charbonnier_epsilon=epsilon,
    )
    expected = (
        torch.sqrt((prediction - target).square() + epsilon**2) - epsilon
    ).mean()

    assert torch.allclose(loss, expected)
    loss.backward()
    assert torch.isfinite(prediction.grad).all()


def test_memory_offset_losses_are_summed_without_averaging():
    train_module = _load_ssvm_package_module("train")
    losses = [torch.tensor(1.0), torch.tensor(2.0), torch.tensor(3.0)]

    total = train_module._sum_memory_offset_losses(losses)

    assert total.item() == pytest.approx(6.0)


def test_epoch_metrics_weight_unequal_batches_by_pixel_count():
    module = _load_ssvm_package_module("train")
    stats = module.ReconstructionStats()
    stats.add(torch.ones(3), torch.zeros(3))
    stats.add(torch.tensor([3.0]), torch.zeros(1))
    assert stats.sse == 12
    assert stats.mse == 3


def test_validation_restores_training_mode_and_excludes_missing_targets():
    module = _load_ssvm_package_module("train")
    config = _load_ssvm_package_module("config").SSVMConfig(reconstruction_offsets=(-1, 1))
    encoder = nn.Flatten(start_dim=1).eval()
    memory = nn.Identity().train()

    class Decoder(nn.Module):
        def forward(self, latent):
            return {offset: torch.zeros((*latent.shape[:2], 1, 1, 1)) for offset in (-1, 1)}

    decoder = Decoder().train()
    batch = {"depth": torch.zeros(1, 3, 1, 1, 1),
             "target_depth": torch.tensor([255., 255., 0.]).reshape(1, 3, 1, 1, 1),
             "valid": torch.tensor([[True, True, False]])}
    stats = module._validate_reconstruction([batch], encoder, decoder, torch.device("cpu"), config, memory)
    assert all(item.pixels == 1 and item.mse == 1 for item in stats.values())
    assert not encoder.training
    assert memory.training and decoder.training


def test_mse_reconstruction_sums_pixels_samples_and_gradients():
    module = _load_ssvm_package_module("train")
    prediction = torch.tensor([[[[1.0, 2.0]]], [[[3.0, 4.0]]]], requires_grad=True)
    target = torch.zeros_like(prediction)
    loss = module._reconstruction_loss(prediction, target, "mse")
    assert loss.item() == pytest.approx(30.0)
    loss.backward()
    assert torch.equal(prediction.grad, 2 * prediction.detach())


def test_memory_cosine_schedule_uses_yaml_ratio_and_epoch_horizon():
    module = _load_ssvm_package_module("train")
    config_module = _load_ssvm_package_module("config")
    old = config_module.SSVMConfig(memory_lrf=0.5).to_dict()
    requested = config_module.SSVMConfig(memory_lr=0.02, memory_lrf=0.01)
    config = module._memory_stage_config(old, requested)
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([parameter], lr=config.memory_lr)
    scheduler = module._memory_lr_scheduler(optimizer, config, 200)
    rates = [optimizer.param_groups[0]["lr"]]
    for _ in range(200):
        optimizer.step()
        scheduler.step()
        rates.append(optimizer.param_groups[0]["lr"])
    assert rates[0] == pytest.approx(0.02)
    assert rates[100] == pytest.approx(0.0101)
    assert rates[200] == pytest.approx(0.0002)
    assert all(a >= b for a, b in zip(rates, rates[1:]))


def test_latent_objective_masks_padding_and_never_calls_decoder():
    module = _load_ssvm_package_module("train")
    memory = nn.Linear(2, 2)
    reconstructor = nn.Module()
    reconstructor.projection = nn.Linear(2, 4)
    offsets = (-1, 1)
    batch = {"latent": torch.randn(2, 4, 2),
             "valid": torch.tensor([[True]*4, [True, True, False, False]])}
    stats = {k: module.ReconstructionStats() for k in offsets}
    loss = module._memory_latent_objective(memory, reconstructor, batch, "cpu", offsets, stats)
    modified = {**batch, "latent": batch["latent"].clone()}
    modified["latent"][1, 2:] = 999
    stats2 = {k: module.ReconstructionStats() for k in offsets}
    other = module._memory_latent_objective(memory, reconstructor, modified, "cpu", offsets, stats2)
    assert torch.allclose(loss, other)
    assert all(s.pixels == 8 for s in stats.values())
    loss.backward()
    assert memory.weight.grad.abs().sum() > 0
    assert reconstructor.projection.weight.grad.abs().sum() > 0


def test_random_memory_batches_have_complete_targets_and_causal_prefixes():
    module = _load_ssvm_package_module("train")
    cached = [{"latent": torch.arange(n).float().unsqueeze(-1)+100*i}
              for i, n in enumerate([2, 6, 8])]
    offsets = (-2, -1, 1)
    anchors = module._valid_memory_anchors(cached, offsets)
    assert anchors == [(1, t) for t in range(2, 5)] + [(2, t) for t in range(2, 7)]
    batch = module._sample_memory_batch(cached, anchors, 100, offsets, np.random.default_rng(25))
    assert batch["targets"].shape == (100, 3, 1)
    for i, length in enumerate(batch["lengths"]):
        current = batch["latent"][i, length-1, 0]
        assert torch.equal(batch["targets"][i, :, 0], current+torch.tensor(offsets))
        assert batch["latent"][i, :length, 0].max() == current
        assert torch.all(batch["latent"][i, length:] == 0)
    with pytest.raises(ValueError):
        module._sample_memory_batch(cached, [], 2, offsets, np.random.default_rng(25))


def test_sampled_memory_objective_only_supervises_selected_current_times():
    module = _load_ssvm_package_module("train")
    memory = nn.Identity()
    reconstructor = nn.Module()
    reconstructor.projection = nn.Linear(1, 3, bias=False)
    with torch.no_grad():
        reconstructor.projection.weight.fill_(1)
    batch = {"latent": torch.tensor([[[1.], [2.], [0.]], [[3.], [4.], [5.]]]),
             "lengths": torch.tensor([2, 3]), "targets": torch.zeros(2, 3, 1)}
    stats = {k: module.ReconstructionStats() for k in (-2, -1, 1)}
    loss = module._sampled_memory_objective(memory, reconstructor, batch, "cpu", (-2, -1, 1), stats)
    assert loss.item() == pytest.approx(43.5)
    assert all(s.pixels == 2 for s in stats.values())
    loss.backward()
    assert reconstructor.projection.weight.grad is not None


def test_image_memory_objective_uses_clean_targets_and_trains_decoder():
    module = _load_ssvm_package_module("train")
    config = _load_ssvm_package_module("config").SSVMConfig(reconstruction_offsets=(-1, 1))
    encoder = nn.Sequential(nn.Flatten(), nn.Linear(1, 2))
    encoder.requires_grad_(False)
    memory = nn.Linear(2, 2)
    class Reconstructor(nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder = nn.Linear(2, 2)
        def forward(self, state):
            out = self.decoder(state)
            return {-1: out[:, 0].reshape(-1, 1, 1, 1), 1: out[:, 1].reshape(-1, 1, 1, 1)}
    reconstructor = Reconstructor()
    samples = [{"depth": torch.ones(4, 1, 1, 1, dtype=torch.uint8),
                "target_depth": torch.full((4, 1, 1, 1), 200, dtype=torch.uint8)}]
    batch = module._gather_memory_batch(samples, [(0, 1), (0, 2)], (-1, 1))
    assert torch.all(batch["targets"] == 200)
    stats = {k: module.ReconstructionStats() for k in (-1, 1)}
    loss = module._sampled_image_objective(encoder, memory, reconstructor, batch, torch.device("cpu"), config, stats)
    assert loss.item() == pytest.approx(sum(s.sse for s in stats.values()))
    loss.backward()
    assert all(p.grad is None for p in encoder.parameters())
    assert memory.weight.grad is not None
    assert reconstructor.decoder.weight.grad is not None


def test_memory_stage_overrides_old_loss_with_requested_mse():
    module = _load_ssvm_package_module("train")
    config_module = _load_ssvm_package_module("config")
    old = config_module.SSVMConfig(reconstruction_loss="charbonnier").to_dict()
    result = module._memory_stage_config(old, config_module.SSVMConfig())
    assert result.reconstruction_loss == "mse"


def test_memory_stage_uses_requested_learning_rate_and_decay():
    module = _load_ssvm_package_module("train")
    config_module = _load_ssvm_package_module("config")
    old = config_module.SSVMConfig(
        latent_dim=64, memory_lr=0.004, offline_lr_decay=0.95,
    ).to_dict()
    requested = config_module.SSVMConfig(
        latent_dim=128, memory_lr=0.04, offline_lr_decay=0.99,
    )
    result = module._memory_stage_config(old, requested)
    assert result.latent_dim == 64
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([parameter], lr=result.memory_lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=result.offline_lr_decay,
    )
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.04)
    optimizer.step()
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0396)


def test_memory_visualizer_selects_a_time_where_every_offset_is_valid():
    module = _load_ssvm_package_module("visualization.visualize_memory_reconstruction")

    assert module._valid_time_bounds(100, (-10, -20, 10)) == (20, 89)
    assert module._select_time_index(100, (-10, -20, 10), None) == 54
    assert module._display_offsets((-10, -20, 10)) == (-20, -10, 0, 10)

    with pytest.raises(ValueError, match="valid range is 20..89"):
        module._select_time_index(100, (-10, -20, 10), 10)


def test_memory_visualizer_writes_two_by_four_grid(tmp_path):
    module = _load_ssvm_package_module("visualization.visualize_memory_reconstruction")
    offsets = (-20, -10, 0, 10)
    originals = {
        offset: np.full((1, 8, 8), index / 4, dtype=np.float32)
        for index, offset in enumerate(offsets)
    }
    reconstructions = {
        offset: np.clip(image + 0.01, 0.0, 1.0)
        for offset, image in originals.items()
    }

    output = module._save_grid(
        originals,
        reconstructions,
        tmp_path / "comparison.png",
        episode_index=2,
        time_index=40,
    )

    assert output.is_file()
    assert output.stat().st_size > 0


def test_memory_visualizer_uses_project_training_paths_without_arguments():
    module = _load_ssvm_package_module("visualization.visualize_memory_reconstruction")

    args = module.build_parser().parse_args([])

    project_root = Path(__file__).parents[1]
    assert Path(args.dataset) == project_root / "datasets" / "SSVM_SAC"
    assert Path(args.vision_checkpoint) == (
        project_root / "runs" / "SSVM_SAC" / "vision" / "vision_latest.pt"
    )
    assert Path(args.memory_checkpoint) == (
        project_root / "runs" / "SSVM_SAC" / "memory" / "memory_latest.pt"
    )


def test_episode_summary_matches_main_async_format():
    train_module = _load_ssvm_package_module("train")

    summary = train_module._format_episode_summary(
        display_name="bootstrap",
        episode=5,
        reward=-146.791,
        episode_length=198,
        success_rate=0.0,
        level=0,
        total_timesteps=1098,
        total_successes=0,
    )

    assert summary == (
        "[bootstrap] Episode 5, Reward: -146.79, Length: 198, "
        "Success Rate: 0.000, Level: 0, Total Timesteps: 1098, "
        "Total Successes: 0"
    )


class _FakeMamba(nn.Module):
    def __init__(self, d_model, **_):
        super().__init__()
        self.d_model = d_model

    def forward(self, values):
        return values

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None):
        del max_seqlen
        return (
            torch.zeros(batch_size, self.d_model, 1, dtype=dtype),
            torch.zeros(batch_size, self.d_model, 1, dtype=dtype),
        )

    def step(self, values, conv_state, ssm_state):
        return values, conv_state, ssm_state


def test_temporal_memory_online_step_matches_full_causal_forward():
    TemporalMambaMemory = _load_ssvm_module("networks").TemporalMambaMemory

    torch.manual_seed(3)
    memory = TemporalMambaMemory(
        latent_dim=4,
        memory_dim=4,
        depth=2,
        mamba_factory=_FakeMamba,
    )
    latent = torch.randn(2, 5, 4)

    full = memory(latent)
    assert not hasattr(memory, "input_projection")
    assert full.shape == (2, 5, 4)
    cache = memory.reset(batch_size=2, device=latent.device, dtype=latent.dtype)
    online = []
    for index in range(latent.shape[1]):
        value, cache = memory.step(latent[:, index], cache)
        online.append(value)

    assert torch.allclose(full, torch.stack(online, dim=1), atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="real Mamba requires a CUDA driver")
def test_real_mamba_online_step_matches_full_causal_forward_on_gpu():
    TemporalMambaMemory = _load_ssvm_module("networks").TemporalMambaMemory
    try:
        memory = TemporalMambaMemory(
            latent_dim=8, memory_dim=8, depth=1, d_state=4, d_conv=3
        ).cuda().eval()
    except (ImportError, RuntimeError) as error:
        pytest.skip(f"Mamba CUDA extension is unavailable: {error}")
    latent = torch.randn(2, 7, 8, device="cuda")
    with torch.no_grad():
        full = memory.forward_sequence(latent)
        cache = memory.reset(2, latent.device, latent.dtype)
        online = []
        for index in range(7):
            value, cache = memory.step(latent[:, index], cache)
            online.append(value)

    assert torch.allclose(full, torch.stack(online, dim=1), atol=2e-4, rtol=2e-4)


def test_vision_mamba_decoder_unpatchifies_one_reconstruction_latent():
    VisionMambaDecoder = _load_ssvm_module("networks").VisionMambaDecoder

    decoder = VisionMambaDecoder(
        input_dim=8,
        image_size=(8, 12),
        patch_size=4,
        channels=1,
        embed_dim=8,
        depth=1,
        mamba_factory=_FakeMamba,
    )
    latent = torch.randn(2, 3, 8)

    reconstruction = decoder(latent)

    assert reconstruction.shape == (2, 3, 1, 8, 12)
    assert reconstruction.min() >= 0.0
    assert reconstruction.max() <= 1.0
    assert decoder(latent[:, 0]).shape == (2, 1, 8, 12)


def test_linear_patch_expand_places_children_in_spatial_order():
    module = _load_ssvm_module("networks")
    layer = module._LinearPatchExpand(1)
    with torch.no_grad():
        layer.projection.weight.fill_(1)
        layer.projection.bias.copy_(torch.arange(4))
    result = layer(torch.tensor([[[0.], [10.]]]), 1, 2).reshape(2, 4)
    assert torch.equal(result, torch.tensor([[0., 1., 10., 11.], [2., 3., 12., 13.]]))


def test_pyramid_decoder_backpropagates_through_every_resolution():
    module = _load_ssvm_module("networks")
    decoder = module.VisionMambaDecoder(8, (128, 128), patch_size=4,
        embed_dim=4, depth=1, mamba_factory=_FakeMamba)
    assert decoder.stage_grids == ((4, 4), (8, 8), (16, 16), (32, 32))
    assert decoder.input_projection.out_features == 4 * 4 * 4
    latent = torch.randn(2, 8, requires_grad=True)
    output = decoder(latent)
    assert output.shape == (2, 1, 128, 128)
    output.square().sum().backward()
    assert latent.grad is not None and torch.isfinite(latent.grad).all()
    for expand in decoder.upsamplers:
        assert expand.projection.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("offsets", [(-5, -10, 5), (-20, -10, 0, 10)])
def test_reconstruction_latents_use_one_shared_decoder(offsets):
    module = _load_ssvm_module("networks")
    shared_decoder = module.VisionMambaDecoder(
        input_dim=4,
        image_size=(8, 8),
        patch_size=4,
        channels=1,
        embed_dim=8,
        depth=1,
        mamba_factory=_FakeMamba,
    )
    reconstructor = module.MultiFrameMambaReconstructor(
        memory_dim=4 * len(offsets),
        reconstruction_latent_dim=4,
        offsets=offsets,
        decoder=shared_decoder,
    )

    features = torch.randn(2, 5, 6, requires_grad=True)
    reconstructions = reconstructor(features)

    assert reconstructor.projection.out_features == 4 * len(offsets)
    assert reconstructor.decoder is shared_decoder
    assert set(reconstructions) == set(offsets)
    assert all(value.shape == (2, 5, 1, 8, 8) for value in reconstructions.values())
    if 0 in offsets:
        reconstructions[0].square().sum().backward()


def test_direct_split_checkpoint_version_compatibility(tmp_path):
    module = _load_ssvm_package_module("checkpoints")
    path = tmp_path / "checkpoint.pt"
    for stage, version, compatible in [("vision", 3, True), ("memory", 3, False), ("memory", 4, True)]:
        torch.save({"stage": stage, "model_version": version, "config": {}}, path)
        if compatible:
            assert module.load_checkpoint(path, "cpu")["stage"] == stage
        else:
            with pytest.raises(ValueError, match="Retrain Memory"):
                module.load_checkpoint(path, "cpu")


def test_feature_replay_buffer_returns_fixed_representation_transitions():
    FeatureReplayBuffer = _load_ssvm_package_module("buffer").FeatureReplayBuffer

    replay = FeatureReplayBuffer(capacity=8, base_dim=2, memory_dim=3, action_dim=1, seed=7)
    for value in range(4):
        replay.add(
            np.full(2, value),
            np.full(3, value),
            np.full(1, value),
            float(value),
            np.full(2, value + 1),
            np.full(3, value + 1),
            value == 3,
        )

    batch = replay.sample(3, torch.device("cpu"))

    assert len(replay) == 4
    assert batch.base.shape == (3, 2)
    assert batch.memory.shape == (3, 3)
    assert batch.done.dtype == torch.float32


def test_dual_prioritized_feature_replay_routes_complete_episodes_by_success():
    buffer_module = _load_ssvm_package_module("buffer")
    replay = buffer_module.DualPrioritizedFeatureReplayBuffer(
        capacity=20,
        base_dim=1,
        memory_dim=2,
        action_dim=1,
        success_capacity_ratio=0.5,
        success_sample_ratio=0.5,
        seed=7,
    )

    def add(value, *, done=False, is_success=False):
        replay.add(
            np.array([value], dtype=np.float32),
            np.full(2, value, dtype=np.float32),
            np.array([value], dtype=np.float32),
            float(value),
            np.array([value + 1], dtype=np.float32),
            np.full(2, value + 1, dtype=np.float32),
            done,
            is_success=is_success,
        )

    add(1)
    assert len(replay) == 0
    add(2, done=True)
    add(10)
    add(11, is_success=True)
    add(12, done=True)

    assert replay.regular_buffer.size == 2
    assert replay.success_buffer.size == 3
    batch, refs, weights, info = replay.sample(
        4, torch.device("cpu"), beta=0.4
    )

    assert batch.base.shape == (4, 1)
    assert len(refs) == 4
    assert weights.shape == (4, 1)
    assert info == {
        "batch_success_fraction": 0.5,
        "success_size": 3,
        "regular_size": 2,
    }
    for value, (pool_name, _) in zip(batch.base[:, 0].tolist(), refs):
        if pool_name == "success":
            assert value >= 10
        else:
            assert value < 10


def test_dual_prioritized_feature_replay_updates_priorities_in_origin_pool():
    buffer_module = _load_ssvm_package_module("buffer")
    replay = buffer_module.DualPrioritizedFeatureReplayBuffer(
        capacity=10,
        base_dim=1,
        memory_dim=1,
        action_dim=1,
        seed=3,
    )
    transition = (
        np.zeros(1, dtype=np.float32),
        np.zeros(1, dtype=np.float32),
        np.zeros(1, dtype=np.float32),
        0.0,
        np.ones(1, dtype=np.float32),
        np.ones(1, dtype=np.float32),
        True,
    )
    replay.add(*transition, is_success=True)
    replay.add(*transition, is_success=False)

    replay.update_priorities(
        [("success", 0), ("regular", 0)],
        np.array([2.0, 3.0], dtype=np.float32),
    )

    assert replay.success_buffer.priorities[0] == pytest.approx(2.0 + replay.eps)
    assert replay.regular_buffer.priorities[0] == pytest.approx(3.0 + replay.eps)


class _FakePerception(nn.Module):
    def __init__(self, memory_dim):
        super().__init__()
        self.anchor = nn.Parameter(torch.ones(()))
        self.memory_dim = memory_dim

    def freeze(self):
        self.eval().requires_grad_(False)
        return self

    def reset(self, batch_size=1):
        self.batch_size = batch_size

    def step(self, frame):
        return torch.ones(frame.shape[0], self.memory_dim, device=frame.device)


def test_sac_update_uses_frozen_memory_features_without_encoder_gradients():
    agent_module = _load_ssvm_package_module("agent")
    config_module = _load_ssvm_package_module("config")
    config = config_module.SSVMConfig(
        memory_dim=3,
        hidden_dim=16,
        replay_capacity=16,
        batch_size=4,
    )
    agent = agent_module.SSVMSACAgent(
        _FakePerception(3),
        base_dim=2,
        action_low=np.array([-1.0], dtype=np.float32),
        action_high=np.array([1.0], dtype=np.float32),
        config=config,
        device="cpu",
    )
    memory = np.ones(3, dtype=np.float32)
    assert agent._success_sample_ratio(0.0) == pytest.approx(0.30)
    assert agent._success_sample_ratio(0.25) == pytest.approx(0.40)
    assert agent._success_sample_ratio(0.70) == pytest.approx(0.45)
    for index in range(8):
        agent.add_transition(
            np.zeros(2, dtype=np.float32),
            memory,
            np.zeros(1, dtype=np.float32),
            1.0,
            np.ones(2, dtype=np.float32),
            memory,
            index == 7,
            is_success=index == 7,
        )

    metrics = agent.update(progress_ratio=0.8)

    assert {
        "actor_loss",
        "critic_loss",
        "alpha_loss",
        "alpha",
        "sb_per_beta",
        "replay/success_sample_ratio_target",
        "replay/success_batch_fraction",
        "replay/success_size",
        "replay/regular_size",
    } == set(metrics)
    assert metrics["sb_per_beta"] == pytest.approx(0.88)
    assert metrics["replay/success_sample_ratio_target"] == pytest.approx(0.45)
    assert metrics["replay/success_batch_fraction"] == pytest.approx(1.0)
    assert metrics["replay/success_size"] == 5
    assert metrics["replay/regular_size"] == 0
    assert all(parameter.grad is None for parameter in agent.perception.parameters())
