import dataclasses
import json

import jax
import numpy as np
import pytest
import torch

from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


class _TinySidecarBaseDataset:
    def __init__(self):
        self._samples = [
            {"episode_index": np.int64(7), "frame_index": np.int64(0), "value": np.int64(10)},
            {"episode_index": np.int64(7), "frame_index": np.int64(1), "value": np.int64(11)},
        ]

    def __getitem__(self, index):
        return self._samples[index]

    def __len__(self):
        return len(self._samples)


def test_tactile_sidecar_dataset(tmp_path):
    episode = np.arange(2 * 4 * 700 * 3, dtype=np.float32).reshape(2, 4, 700, 3)
    (tmp_path / "episodes").mkdir()
    (tmp_path / "masks").mkdir()
    np.save(tmp_path / "episodes" / "source.npy", episode)
    tactile_mask = np.asarray([[True, True, True, True], [True, False, True, False]], dtype=np.bool_)
    np.save(tmp_path / "masks" / "source.npy", tactile_mask)
    (tmp_path / "summary.json").write_text(json.dumps({"complete": True, "mapped_variants": 1}))
    (tmp_path / "episodes.jsonl").write_text(
        json.dumps(
            {
                "global_episode_index": 7,
                "tactile_relpath": "episodes/source.npy",
                "tactile_mask_relpath": "masks/source.npy",
                "frame_count": 2,
            }
        )
        + "\n"
    )

    dataset = _data_loader.TactileSidecarDataset(_TinySidecarBaseDataset(), str(tmp_path))
    sample = dataset[1]

    np.testing.assert_array_equal(sample["observation.tactile"], episode[1])
    np.testing.assert_array_equal(sample["observation.tactile_mask"], tactile_mask[1])
    assert sample["observation.tactile"].flags.owndata
    assert sample["value"] == 11


def test_tactile_sidecar_can_skip_payload_io_for_normalizer(tmp_path):
    (tmp_path / "summary.json").write_text(json.dumps({"complete": True, "mapped_variants": 1}))
    (tmp_path / "episodes.jsonl").write_text(
        json.dumps(
            {
                "global_episode_index": 7,
                "tactile_relpath": "episodes/not_needed.npy",
                "tactile_mask_relpath": "masks/not_needed.npy",
                "frame_count": 2,
            }
        )
        + "\n"
    )

    dataset = _data_loader.TactileSidecarDataset(
        _TinySidecarBaseDataset(), str(tmp_path), load_tactile=False
    )
    sample = dataset[1]

    np.testing.assert_array_equal(sample["observation.tactile"], np.zeros((4, 700, 3), dtype=np.float32))
    np.testing.assert_array_equal(sample["observation.tactile_mask"], np.zeros((4,), dtype=np.bool_))


def test_tactile_sidecar_rejects_incomplete(tmp_path):
    (tmp_path / "summary.json").write_text(json.dumps({"complete": False}))
    (tmp_path / "episodes.jsonl").write_text("")
    with pytest.raises(ValueError, match="not marked complete"):
        _data_loader.TactileSidecarDataset(_TinySidecarBaseDataset(), str(tmp_path))


def test_create_torch_dataset_applies_episode_subset(monkeypatch):
    captured = {}

    class _FakeMetadata:
        total_episodes = 5
        fps = 10

        def __init__(self, repo_id):
            captured["meta_repo_id"] = repo_id
            self.info = {"features": {"observation.images.base_0": {"dtype": "video"}}}

        @property
        def video_keys(self):
            return [key for key, feature in self.info["features"].items() if feature["dtype"] == "video"]

    class _FakeHFDataset:
        column_names = []

    class _FakeLeRobotDataset:
        def __init__(self, repo_id, **kwargs):
            captured["repo_id"] = repo_id
            captured.update(kwargs)
            self.meta = _FakeMetadata(repo_id)
            self.hf_dataset = _FakeHFDataset()
            self.episode_data_index = {
                "from": torch.tensor([0, 10, 20]),
                "to": torch.tensor([10, 20, 30]),
            }

        def __len__(self):
            return 1

    monkeypatch.setattr(_data_loader.lerobot_dataset, "LeRobotDatasetMetadata", _FakeMetadata)
    monkeypatch.setattr(_data_loader.lerobot_dataset, "LeRobotDataset", _FakeLeRobotDataset)
    data_config = _config.DataConfig(repo_id="test/repo", episode_indices=(0, 2, 4))
    model_config = pi0_config.Pi0Config(action_horizon=2)

    dataset = _data_loader.create_torch_dataset(data_config, action_horizon=2, model_config=model_config)

    assert captured["episodes"] == [0, 2, 4]
    torch.testing.assert_close(dataset.episode_data_index["from"], torch.tensor([0, 0, 10, 0, 20]))
    torch.testing.assert_close(dataset.episode_data_index["to"], torch.tensor([10, 0, 20, 0, 30]))

    video_free_dataset = _data_loader.create_torch_dataset(
        data_config, action_horizon=2, model_config=model_config, load_videos=False
    )
    assert video_free_dataset.meta.video_keys == []


@pytest.mark.parametrize("episode_indices", [(), (1, 1), (-1,), (5,)])
def test_create_torch_dataset_rejects_invalid_episode_subset(monkeypatch, episode_indices):
    class _FakeMetadata:
        total_episodes = 5

        def __init__(self, repo_id):
            pass

    monkeypatch.setattr(_data_loader.lerobot_dataset, "LeRobotDatasetMetadata", _FakeMetadata)
    data_config = _config.DataConfig(repo_id="test/repo", episode_indices=episode_indices)

    with pytest.raises(ValueError):
        _data_loader.create_torch_dataset(data_config, action_horizon=2, model_config=pi0_config.Pi0Config())


def test_torch_data_loader():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_torch_data_loader_infinite():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4)
    data_iter = iter(loader)

    for _ in range(10):
        _ = next(data_iter)


def test_torch_data_loader_parallel():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 10)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=2, num_workers=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_with_fake_dataset():
    config = _config.get_config("debug")

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == config.batch_size for x in jax.tree.leaves(batch))

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_with_real_dataset():
    config = _config.get_config("pi0_aloha_sim")
    config = dataclasses.replace(config, batch_size=4)

    loader = _data_loader.create_data_loader(
        config,
        # Skip since we may not have the data available.
        skip_norm_stats=True,
        num_batches=2,
        shuffle=True,
    )
    # Make sure that we can get the data config.
    assert loader.data_config().repo_id == config.data.repo_id

    batches = list(loader)

    assert len(batches) == 2

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)
