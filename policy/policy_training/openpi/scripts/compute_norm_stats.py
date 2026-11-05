"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

import numpy as np
import tqdm
import tyro

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    # Tactile is deliberately excluded from norm stats. Preserve the same
    # episode/action-window dataset while skipping unused visual/tactile I/O.
    dataset = _data_loader.create_torch_dataset(
        data_config, action_horizon, model_config, load_tactile=False, load_videos=False
    )
    if data_config.tactile_sidecar_root is not None:
        # This ACWM tactile config uses adapt_to_pi=False; its image resize and
        # tactile pass-through cannot alter state/actions. Repack just the two
        # fields accumulated below so normalization is numerically identical
        # without decoding or resizing unused images.
        input_transforms = [
            transforms.RepackTransform(
                {
                    "state": "observation.state",
                    "actions": "action",
                }
            )
        ]
    else:
        input_transforms = [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
        ]
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *input_transforms,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def create_tactile_acwm_array_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
) -> tuple[object, int]:
    """Vectorized state/action batches for the tactile ACWM normalizer.

    This preserves the exact sequential frame order, batch size, episode-tail
    clamp, and drop-last behavior of LeRobot + TorchDataLoader while avoiding
    per-sample Python dispatch for modalities that norm stats never consume.
    """
    if tuple(data_config.action_sequence_keys) != ("action",):
        raise ValueError(
            "Vectorized tactile norm stats require action_sequence_keys=('action',), "
            f"got {data_config.action_sequence_keys}"
        )
    dataset = _data_loader.create_torch_dataset(
        data_config, action_horizon, model_config, load_tactile=False, load_videos=False
    )
    base_dataset = dataset
    while hasattr(base_dataset, "_dataset"):
        base_dataset = base_dataset._dataset
    if not hasattr(base_dataset, "hf_dataset") or not hasattr(base_dataset, "episode_data_index"):
        raise TypeError(f"Expected a LeRobot dataset, got {type(base_dataset)!r}")

    columns = ["observation.state", "action", "episode_index"]
    arrays = base_dataset.hf_dataset.with_format(
        "numpy", columns=columns, output_all_columns=False
    )[:]
    states = np.asarray(arrays["observation.state"], dtype=np.float32)
    actions = np.asarray(arrays["action"], dtype=np.float32)
    episode_indices = np.asarray(arrays["episode_index"], dtype=np.int64).reshape(-1)
    if states.ndim != 2 or actions.ndim != 2 or len(states) != len(actions):
        raise ValueError(
            f"Unexpected ACWM arrays: state={states.shape}, action={actions.shape}, episodes={episode_indices.shape}"
        )
    if len(episode_indices) != len(states):
        raise ValueError("episode_index length does not match state/action arrays")

    episode_to = np.asarray(base_dataset.episode_data_index["to"], dtype=np.int64)
    offsets = np.arange(action_horizon, dtype=np.int64)[None, :]

    def make_batch(start: int, stop: int) -> dict[str, np.ndarray]:
        positions = np.arange(start, stop, dtype=np.int64)
        ends = episode_to[episode_indices[start:stop]]
        query_indices = np.minimum(positions[:, None] + offsets, ends[:, None] - 1)
        return {"state": states[start:stop], "actions": actions[query_indices]}

    # Guard the vectorized episode-tail logic against LeRobot itself at the
    # beginning, middle, and final retained episode ranges.
    for index in (0, len(states) // 2, len(states) - 1):
        expected = base_dataset[index]
        actual = make_batch(index, index + 1)
        np.testing.assert_array_equal(actual["state"][0], np.asarray(expected["observation.state"]))
        np.testing.assert_array_equal(actual["actions"][0], np.asarray(expected["action"]))

    num_batches = len(states) // batch_size

    def batches():
        for batch_index in range(num_batches):
            start = batch_index * batch_size
            yield make_batch(start, start + batch_size)

    return batches(), num_batches


def create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    dataset = _data_loader.create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=False)
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
        is_batched=True,
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        # NOTE: this length is currently hard-coded for DROID.
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def main(config_name: str, max_frames: int | None = None):
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)

    if data_config.rlds_data_dir is not None:
        data_loader, num_batches = create_rlds_dataloader(
            data_config, config.model.action_horizon, config.batch_size, max_frames
        )
    elif data_config.tactile_sidecar_root is not None:
        data_loader, num_batches = create_tactile_acwm_array_dataloader(
            data_config, config.model.action_horizon, config.batch_size, config.model
        )
    else:
        data_loader, num_batches = create_torch_dataloader(
            data_config, config.model.action_horizon, config.batch_size, config.model, config.num_workers, max_frames
        )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    output_path = config.assets_dirs / data_config.repo_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
