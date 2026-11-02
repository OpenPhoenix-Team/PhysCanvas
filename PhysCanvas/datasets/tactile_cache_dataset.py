import copy
import json
import os
import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from giga_datasets import BaseDataset, register_dataset


VIEW_TO_CAMERA = {
    "high": "base_0",
    "left_wrist": "left_wrist_0",
    "right_wrist": "right_wrist_0",
}

DEFAULT_TACTILE_SIDES = ("left_wrist_0", "left_wrist_1", "right_wrist_0", "right_wrist_1")
REMOTE_CACHE_PREFIXES = ("/mnt/oss_models", "/mnt/oss_data")


def _cache_mmap_mode(path: str | Path) -> str | None:
    mode = os.environ.get("PHYSCANVAS_CACHE_MMAP", "auto").strip().lower()
    if mode in ("1", "true", "yes", "on", "r"):
        return "r"
    if mode in ("0", "false", "no", "off", "none"):
        return None
    path_str = os.path.abspath(os.path.expanduser(str(path)))
    if path_str.startswith(REMOTE_CACHE_PREFIXES):
        return None
    return "r"


def _load_npy(path: str | Path, *, mmap: bool | None = None):
    mmap_mode = _cache_mmap_mode(path) if mmap is None else ("r" if mmap else None)
    return np.load(path, mmap_mode=mmap_mode, allow_pickle=False)


def _npy_len(path: str | Path) -> int:
    """Return the first dimension of a .npy file by reading only its header."""
    path = Path(path)
    if not path.exists():
        return 0
    with path.open("rb") as f:
        version = np.lib.format.read_magic(f)
        if version == (1, 0):
            shape, _, _ = np.lib.format.read_array_header_1_0(f)
        elif version in ((2, 0), (3, 0)):
            shape, _, _ = np.lib.format.read_array_header_2_0(f)
        else:
            raise ValueError(f"Unsupported npy version {version} for {path}")
    return int(shape[0]) if shape else 0


def _normalise_roots(cache_root: str | Sequence[str] | None, cache_roots: Sequence[str] | None = None) -> list[str]:
    roots: list[str] = []
    if cache_roots:
        roots.extend(str(x) for x in cache_roots)
    if cache_root:
        if isinstance(cache_root, (list, tuple)):
            roots.extend(str(x) for x in cache_root)
        else:
            roots.extend(str(x) for x in str(cache_root).split(os.pathsep) if str(x).strip())
    return [os.path.abspath(os.path.expanduser(x)) for x in roots if x]


def _load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with path.open("r") as f:
        return json.load(f)


def _episode_id_from_meta(meta: dict[str, Any], episode_dir: Path) -> str:
    source_task = meta.get("source_task") or Path(str(meta.get("source_episode", ""))).parent.name
    episode = str(meta.get("episode") or episode_dir.name.split("__")[-1])
    if source_task:
        return f"{source_task}_{episode}"
    task = str(meta.get("task") or episode_dir.name.split("__")[0])
    return f"{task}_{episode}"


def _available_count(episode_dir: Path, camera_names: dict[str, str]) -> int:
    counts = []
    for camera_name in camera_names.values():
        path = episode_dir / f"camera_{camera_name}_rgb.npy"
        if not path.exists():
            return 0
        counts.append(_npy_len(path))
    return min(counts) if counts else 0


@register_dataset
class TactileCacheDataset(BaseDataset):
    """Read preprocessed ARX tactile/RGB cache episodes.

    The cache root is expected to contain preprocess_config*.json,
    manifest*.json, and episodes/<task>__<episode>/*.npy files. Depth and
    motion are not stored in this cache; the sample transform reads them from
    the cache paths given in the record.
    """

    def __init__(
        self,
        cache_root: str | Sequence[str] | None = None,
        cache_roots: Sequence[str] | None = None,
        depth_cache_dir: str = "",
        motion_cache_dir: str = "",
        motion_cache_variant: str = "arxx5_gripper_filled_gray_055",
        cameras: dict[str, str] | None = None,
        data_size: int | None = None,
        max_episodes: int | None = None,
        max_frames: int | None = None,
        expand_clips: bool = False,
        clip_num_frames: int = 33,
        clip_max_stride: int = 4,
        clip_start_stride: int = 32,
        clip_include_all_strides: bool = False,
        fixed_clip_count: int = 0,
        fixed_clip_seed: int = 2026,
        fixed_clip_manifest_path: str = "",
        fixed_clip_candidate_stride: int = 1,
        fixed_clip_override_num_frames: bool = False,
        fixed_clip_start_mode: str = "random",
        require_depth_cache: bool = False,
        require_motion_cache: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.cache_roots = _normalise_roots(cache_root, cache_roots)
        if not self.cache_roots:
            raise ValueError("TactileCacheDataset requires cache_root or cache_roots")
        self.depth_cache_dir = depth_cache_dir
        self.motion_cache_dir = motion_cache_dir
        self.motion_cache_variant = motion_cache_variant
        self.cameras = cameras or VIEW_TO_CAMERA
        self.data_size = data_size
        self.max_episodes = max_episodes
        self.max_frames = max_frames
        self.expand_clips = bool(expand_clips)
        self.clip_num_frames = int(clip_num_frames)
        self.clip_max_stride = max(1, int(clip_max_stride))
        self.clip_start_stride = max(1, int(clip_start_stride))
        self.clip_include_all_strides = bool(clip_include_all_strides)
        self.fixed_clip_count = max(0, int(fixed_clip_count))
        self.fixed_clip_seed = int(fixed_clip_seed)
        self.fixed_clip_manifest_path = fixed_clip_manifest_path
        self.fixed_clip_candidate_stride = max(1, int(fixed_clip_candidate_stride))
        self.fixed_clip_override_num_frames = bool(fixed_clip_override_num_frames)
        self.fixed_clip_start_mode = str(fixed_clip_start_mode or "random")
        if self.fixed_clip_start_mode not in ("random", "episode_start"):
            raise ValueError(f"Unsupported fixed_clip_start_mode={self.fixed_clip_start_mode!r}")
        self.require_depth_cache = bool(require_depth_cache)
        self.require_motion_cache = bool(require_motion_cache)
        self.data_list: list[dict[str, Any]] | None = None
        self._frame_index_cache: dict[tuple[str, str], set[int]] = {}
        self._array_len_cache: dict[str, int] = {}
        self._file_count_cache: dict[tuple[str, str], int] = {}

    @classmethod
    def load(cls, data_or_config: str | dict) -> "TactileCacheDataset":
        from giga_datasets.datasets.dataset import load_config

        config = load_config(data_or_config)
        kwargs = {k: v for k, v in config.items() if not k.startswith("_")}
        kwargs.pop("config_path", None)
        kwargs.pop("data_path", None)
        return cls(config_path=config.get("config_path"), data_path=config.get("data_path"), **kwargs)

    def _episode_dirs_from_root(self, root: Path) -> list[Path]:
        manifest = _load_json(root / "manifest_expanded.json")
        if manifest is None:
            manifest = _load_json(root / "manifest.json")
        dirs: list[Path] = []
        if isinstance(manifest, list):
            for item in manifest:
                if not isinstance(item, dict):
                    continue
                out = item.get("output")
                if out:
                    path = Path(out)
                else:
                    episode = str(item.get("episode", ""))
                    task = Path(episode).parent.name
                    if task == "huanggua1_0629":
                        task = "peel_huanggua_0629"
                    path = root / "episodes" / f"{task}__{Path(episode).name}"
                if path.is_dir():
                    dirs.append(path)
        episode_root = root / "episodes"
        if not dirs and episode_root.is_dir():
            dirs = [p for p in sorted(episode_root.iterdir()) if p.is_dir()]
        return sorted(dict.fromkeys(dirs))

    def open(self) -> None:
        if self.data_list is not None:
            return
        records: list[dict[str, Any]] = []
        for root_str in self.cache_roots:
            root = Path(root_str)
            preprocess_config = _load_json(root / "preprocess_config_expanded.json", None)
            if preprocess_config is None:
                preprocess_config = _load_json(root / "preprocess_config.json", {})
            for episode_dir in self._episode_dirs_from_root(root):
                if not (episode_dir / "_SUCCESS").exists():
                    continue
                meta = _load_json(episode_dir / "meta.json", {})
                camera_names = {k: v for k, v in self.cameras.items() if (episode_dir / f"camera_{v}_rgb.npy").exists()}
                if set(("high", "left_wrist", "right_wrist")) - set(camera_names):
                    print(f"[TactileCacheDataset] skip {episode_dir}: missing one or more RGB cache views")
                    continue
                num_frames = int(meta.get("num_frames") or _available_count(episode_dir, camera_names))
                num_frames = min(num_frames, _available_count(episode_dir, camera_names))
                if self.max_frames is not None and self.max_frames > 0:
                    num_frames = min(num_frames, int(self.max_frames))
                if num_frames <= 0:
                    continue
                episode_id = _episode_id_from_meta(meta, episode_dir)
                if self.require_depth_cache and not self._has_depth_cache(episode_id, camera_names):
                    print(f"[TactileCacheDataset] skip {episode_id}: missing depth cache")
                    continue
                if self.require_motion_cache and not self._has_motion_cache(episode_id):
                    print(f"[TactileCacheDataset] skip {episode_id}: missing motion cache")
                    continue
                record = {
                    "data_index": len(records),
                    "episode_id": episode_id,
                    "episode_name": str(meta.get("episode") or episode_dir.name.split("__")[-1]),
                    "sample_num_frames": num_frames,
                    "sample_cache_source": "tactile_cache",
                    "sample_cache_root": str(root),
                    "sample_cache_episode_dir": str(episode_dir),
                    "sample_cache_meta": meta,
                    "sample_cache_preprocess_config": preprocess_config,
                    "sample_camera_names": camera_names,
                    "sample_depth_cache_dir": self.depth_cache_dir,
                    "sample_motion": {
                        "cache_dir": self.motion_cache_dir,
                        "cache_variant": self.motion_cache_variant,
                        "use_cache": True,
                    },
                    "sample_view_indices": {
                        view_key: np.arange(num_frames, dtype=np.int64) for view_key in camera_names
                    },
                    "target_timestamps": self._load_timestamps(episode_dir, num_frames),
                    "bucket_index": 0,
                    "video_info": (224, 224),
                }
                records.append(record)
                if self.max_episodes is not None and self.max_episodes > 0 and len(records) >= self.max_episodes:
                    break
            if self.max_episodes is not None and self.max_episodes > 0 and len(records) >= self.max_episodes:
                break

        if self.fixed_clip_count > 0:
            records = self._build_or_load_fixed_clips(records)
        elif self.expand_clips:
            records = self._expand_episode_records_to_clips(records)
        self.data_list = records
        self.data_size = len(records)

    def _load_timestamps(self, episode_dir: Path, num_frames: int) -> np.ndarray:
        path = episode_dir / "timestamps.npy"
        if path.exists():
            return _load_npy(path)[:num_frames].astype(np.int64)
        return np.arange(num_frames, dtype=np.int64)

    def _has_depth_cache(self, episode_id: str, camera_names: dict[str, str]) -> bool:
        if not self.depth_cache_dir:
            return False
        root = Path(self.depth_cache_dir) / "foundation_stereo" / episode_id
        return all(any((root / camera_name).glob("frame_*.npy")) for camera_name in camera_names.values())

    def _has_motion_cache(self, episode_id: str) -> bool:
        if not self.motion_cache_dir:
            return False
        root = Path(self.motion_cache_dir) / self.motion_cache_variant / episode_id
        return all(any((root / view_key).glob("h*_w*_frame_*.png")) for view_key in ("high", "left_wrist", "right_wrist"))

    def _array_len(self, path: Path) -> int:
        key = str(path)
        if key not in self._array_len_cache:
            if not path.exists():
                self._array_len_cache[key] = 0
            else:
                self._array_len_cache[key] = _npy_len(path)
        return self._array_len_cache[key]

    @staticmethod
    def _frame_index_from_name(path: Path) -> int | None:
        stem = path.stem
        marker = "frame_"
        pos = stem.rfind(marker)
        if pos < 0:
            return None
        value = stem[pos + len(marker):]
        return int(value) if value.isdigit() else None

    def _frame_indices(self, directory: Path, pattern: str) -> set[int]:
        key = (str(directory), pattern)
        if key not in self._frame_index_cache:
            indices = set()
            for path in directory.glob(pattern):
                index = self._frame_index_from_name(path)
                if index is not None:
                    indices.add(index)
            self._frame_index_cache[key] = indices
        return self._frame_index_cache[key]

    def _file_count(self, directory: Path, pattern: str) -> int:
        key = (str(directory), pattern)
        if key not in self._file_count_cache:
            self._file_count_cache[key] = sum(1 for _ in directory.glob(pattern))
        return self._file_count_cache[key]

    @staticmethod
    def _clip_indexes(start: int, stride: int, num_frames: int) -> np.ndarray:
        return int(start) + int(stride) * np.arange(int(num_frames), dtype=np.int64)

    def _has_tactile_cache_for_indexes(self, record: dict[str, Any], indexes: np.ndarray) -> bool:
        episode_dir = Path(record["sample_cache_episode_dir"])
        meta = record.get("sample_cache_meta", {})
        tactile_side = meta.get("primary_tactile_side") or meta.get("tactile_side") or "left_wrist_0"
        point_path = episode_dir / f"point_{tactile_side}_xy.npy"
        if not point_path.exists():
            point_path = episode_dir / "point_xy.npy"
        diff_path = episode_dir / f"tactile_{tactile_side}_diff.npy"
        if not point_path.exists() or not diff_path.exists():
            return False
        max_index = int(np.max(indexes)) if indexes.size else -1
        return self._array_len(point_path) > max_index and self._array_len(diff_path) > max_index

    def _has_rgb_cache_for_indexes(self, record: dict[str, Any], indexes: np.ndarray) -> bool:
        episode_dir = Path(record["sample_cache_episode_dir"])
        max_index = int(np.max(indexes)) if indexes.size else -1
        for camera_name in record["sample_camera_names"].values():
            if self._array_len(episode_dir / f"camera_{camera_name}_rgb.npy") <= max_index:
                return False
        return True

    def _has_depth_cache_for_indexes(self, record: dict[str, Any], indexes: np.ndarray) -> bool:
        if not self.depth_cache_dir:
            return False
        root = Path(self.depth_cache_dir) / "foundation_stereo" / record["episode_id"]
        max_index = int(np.max(indexes)) if indexes.size else -1
        for camera_name in record["sample_camera_names"].values():
            if self._file_count(root / camera_name, "frame_*.npy") <= max_index:
                return False
        return True

    def _has_motion_cache_for_indexes(self, record: dict[str, Any], indexes: np.ndarray) -> bool:
        if not self.motion_cache_dir:
            return False
        root = Path(self.motion_cache_dir) / self.motion_cache_variant / record["episode_id"]
        max_index = int(np.max(indexes)) if indexes.size else -1
        for view_key in ("high", "left_wrist", "right_wrist"):
            if self._file_count(root / view_key, "h*_w*_frame_*.png") <= max_index:
                return False
        return True

    def _clip_has_required_cache(self, record: dict[str, Any], start: int, stride: int, num_frames: int) -> bool:
        indexes = self._clip_indexes(start, stride, num_frames)
        if indexes.size == 0:
            return False
        if int(indexes[-1]) >= int(record.get("sample_num_frames", 0)):
            return False
        if not self._has_rgb_cache_for_indexes(record, indexes):
            return False
        if not self._has_tactile_cache_for_indexes(record, indexes):
            return False
        if self.require_depth_cache and not self._has_depth_cache_for_indexes(record, indexes):
            return False
        if self.require_motion_cache and not self._has_motion_cache_for_indexes(record, indexes):
            return False
        return True

    def _record_by_episode_id(self, records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        return {str(record["episode_id"]): record for record in records}

    def _build_or_load_fixed_clips(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        manifest_path = Path(self.fixed_clip_manifest_path) if self.fixed_clip_manifest_path else None
        by_episode = self._record_by_episode_id(records)
        if manifest_path is not None and manifest_path.exists():
            payload = _load_json(manifest_path, {})
            manifest_start_mode = payload.get("fixed_clip_start_mode", "random") if isinstance(payload, dict) else "random"
            if manifest_start_mode != self.fixed_clip_start_mode:
                print(
                    f"[TactileCacheDataset] ignore fixed clip manifest with start_mode={manifest_start_mode!r}; "
                    f"requested {self.fixed_clip_start_mode!r}: {manifest_path}"
                )
            else:
                clips = payload.get("clips", payload if isinstance(payload, list) else [])
                if self.fixed_clip_count > 0:
                    clips = clips[: self.fixed_clip_count]
                fixed = self._materialize_fixed_clips(clips, by_episode)
                print(
                    f"[TactileCacheDataset] loaded {len(fixed)} usable fixed clips "
                    f"from {manifest_path} ({len(clips) - len(fixed)} skipped)"
                )
                if fixed:
                    return fixed
                print(f"[TactileCacheDataset] warning: {manifest_path} has no usable clips; regenerating")

        candidates = []
        for record in records:
            num_frames = int(record.get("sample_num_frames", 0))
            strides = range(1, self.clip_max_stride + 1) if self.clip_include_all_strides else (self.clip_max_stride,)
            for stride in strides:
                max_span = (self.clip_num_frames - 1) * int(stride) + 1
                max_start = num_frames - max_span
                if max_start < 0:
                    continue
                start_values = [0] if self.fixed_clip_start_mode == "episode_start" else range(0, max_start + 1, self.fixed_clip_candidate_stride)
                for start in start_values:
                    if not self._clip_has_required_cache(record, int(start), int(stride), self.clip_num_frames):
                        continue
                    candidates.append(
                        {
                            "episode_id": record["episode_id"],
                            "start": int(start),
                            "stride": int(stride),
                            "num_frames": int(self.clip_num_frames),
                        }
                    )
                if (
                    self.fixed_clip_start_mode != "episode_start"
                    and max_start >= 0
                    and self._clip_has_required_cache(record, max_start, int(stride), self.clip_num_frames)
                    and (not candidates or candidates[-1]["start"] != max_start)
                ):
                    candidates.append(
                        {
                            "episode_id": record["episode_id"],
                            "start": int(max_start),
                            "stride": int(stride),
                            "num_frames": int(self.clip_num_frames),
                        }
                    )

        if not candidates:
            print("[TactileCacheDataset] fixed clip mode found no candidates")
            return []
        rng = np.random.default_rng(self.fixed_clip_seed)
        count = min(self.fixed_clip_count, len(candidates))
        choice = np.sort(rng.choice(len(candidates), size=count, replace=False))
        clips = [candidates[int(idx)] for idx in choice]
        if manifest_path is not None:
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "seed": self.fixed_clip_seed,
                "requested_count": self.fixed_clip_count,
                "actual_count": len(clips),
                "clip_num_frames": self.clip_num_frames,
                "clip_max_stride": self.clip_max_stride,
                "clip_include_all_strides": self.clip_include_all_strides,
                "candidate_stride": self.fixed_clip_candidate_stride,
                "fixed_clip_start_mode": self.fixed_clip_start_mode,
                "clips": clips,
            }
            tmp_path = manifest_path.with_suffix(manifest_path.suffix + f".tmp.{os.getpid()}")
            with tmp_path.open("w") as f:
                json.dump(payload, f, indent=2)
            os.replace(tmp_path, manifest_path)
            print(f"[TactileCacheDataset] wrote {len(clips)} fixed clips to {manifest_path}")
        fixed = self._materialize_fixed_clips(clips, by_episode)
        print(
            f"[TactileCacheDataset] fixed clip mode: {len(records)} episodes, "
            f"{len(candidates)} candidates, {len(fixed)} selected"
        )
        return fixed

    def _materialize_fixed_clips(
        self,
        clips: list[dict[str, Any]],
        by_episode: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        fixed = []
        for clip_index, clip in enumerate(clips):
            episode_id = str(clip.get("episode_id"))
            record = by_episode.get(episode_id)
            if record is None:
                continue
            stride = int(clip.get("stride", self.clip_max_stride))
            start = int(clip.get("start", 0))
            num_frames = int(self.clip_num_frames if self.fixed_clip_override_num_frames else clip.get("num_frames", self.clip_num_frames))
            max_start = int(record.get("sample_num_frames", 0)) - ((num_frames - 1) * stride + 1)
            if max_start < 0:
                continue
            clipped_start = max(0, min(start, max_start))
            clipped_stride = max(1, stride)
            if not self._clip_has_required_cache(record, clipped_start, clipped_stride, num_frames):
                continue
            item = copy.deepcopy(record)
            item["sample_clip_start"] = clipped_start
            item["sample_clip_stride"] = clipped_stride
            item["sample_clip_num_frames"] = num_frames
            item["sample_clip_id"] = (
                f"{episode_id}_fixed{clip_index:04d}_s{item['sample_clip_start']:06d}_k{item['sample_clip_stride']}"
            )
            fixed.append(item)
        return fixed

    def _expand_episode_records_to_clips(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        expanded = []
        for record in records:
            num_frames = int(record.get("sample_num_frames", 0))
            max_span = (self.clip_num_frames - 1) * self.clip_max_stride + 1
            max_start = num_frames - max_span
            if max_start < 0:
                continue
            starts = list(range(0, max_start + 1, self.clip_start_stride))
            if starts and starts[-1] != max_start:
                starts.append(max_start)
            strides = range(1, self.clip_max_stride + 1) if self.clip_include_all_strides else (self.clip_max_stride,)
            for start in starts:
                for stride in strides:
                    if not self._clip_has_required_cache(record, int(start), int(stride), self.clip_num_frames):
                        continue
                    item = copy.deepcopy(record)
                    item["sample_clip_start"] = int(start)
                    item["sample_clip_stride"] = int(stride)
                    item["sample_clip_num_frames"] = int(self.clip_num_frames)
                    item["sample_clip_id"] = f"{record.get('episode_id')}_clip_s{int(start):06d}_k{int(stride)}"
                    expanded.append(item)
        print(
            f"[TactileCacheDataset] expanded {len(records)} episodes to {len(expanded)} clips "
            f"(clip_num_frames={self.clip_num_frames}, clip_max_stride={self.clip_max_stride}, "
            f"clip_start_stride={self.clip_start_stride}, include_all_strides={self.clip_include_all_strides})"
        )
        return expanded

    def close(self) -> None:
        self.data_list = None
        super().close()

    def filter(self, mode: str, **kwargs: Any) -> None:
        self.open()
        if mode in ("none", None):
            return
        if mode == "min_frames":
            min_num_frames = int(kwargs.get("min_num_frames", 1))
            self.data_list = [item for item in self.data_list if item.get("sample_num_frames", 0) >= min_num_frames]
            self.data_size = len(self.data_list)
            return
        raise ValueError(f"Unsupported TactileCacheDataset filter mode: {mode}")

    def __len__(self) -> int:
        if self.data_list is None:
            self.open()
        return self.data_size or 0

    def _get_data(self, index: int) -> dict[str, Any]:
        if self.data_list is None:
            self.open()
        return copy.deepcopy(self.data_list[index])


@register_dataset
class UnifiedCacheDataset(BaseDataset):
    """Read unified PhysCanvas cache episodes.

    A unified cache task contains ``episodes/<episode_id>`` directories with
    RGB/depth/motion arrays, timestamps, qpos, meta, and optional four-side
    tactile arrays under ``tactile/``.
    """

    def __init__(
        self,
        cache_root: str | Sequence[str] | None = None,
        cache_roots: Sequence[str] | None = None,
        cameras: dict[str, str] | None = None,
        tactile_sides: Sequence[str] | str | None = None,
        target_hz: float = 10.0,
        strict_hz: bool = True,
        data_size: int | None = None,
        max_episodes: int | None = None,
        max_frames: int | None = None,
        expand_clips: bool = False,
        clip_num_frames: int = 33,
        clip_max_stride: int = 4,
        clip_start_stride: int = 32,
        clip_include_all_strides: bool = False,
        fixed_clip_count: int = 0,
        fixed_clip_seed: int = 2026,
        fixed_clip_manifest_path: str = "",
        fixed_clip_candidate_stride: int = 1,
        fixed_clip_override_num_frames: bool = False,
        fixed_clip_start_mode: str = "random",
        cache_manifest_path: str = "",
        use_cache_manifest: bool = True,
        write_cache_manifest: bool = True,
        episode_include_regex: str = "",
        episode_exclude_regex: str = "",
        require_tactile: bool = True,
        require_motion: bool = True,
        require_depth: bool = True,
        object_state_manifest_path: str = "",
        require_object_state_labels: bool = False,
        object_state_sampling_mode: str = "uniform",
        object_state_balanced_sample_count: int = 0,
        object_state_transition_fraction: float = 0.30,
        object_state_all_one_fraction: float = 0.30,
        object_state_all_one_min_real_frames: int = 65,
        object_state_transition_max_offset: int = 32,
        object_state_sampling_seed: int = 20260912,
        object_state_online_mofang: bool = False,
        object_state_online_rewards: bool = False,
        object_state_online_label_root: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.cache_roots = _normalise_roots(cache_root, cache_roots)
        if not self.cache_roots:
            raise ValueError("UnifiedCacheDataset requires cache_root or cache_roots")
        self.cameras = cameras or VIEW_TO_CAMERA
        if tactile_sides is None:
            self.tactile_sides = list(DEFAULT_TACTILE_SIDES)
        elif isinstance(tactile_sides, str):
            self.tactile_sides = [x.strip() for x in tactile_sides.split(",") if x.strip()]
        else:
            self.tactile_sides = [str(x) for x in tactile_sides]
        self.target_hz = float(target_hz)
        self.strict_hz = bool(strict_hz)
        self.data_size = data_size
        self.max_episodes = max_episodes
        self.max_frames = max_frames
        self.expand_clips = bool(expand_clips)
        self.clip_num_frames = int(clip_num_frames)
        self.clip_max_stride = max(1, int(clip_max_stride))
        self.clip_start_stride = max(1, int(clip_start_stride))
        self.clip_include_all_strides = bool(clip_include_all_strides)
        self.fixed_clip_count = max(0, int(fixed_clip_count))
        self.fixed_clip_seed = int(fixed_clip_seed)
        self.fixed_clip_manifest_path = fixed_clip_manifest_path
        self.fixed_clip_candidate_stride = max(1, int(fixed_clip_candidate_stride))
        self.fixed_clip_override_num_frames = bool(fixed_clip_override_num_frames)
        self.fixed_clip_start_mode = str(fixed_clip_start_mode or "random")
        if self.fixed_clip_start_mode not in ("random", "episode_start"):
            raise ValueError(f"Unsupported fixed_clip_start_mode={self.fixed_clip_start_mode!r}")
        self.cache_manifest_path = cache_manifest_path
        self.use_cache_manifest = bool(use_cache_manifest)
        self.write_cache_manifest = bool(write_cache_manifest)
        self.episode_include_regex = str(episode_include_regex or "")
        self.episode_exclude_regex = str(episode_exclude_regex or "")
        self._episode_include_pattern = re.compile(self.episode_include_regex) if self.episode_include_regex else None
        self._episode_exclude_pattern = re.compile(self.episode_exclude_regex) if self.episode_exclude_regex else None
        self.require_tactile = bool(require_tactile)
        self.require_motion = bool(require_motion)
        self.require_depth = bool(require_depth)
        self.object_state_manifest_path = str(object_state_manifest_path or "")
        self.require_object_state_labels = bool(require_object_state_labels)
        self.object_state_sampling_mode = str(object_state_sampling_mode or "uniform").lower()
        if self.object_state_sampling_mode not in ("uniform", "balanced_binary"):
            raise ValueError(
                "object_state_sampling_mode must be 'uniform' or 'balanced_binary', "
                f"got {self.object_state_sampling_mode!r}"
            )
        self.object_state_balanced_sample_count = max(0, int(object_state_balanced_sample_count))
        self.object_state_transition_fraction = float(object_state_transition_fraction)
        self.object_state_all_one_fraction = float(object_state_all_one_fraction)
        if min(self.object_state_transition_fraction, self.object_state_all_one_fraction) < 0:
            raise ValueError("object-state sampling fractions must be non-negative")
        if self.object_state_transition_fraction + self.object_state_all_one_fraction > 1.0:
            raise ValueError("transition_fraction + all_one_fraction must be <= 1")
        self.object_state_all_one_min_real_frames = max(1, int(object_state_all_one_min_real_frames))
        self.object_state_transition_max_offset = max(1, int(object_state_transition_max_offset))
        self.object_state_sampling_seed = int(object_state_sampling_seed)
        self.object_state_online_mofang = bool(object_state_online_mofang)
        self.object_state_online_rewards = bool(object_state_online_rewards)
        self.object_state_online_label_root = str(object_state_online_label_root)
        if self.object_state_online_rewards and not self.object_state_online_label_root:
            raise ValueError("object_state_online_label_root is required for online rewards")
        self.data_list: list[dict[str, Any]] | None = None
        self._array_len_cache: dict[str, int] = {}

    @classmethod
    def load(cls, data_or_config: str | dict) -> "UnifiedCacheDataset":
        from giga_datasets.datasets.dataset import load_config

        config = load_config(data_or_config)
        kwargs = {k: v for k, v in config.items() if not k.startswith("_")}
        kwargs.pop("config_path", None)
        kwargs.pop("data_path", None)
        return cls(config_path=config.get("config_path"), data_path=config.get("data_path"), **kwargs)

    def _task_dirs_from_root(self, root: Path) -> list[Path]:
        if (root / "episodes").is_dir():
            return [root]
        return sorted([p for p in root.iterdir() if p.is_dir() and (p / "episodes").is_dir()]) if root.exists() else []

    def _episode_dirs_from_root(self, root: Path) -> list[Path]:
        dirs: list[Path] = []
        for task_dir in self._task_dirs_from_root(root):
            dirs.extend(sorted([p for p in (task_dir / "episodes").iterdir() if p.is_dir()]))
        return dirs

    def _array_len(self, path: Path) -> int:
        key = str(path)
        if key not in self._array_len_cache:
            self._array_len_cache[key] = _npy_len(path) if path.exists() else 0
        return self._array_len_cache[key]

    def _source_hz_and_step(self, meta: dict[str, Any], episode_dir: Path) -> tuple[float, int]:
        source_hz = float(meta.get("target_fps") or meta.get("fps") or 10.0)
        if self.target_hz <= 0:
            return source_hz, 1
        ratio = source_hz / self.target_hz
        step = int(round(ratio))
        if abs(ratio - step) > 1e-3 or step <= 0:
            message = (
                f"Unified cache hz mismatch for {episode_dir}: source_hz={source_hz}, "
                f"target_hz={self.target_hz}. target_hz must divide source_hz."
            )
            if self.strict_hz:
                raise ValueError(message)
            print(f"[UnifiedCacheDataset] warning: {message}; using nearest frame_step={max(step, 1)}")
        if self.strict_hz and abs(source_hz - self.target_hz * step) > 1e-3:
            raise ValueError(f"Invalid target_hz={self.target_hz} for source_hz={source_hz} in {episode_dir}")
        return source_hz, max(step, 1)

    def _available_count(self, episode_dir: Path, camera_names: dict[str, str]) -> int:
        paths = []
        for camera_name in camera_names.values():
            paths.extend([
                episode_dir / f"{camera_name}_rgb.npy",
                episode_dir / f"{camera_name}_depth.npy",
                episode_dir / f"{camera_name}_motion.npy",
            ])
        paths.extend([episode_dir / "timestamps.npy", episode_dir / "qpos.npy"])
        counts = [self._array_len(path) for path in paths if path.exists()]
        if self.require_tactile:
            for side in self.tactile_sides:
                counts.append(self._array_len(episode_dir / "tactile" / f"{side}.difference.npy"))
                counts.append(self._array_len(episode_dir / "tactile" / f"{side}.pointcloud.npy"))
        return min(counts) if counts else 0

    def _has_required_files(self, episode_dir: Path, camera_names: dict[str, str]) -> bool:
        for camera_name in camera_names.values():
            if not (episode_dir / f"{camera_name}_rgb.npy").exists():
                return False
            if self.require_depth and not (episode_dir / f"{camera_name}_depth.npy").exists():
                return False
            if self.require_motion and not (episode_dir / f"{camera_name}_motion.npy").exists():
                return False
        if self.require_tactile:
            for side in self.tactile_sides:
                if not (episode_dir / "tactile" / f"{side}.difference.npy").exists():
                    return False
                if not (episode_dir / "tactile" / f"{side}.pointcloud.npy").exists():
                    return False
        return True

    @staticmethod
    def _clip_indexes(start: int, stride: int, num_frames: int) -> np.ndarray:
        return int(start) + int(stride) * np.arange(int(num_frames), dtype=np.int64)

    def _clip_has_required_cache(self, record: dict[str, Any], start: int, stride: int, num_frames: int) -> bool:
        indexes = self._clip_indexes(start, stride, num_frames)
        return indexes.size > 0 and int(indexes[-1]) < int(record.get("sample_num_frames", 0))

    def _regex_accepts_text(self, text: str) -> bool:
        if self._episode_include_pattern is not None and self._episode_include_pattern.search(text) is None:
            return False
        if self._episode_exclude_pattern is not None and self._episode_exclude_pattern.search(text) is not None:
            return False
        return True

    def _episode_dir_matches_regex(self, episode_dir: Path) -> bool:
        if self._episode_include_pattern is None and self._episode_exclude_pattern is None:
            return True
        return self._regex_accepts_text(f"{episode_dir.name} {episode_dir}")

    def _episode_filter_text(self, record: dict[str, Any]) -> str:
        parts = [
            str(record.get("episode_id", "")),
            str(record.get("episode_name", "")),
            str(record.get("sample_cache_root", "")),
            str(record.get("sample_cache_episode_dir", "")),
        ]
        return " ".join(parts)

    def _filter_episode_records(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self._episode_include_pattern is None and self._episode_exclude_pattern is None:
            return records
        filtered = [record for record in records if self._regex_accepts_text(self._episode_filter_text(record))]
        print(
            "[UnifiedCacheDataset] episode regex filter: "
            f"{len(records)} -> {len(filtered)} "
            f"include={self.episode_include_regex!r} exclude={self.episode_exclude_regex!r}"
        )
        return filtered

    def _manifest_config(self) -> dict[str, Any]:
        return {
            "cache_roots": self.cache_roots,
            "cameras": self.cameras,
            "tactile_sides": self.tactile_sides,
            "target_hz": self.target_hz,
            "strict_hz": self.strict_hz,
            "max_episodes": self.max_episodes,
            "max_frames": self.max_frames,
            "require_tactile": self.require_tactile,
            "require_motion": self.require_motion,
            "require_depth": self.require_depth,
        }

    @staticmethod
    def _jsonable_record(record: dict[str, Any]) -> dict[str, Any]:
        item = copy.deepcopy(record)
        timestamps = item.get("target_timestamps")
        if isinstance(timestamps, np.ndarray):
            item["target_timestamps"] = timestamps.tolist()
        elif timestamps is not None:
            item["target_timestamps"] = list(timestamps)
        return item

    def _load_cache_manifest(self) -> list[dict[str, Any]] | None:
        if not self.use_cache_manifest or not self.cache_manifest_path:
            return None
        manifest_path = Path(self.cache_manifest_path)
        payload = _load_json(manifest_path, None)
        if not isinstance(payload, dict):
            return None
        if payload.get("manifest_type") != "unified_cache_episodes_v1":
            return None
        if payload.get("config") != self._manifest_config():
            print(f"[UnifiedCacheDataset] ignore stale cache manifest: {manifest_path}")
            return None
        records = payload.get("records")
        if not isinstance(records, list):
            return None
        return [copy.deepcopy(record) for record in records]

    def _write_cache_manifest(self, records: list[dict[str, Any]]) -> None:
        if not self.write_cache_manifest or not self.cache_manifest_path:
            return
        manifest_path = Path(self.cache_manifest_path)
        payload = {
            "manifest_type": "unified_cache_episodes_v1",
            "config": self._manifest_config(),
            "records": [self._jsonable_record(record) for record in records],
        }
        tmp_path = manifest_path.with_name(f".{manifest_path.name}.tmp.{os.getpid()}")
        try:
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            with tmp_path.open("w") as f:
                json.dump(payload, f, indent=2)
            os.replace(tmp_path, manifest_path)
            print(f"[UnifiedCacheDataset] wrote cache manifest: {manifest_path}")
        except OSError as exc:
            print(f"[UnifiedCacheDataset] warning: failed to write cache manifest {manifest_path}: {exc}")
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass

    def _attach_object_state_labels(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not self.object_state_manifest_path:
            return records
        payload = _load_json(Path(self.object_state_manifest_path), None)
        if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
            raise ValueError(f"Invalid object-state manifest: {self.object_state_manifest_path}")
        annotations = {str(item.get("episode_id")): item for item in payload["records"]}
        attached = []
        for record in records:
            annotation = annotations.get(str(record.get("episode_id")))
            if annotation is None:
                if not self.require_object_state_labels:
                    attached.append(record)
                continue
            item = copy.deepcopy(record)
            item["object_state_label_path"] = annotation["object_state_label_path"]
            item["object_state_task"] = annotation["object_state_task"]
            item["sample_num_frames"] = min(
                int(item.get("sample_num_frames", 0)), int(annotation.get("sample_num_frames", 0))
            )
            item["sample_raw_num_frames"] = min(
                int(item.get("sample_raw_num_frames", 0)), int(annotation.get("sample_raw_num_frames", 0))
            )
            attached.append(item)
        print(f"[UnifiedCacheDataset] object-state labels: {len(records)} -> {len(attached)}")
        return attached

    def _load_object_state_manifest_records(self) -> list[dict[str, Any]] | None:
        if not self.object_state_manifest_path or not self.require_object_state_labels:
            return None
        payload = _load_json(Path(self.object_state_manifest_path), None)
        if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
            raise ValueError(f"Invalid object-state manifest: {self.object_state_manifest_path}")
        records = []
        for source in payload["records"]:
            item = copy.deepcopy(source)
            label_path = Path(str(item.get("object_state_label_path", "")))
            episode_dir = Path(str(item.get("sample_cache_episode_dir", "")))
            online_mofang = self.object_state_online_mofang and item.get("object_state_task") == "mofang"
            if (not label_path.is_file() and not online_mofang) or not episode_dir.is_dir():
                continue
            if not self._has_required_files(episode_dir, item.get("sample_camera_names", self.cameras)):
                continue
            records.append(item)
        print(
            f"[UnifiedCacheDataset] loaded {len(records)} audited object-state episodes "
            f"from {self.object_state_manifest_path}"
        )
        return records

    @staticmethod
    def _object_state_logical_labels(record: dict[str, Any]) -> np.ndarray:
        label_path = Path(str(record.get("object_state_label_path", "")))
        if not label_path.is_file():
            return np.empty(0, dtype=np.int64)
        with np.load(label_path, allow_pickle=False) as state_file:
            labels = np.asarray(state_file["labels"], dtype=np.int64).reshape(-1)
        frame_step = max(1, int(record.get("sample_frame_step", 1)))
        num_frames = max(0, int(record.get("sample_num_frames", 0)))
        raw_indexes = np.arange(num_frames, dtype=np.int64) * frame_step
        raw_indexes = raw_indexes[raw_indexes < labels.size]
        return labels[raw_indexes]

    @staticmethod
    def _sample_balanced_pool(
        pool: list[dict[str, Any]], count: int, rng: np.random.Generator
    ) -> list[dict[str, Any]]:
        if count <= 0:
            return []
        if not pool:
            raise RuntimeError("object-state balanced sampling requested an empty category")
        selected: list[dict[str, Any]] = []
        repeat_index = 0
        while len(selected) < count:
            for index in rng.permutation(len(pool)):
                if len(selected) >= count:
                    break
                item = copy.deepcopy(pool[int(index)])
                item["object_state_sampling_repeat_index"] = repeat_index
                selected.append(item)
            repeat_index += 1
        return selected

    def _build_balanced_object_state_records(
        self, records: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        if self.object_state_online_rewards:
            return records
        if self.object_state_sampling_mode != "balanced_binary":
            return records

        online_records = []
        if self.object_state_online_mofang:
            online_records = [
                record for record in records
                if record.get("object_state_task") == "mofang"
            ]
            records = [record for record in records if record.get("object_state_task") != "mofang"]

        pools: dict[str, list[dict[str, Any]]] = {
            "all_zero": [],
            "transition": [],
            "all_one": [],
        }
        invalid = 0
        for record in records:
            if int(record.get("sample_num_frames", 0)) < self.clip_num_frames:
                invalid += 1
                continue
            labels = self._object_state_logical_labels(record)
            if labels.size < self.clip_num_frames or not np.isin(labels, (0, 1)).all():
                invalid += 1
                continue
            positive = np.flatnonzero(labels == 1)
            if positive.size == 0:
                pools["all_zero"].append(record)
                continue
            first_one = int(positive[0])
            if first_one > 0 and np.any(labels[:first_one] == 0):
                pools["transition"].append(record)
            trailing_ones = 0
            if labels[-1] == 1:
                trailing_ones = int(labels.size - np.flatnonzero(labels == 0)[-1] - 1) if np.any(labels == 0) else int(labels.size)
            if trailing_ones >= self.object_state_all_one_min_real_frames:
                pools["all_one"].append(record)

        sample_count = min(self.object_state_balanced_sample_count, len(records)) if self.object_state_balanced_sample_count else len(records)
        transition_count = int(round(sample_count * self.object_state_transition_fraction))
        all_one_count = int(round(sample_count * self.object_state_all_one_fraction))
        all_zero_count = sample_count - transition_count - all_one_count
        category_counts = {
            "all_zero": all_zero_count,
            "transition": transition_count,
            "all_one": all_one_count,
        }
        rng = np.random.default_rng(self.object_state_sampling_seed)
        balanced: list[dict[str, Any]] = []
        for category in ("all_zero", "transition", "all_one"):
            selected = self._sample_balanced_pool(pools[category], category_counts[category], rng)
            for item in selected:
                item["object_state_sampling_category"] = category
                item["object_state_sampling_allow_terminal_padding"] = category != "all_zero"
                item["object_state_all_one_min_real_frames"] = self.object_state_all_one_min_real_frames
                item["object_state_transition_max_offset"] = self.object_state_transition_max_offset
            balanced.extend(selected)
        for record in online_records:
            if int(record.get("sample_num_frames", 0)) < self.clip_num_frames:
                invalid += 1
                continue
            item = copy.deepcopy(record)
            item["object_state_sampling_category"] = "uniform"
            balanced.append(item)
        if balanced:
            permutation = rng.permutation(len(balanced))
            balanced = [balanced[int(index)] for index in permutation]

        positive_count = transition_count + all_one_count
        print(
            "[UnifiedCacheDataset] balanced object-state sampling: "
            f"samples={len(balanced)} all_zero={all_zero_count} "
            f"transition={transition_count} all_one={all_one_count} "
            f"contains_state1={positive_count} ({positive_count / max(1, sample_count):.1%}); "
            f"eligible_unique=all_zero:{len(pools['all_zero'])},"
            f"transition:{len(pools['transition'])},all_one:{len(pools['all_one'])}; "
            f"invalid={invalid} seed={self.object_state_sampling_seed}"
        )
        return balanced

    def open(self) -> None:
        if self.data_list is not None:
            return
        records = self._load_object_state_manifest_records()
        loaded_object_state_manifest = records is not None
        if records is None:
            records = self._load_cache_manifest()
        if records is not None:
            if not loaded_object_state_manifest:
                print(f"[UnifiedCacheDataset] loaded episode manifest: {self.cache_manifest_path}")
        else:
            records = []
            for root_str in self.cache_roots:
                root = Path(root_str)
                for episode_dir in self._episode_dirs_from_root(root):
                    if not self._episode_dir_matches_regex(episode_dir):
                        continue
                    meta = _load_json(episode_dir / "meta.json", {})
                    camera_names = {k: v for k, v in self.cameras.items() if (episode_dir / f"{v}_rgb.npy").exists()}
                    if set(("high", "left_wrist", "right_wrist")) - set(camera_names):
                        print(f"[UnifiedCacheDataset] skip {episode_dir}: missing one or more RGB cache views")
                        continue
                    if not self._has_required_files(episode_dir, camera_names):
                        print(f"[UnifiedCacheDataset] skip {episode_dir}: missing required unified cache files")
                        continue
                    source_hz, frame_step = self._source_hz_and_step(meta, episode_dir)
                    raw_frames = int(meta.get("num_frames") or self._available_count(episode_dir, camera_names))
                    raw_frames = min(raw_frames, self._available_count(episode_dir, camera_names))
                    if self.max_frames is not None and self.max_frames > 0:
                        raw_frames = min(raw_frames, int(self.max_frames))
                    num_frames = int((raw_frames + frame_step - 1) // frame_step)
                    if num_frames <= 0:
                        continue
                    episode_id = str(meta.get("episode_id") or episode_dir.name)
                    record = {
                        "data_index": len(records),
                        "episode_id": episode_id,
                        "episode_name": episode_id,
                        "sample_num_frames": num_frames,
                        "sample_raw_num_frames": raw_frames,
                        "sample_frame_step": frame_step,
                        "sample_source_hz": source_hz,
                        "sample_target_hz": self.target_hz,
                        "sample_cache_source": "unified_cache",
                        "sample_cache_root": str(root),
                        "sample_cache_episode_dir": str(episode_dir),
                        "sample_cache_meta": meta,
                        "sample_camera_names": camera_names,
                        "sample_tactile_sides": self.tactile_sides,
                        "target_timestamps": _load_npy(episode_dir / "timestamps.npy")[:raw_frames:frame_step],
                        "bucket_index": 0,
                        "video_info": tuple(meta.get("output_hw", [224, 224])),
                    }
                    records.append(record)
                    if self.max_episodes is not None and len(records) >= int(self.max_episodes):
                        break
                if self.max_episodes is not None and len(records) >= int(self.max_episodes):
                    break
            self._write_cache_manifest(records)
        if self.object_state_online_rewards:
            root = Path(self.object_state_online_label_root)
            task_by_cache_root = {
                "bottle_cap_rollout": "bottle",
                "shiguan_saizi_rollout": "shiguan",
                "jimu_rollout": "jimu",
                "mofang_v2_rollout": "mofang",
            }
            for record in records:
                cache_root = Path(str(record["sample_cache_root"])).name
                task = task_by_cache_root.get(cache_root)
                if task is None:
                    raise ValueError(f"Unknown online Reward task root: {cache_root}")
                record["object_state_task"] = task
                record["object_state_label_path"] = str(root / task / (record["episode_id"] + ".npz"))
        elif not loaded_object_state_manifest:
            records = self._attach_object_state_labels(records)
        records = self._filter_episode_records(records)
        records = self._build_balanced_object_state_records(records)
        if self.fixed_clip_count > 0:
            records = self._build_or_load_fixed_clips(records)
        elif self.expand_clips:
            records = self._expand_episode_records_to_clips(records)
        self.data_list = records
        self.data_size = len(records)
        print(f"[UnifiedCacheDataset] loaded {self.data_size} samples from {self.cache_roots}")

    def _candidate_fixed_clips(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        for record in records:
            max_start = int(record.get("sample_num_frames", 0)) - ((self.clip_num_frames - 1) * self.clip_max_stride + 1)
            if max_start < 0:
                continue
            start_values = [0] if self.fixed_clip_start_mode == "episode_start" else range(0, max_start + 1, self.fixed_clip_candidate_stride)
            for start in start_values:
                strides = range(1, self.clip_max_stride + 1) if self.clip_include_all_strides else (self.clip_max_stride,)
                for stride in strides:
                    if self._clip_has_required_cache(record, int(start), int(stride), self.clip_num_frames):
                        candidates.append(
                            {
                                "episode_id": record["episode_id"],
                                "start": int(start),
                                "stride": int(stride),
                                "num_frames": int(self.clip_num_frames),
                            }
                        )
        return candidates

    def _materialize_fixed_clips(self, clips: list[dict[str, Any]], by_episode: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        fixed: list[dict[str, Any]] = []
        for clip_index, clip in enumerate(clips):
            episode_id = str(clip.get("episode_id"))
            if episode_id not in by_episode:
                continue
            record = by_episode[episode_id]
            start = int(clip.get("start", 0))
            stride = int(clip.get("stride", self.clip_max_stride))
            num_frames = int(self.clip_num_frames if self.fixed_clip_override_num_frames else clip.get("num_frames", self.clip_num_frames))
            if not self._clip_has_required_cache(record, start, stride, num_frames):
                continue
            item = copy.deepcopy(record)
            item["sample_clip_start"] = start
            item["sample_clip_stride"] = stride
            item["sample_clip_num_frames"] = num_frames
            item["sample_clip_id"] = f"{episode_id}_unified{clip_index:04d}_s{start:06d}_k{stride}"
            fixed.append(item)
        return fixed

    def _build_or_load_fixed_clips(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        by_episode = {record["episode_id"]: record for record in records}
        manifest_path = Path(self.fixed_clip_manifest_path) if self.fixed_clip_manifest_path else None
        if manifest_path and manifest_path.exists():
            payload = _load_json(manifest_path, {})
            manifest_start_mode = payload.get("fixed_clip_start_mode", "random") if isinstance(payload, dict) else "random"
            if manifest_start_mode != self.fixed_clip_start_mode:
                print(
                    f"[UnifiedCacheDataset] ignore fixed clip manifest with start_mode={manifest_start_mode!r}; "
                    f"requested {self.fixed_clip_start_mode!r}: {manifest_path}"
                )
            else:
                clips = payload.get("clips", payload if isinstance(payload, list) else [])
                if self.fixed_clip_count > 0:
                    clips = clips[: self.fixed_clip_count]
                fixed = self._materialize_fixed_clips(clips, by_episode)
                print(
                    f"[UnifiedCacheDataset] loaded {len(fixed)} usable fixed clips "
                    f"from {manifest_path} ({len(clips) - len(fixed)} skipped)"
                )
                if fixed:
                    return fixed
                print(f"[UnifiedCacheDataset] warning: {manifest_path} has no usable clips; regenerating")
        candidates = self._candidate_fixed_clips(records)
        rng = np.random.default_rng(self.fixed_clip_seed)
        if candidates:
            order = rng.permutation(len(candidates))
            candidates = [candidates[int(i)] for i in order[: self.fixed_clip_count]]
        fixed = self._materialize_fixed_clips(candidates, by_episode)
        if manifest_path:
            payload = {
                "cache_roots": self.cache_roots,
                "target_hz": self.target_hz,
                "source": "unified_cache",
                "fixed_clip_start_mode": self.fixed_clip_start_mode,
                "clips": candidates,
            }
            tmp_path = manifest_path.with_name(f".{manifest_path.name}.tmp.{os.getpid()}")
            try:
                manifest_path.parent.mkdir(parents=True, exist_ok=True)
                with tmp_path.open("w") as f:
                    json.dump(payload, f, indent=2)
                os.replace(tmp_path, manifest_path)
            except OSError as exc:
                print(f"[UnifiedCacheDataset] warning: failed to write fixed clip manifest {manifest_path}: {exc}")
                try:
                    tmp_path.unlink(missing_ok=True)
                except OSError:
                    pass
        return fixed

    def _expand_episode_records_to_clips(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        expanded: list[dict[str, Any]] = []
        for record in records:
            max_start = int(record.get("sample_num_frames", 0)) - ((self.clip_num_frames - 1) * self.clip_max_stride + 1)
            if max_start < 0:
                continue
            starts = list(range(0, max_start + 1, self.clip_start_stride))
            if starts and starts[-1] != max_start:
                starts.append(max_start)
            strides = range(1, self.clip_max_stride + 1) if self.clip_include_all_strides else (self.clip_max_stride,)
            for start in starts:
                for stride in strides:
                    if not self._clip_has_required_cache(record, int(start), int(stride), self.clip_num_frames):
                        continue
                    item = copy.deepcopy(record)
                    item["sample_clip_start"] = int(start)
                    item["sample_clip_stride"] = int(stride)
                    item["sample_clip_num_frames"] = int(self.clip_num_frames)
                    item["sample_clip_id"] = f"{record.get('episode_id')}_unified_s{int(start):06d}_k{int(stride)}"
                    expanded.append(item)
        print(f"[UnifiedCacheDataset] expanded {len(records)} episodes to {len(expanded)} clips")
        return expanded

    def filter(self, mode: str, **kwargs: Any) -> None:
        self.open()
        if mode in ("none", None):
            return
        if mode == "min_frames":
            min_num_frames = int(kwargs.get("min_num_frames", 1))
            self.data_list = [item for item in self.data_list if item.get("sample_num_frames", 0) >= min_num_frames]
            self.data_size = len(self.data_list)
            return
        raise ValueError(f"Unsupported UnifiedCacheDataset filter mode: {mode}")

    def __len__(self) -> int:
        if self.data_list is None:
            self.open()
        return self.data_size or 0

    def _get_data(self, index: int) -> dict[str, Any]:
        if self.data_list is None:
            self.open()
        return copy.deepcopy(self.data_list[index])
