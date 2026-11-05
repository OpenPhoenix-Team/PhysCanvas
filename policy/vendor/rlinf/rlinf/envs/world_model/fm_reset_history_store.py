"""Read-only lookup of real FM reset histories by WM episode identity."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from rlinf.envs.world_model.fm_wm_history import validate_fm_reset_history


def install_fm_reset_stream_cursor(engine: Any) -> None:
    """Keep FM loader progress separate from candidate-restored sample indices.

    Install after the engine's source-routed reset wrapper. EngineState.restore
    must retain each candidate's sample_index for rollout/audit identity, but
    the next reset needs the counter associated with the advancing DataLoader.
    This guard is installed only by backends with an FM reset-history manifest.
    """
    if hasattr(engine, "_fm_reset_stream_cursor"):
        return
    engine._fm_reset_stream_cursor = int(engine.sample_index)
    original_reset = engine.reset

    def reset_with_fm_stream(data: dict[str, Any] | None = None) -> dict[str, Any]:
        engine.sample_index = engine._fm_reset_stream_cursor
        observation = original_reset(data)
        engine._fm_reset_stream_cursor = int(engine.sample_index)
        return observation

    engine.reset = reset_with_fm_stream


class FmResetHistoryStore:
    """Attach complete, aligned FM histories without changing reset source data."""

    def __init__(self, manifest: str | Path, expected_count: int = 128) -> None:
        self.path = Path(manifest)
        payload = json.loads(self.path.read_text())
        if payload.get("schema") != "fm_reset_history/v2" or not payload.get("complete"):
            raise ValueError("FM reset histories require a complete audited v2 manifest")
        entries = payload["samples"]
        if len(entries) != expected_count:
            raise ValueError(f"FM history requires {expected_count} entries; found {len(entries)}")
        if payload.get("history_hz") != 30 or payload.get("history_steps") != 8:
            raise ValueError("FM reset history must be 8 frames at 30 Hz")
        self.histories: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        aliases: dict[str, list[str]] = {}
        for entry in entries:
            episode_id = str(entry["episode_id"])
            if episode_id in self.histories:
                raise ValueError(f"duplicate FM history episode: {episode_id}")
            filename = Path(entry["initial_window_file"])
            if filename.is_absolute() or ".." in filename.parts:
                raise ValueError("FM history paths must remain within the manifest directory")
            with np.load(self.path.parent / filename, allow_pickle=False) as arrays:
                state = arrays["state"].astype(np.float32, copy=True)
                tactile = arrays["tactile"].astype(np.float32, copy=True)
                timestamps = arrays["timestamps"].astype(np.float64, copy=True)
                state_times = arrays["state_source_timestamps"].astype(np.float64, copy=True)
                tactile_times = arrays["tactile_source_timestamps"].astype(np.float64, copy=True)
            if timestamps.shape != (8,) or not np.isfinite(timestamps).all():
                raise ValueError(f"Invalid FM timeline: {episode_id}")
            if not np.allclose(np.diff(timestamps), 1000. / 30., rtol=0, atol=0.001):
                raise ValueError(f"FM timeline is not a uniform 30 Hz history: {episode_id}")
            for source_times, shape, limit in (
                (state_times, (8, 2), 20.), (tactile_times, (8, 4), 65.)
            ):
                if source_times.shape != shape or not np.isfinite(source_times).all():
                    raise ValueError(f"Invalid FM source timestamps: {episode_id}")
                if np.any(np.diff(source_times, axis=0) < 0):
                    raise ValueError(f"Unordered FM source timestamps: {episode_id}")
                if np.max(np.abs(source_times - timestamps[:, None])) > limit:
                    raise ValueError(f"FM source alignment error exceeds limit: {episode_id}")
            validate_fm_reset_history(state, tactile)
            if not np.isfinite(state).all() or not np.isfinite(tactile).all():
                raise ValueError(f"nonfinite FM history: {episode_id}")
            state.setflags(write=False)
            tactile.setflags(write=False)
            self.histories[episode_id] = (state, tactile)
            aliases.setdefault(episode_id.split("__source_", 1)[0], []).append(episode_id)
        # Historical source suffixes may be absent in the WM renderer identity.
        # Only unique aliases are usable; ambiguous aliases fail closed.
        self.aliases = {key: values[0] for key, values in aliases.items() if len(values) == 1}

    def attach(self, observation: dict[str, Any], episode_id: str) -> dict[str, Any]:
        """Return a payload with histories matching the exact reset episode/state."""
        key = episode_id
        if key not in self.histories:
            key = self.aliases.get(episode_id.split("__source_", 1)[0], "")
        if key not in self.histories:
            raise KeyError(f"no unambiguous FM reset history for {episode_id}")
        state, tactile = self.histories[key]
        reset_state = np.asarray(observation["observation.state"], dtype=np.float32)
        if reset_state.shape != (14,) or not np.allclose(state[-1], reset_state, atol=1e-5, rtol=0):
            raise ValueError(f"FM history endpoint differs from WM reset state: {episode_id}")
        return {
            **observation,
            "fm.state_history_30hz": state.copy(),
            "fm.tactile_history_30hz": tactile.copy(),
        }
