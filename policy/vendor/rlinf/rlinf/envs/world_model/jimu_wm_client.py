# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Synchronous client for the stateful PPU Jimu WM rollout service."""

from __future__ import annotations

import io
import threading
import uuid
from typing import Any

import torch

from rlinf.envs.world_model.jimu_wm_protocol import validate_action_batch


class TorchSerializer:
    @staticmethod
    def dumps(data: Any) -> bytes:
        buffer = io.BytesIO()
        torch.save(data, buffer)
        return buffer.getvalue()

    @staticmethod
    def loads(data: bytes) -> Any:
        return torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)


class JimuWmClient:
    def __init__(
        self,
        host: str,
        port: int,
        timeout_ms: int = 1_800_000,
        *,
        owner_id: str | None = None,
    ) -> None:
        import zmq

        self._zmq = zmq
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.setsockopt(zmq.RCVTIMEO, int(timeout_ms))
        self._socket.setsockopt(zmq.SNDTIMEO, int(timeout_ms))
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(f"tcp://{host}:{int(port)}")
        self._lock = threading.Lock()
        self.owner_id = owner_id or uuid.uuid4().hex

    def call(self, endpoint: str, data: dict[str, Any] | None = None) -> Any:
        request = TorchSerializer.dumps({"endpoint": endpoint, "data": data or {}})
        with self._lock:
            self._socket.send(request)
            reply = self._socket.recv()
        if reply == b"ERROR":
            raise RuntimeError(f"WM endpoint {endpoint!r} returned ERROR")
        payload = TorchSerializer.loads(reply)
        if isinstance(payload, dict) and payload.get("ok") is False:
            raise RuntimeError(payload.get("error", f"WM endpoint {endpoint!r} failed"))
        return payload

    def metadata(self) -> dict[str, Any]:
        return self.call("metadata")

    def reset_groups(
        self,
        *,
        num_groups: int,
        group_size: int,
        prompt: str,
    ) -> dict[str, Any]:
        return self.call(
            "reset_groups",
            {
                "owner_id": self.owner_id,
                "num_groups": num_groups,
                "group_size": group_size,
                "prompt": prompt,
            },
        )

    def rollout16(self, session_ids: list[str], actions) -> dict[str, Any]:
        action_batch = validate_action_batch(actions)
        if len(session_ids) != action_batch.shape[0]:
            raise ValueError(
                f"session count {len(session_ids)} != action batch {action_batch.shape[0]}"
            )
        return self.call(
            "rollout16",
            {
                "owner_id": self.owner_id,
                "session_ids": session_ids,
                "actions": action_batch,
            },
        )

    def close(self) -> None:
        try:
            self.call("release_owner", {"owner_id": self.owner_id})
        except Exception:
            # The server may already be unavailable during Ray worker teardown.
            pass
        self._socket.close(0)
        self._context.term()
