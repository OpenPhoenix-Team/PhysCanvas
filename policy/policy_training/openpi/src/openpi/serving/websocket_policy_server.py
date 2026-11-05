import asyncio
import http
import logging
import time
import traceback

import numpy as np
from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames

logger = logging.getLogger(__name__)


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                obs = msgpack_numpy.unpackb(await websocket.recv())
                obs = _maybe_repack_acwm_observation(obs)

                infer_time = time.monotonic()
                action = self._policy.infer(obs)
                infer_time = time.monotonic() - infer_time

                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                if prev_total_time is not None:
                    # We can only record the last total time since we also want to include the send time.
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _maybe_repack_acwm_observation(obs):
    """Accept raw LeRobot/ACWM observations while preserving the existing API."""
    if not isinstance(obs, dict) or "images" in obs:
        return obs

    state = _first_present(
        obs,
        (
            "state",
            "observation.state",
            "observation/state",
            "qpos",
        ),
    )
    base_image = _first_present(
        obs,
        (
            "observation.images.base_0",
            "observation.images.base_0_rgb",
            "observation/images/base_0",
            "observation/images/base_0_rgb",
            "image.base_0_rgb",
            "image.base_0",
            "base_0",
            "base_0_rgb",
        ),
    )

    if state is None or base_image is None:
        return obs

    images = {"cam_high": _as_chw_uint8(base_image)}

    left_wrist = _first_present(
        obs,
        (
            "observation.images.left_wrist_0",
            "observation.images.left_wrist_0_rgb",
            "observation/images/left_wrist_0",
            "observation/images/left_wrist_0_rgb",
            "image.left_wrist_0_rgb",
            "image.left_wrist_0",
            "left_wrist_0",
            "left_wrist_0_rgb",
        ),
    )
    if left_wrist is not None:
        images["cam_left_wrist"] = _as_chw_uint8(left_wrist)

    right_wrist = _first_present(
        obs,
        (
            "observation.images.right_wrist_0",
            "observation.images.right_wrist_0_rgb",
            "observation/images/right_wrist_0",
            "observation/images/right_wrist_0_rgb",
            "image.right_wrist_0_rgb",
            "image.right_wrist_0",
            "right_wrist_0",
            "right_wrist_0_rgb",
        ),
    )
    if right_wrist is not None:
        images["cam_right_wrist"] = _as_chw_uint8(right_wrist)

    repacked = dict(obs)
    repacked["state"] = np.asarray(state, dtype=np.float32)
    repacked["images"] = images

    prompt = _first_present(obs, ("prompt", "task", "instruction", "language_instruction"))
    if prompt is not None:
        repacked["prompt"] = prompt

    return repacked


def _first_present(data: dict, keys: tuple[str, ...]):
    for key in keys:
        value = _get_value(data, key)
        if value is not None:
            return value
    return None


def _get_value(data: dict, key: str):
    if key in data:
        return data[key]

    current = data
    for part in key.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _as_chw_uint8(image):
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image, 0.0, 1.0) * 255.0
    if image.ndim == 3 and image.shape[-1] == 3 and image.shape[0] != 3:
        image = np.transpose(image, (2, 0, 1))
    return image.astype(np.uint8)


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None
