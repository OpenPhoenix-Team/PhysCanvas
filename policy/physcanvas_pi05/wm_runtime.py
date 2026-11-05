"""Frozen, stateful adapter to the *published* PhysCanvas inference functions.

No target/future images are accepted here. The public implementation remains
unmodified in vendor/PhysCanvas; latent packing follows its model_eval.py.
"""

from __future__ import annotations

import gc
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


PUBLIC_COMMIT = "c49ab996738753fb1fce68faabd82fd7ee46d212"
MANIFEST_SHA256 = "72cd121e50d40c874df7716e6086d06c98af6c956754e1d3193dd8b00aef70ea"
WEIGHTS_SHA256 = "7fd2e1b620d2b75e1ca50cfb3949055fe5204d451dbdc1ab1d950ba4e11c7259"
WEIGHTS_BYTES = 10_002_231_473
MODEL_CONFIG_SHA256 = "fc00b8544ef9f4ccde2de02a383422ac7ac4fd431d7b4504fdcaefe750ff1b83"


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_public_contract(cfg, *, hash_weights: bool = True) -> dict:
    """Fail closed on a historical model, incompatible canvas, or corrupt file."""
    source = Path(str(cfg["source_root"])).resolve()
    checkpoint = Path(str(cfg["checkpoint"])).resolve()
    config_path = Path(str(cfg["public_config"])).resolve()
    public = json.loads(config_path.read_text())
    reference = json.loads((source / "evaluation/data/public_eval_config.json").read_text())
    if public != reference:
        raise ValueError("public_config must match the pinned PhysCanvas release")
    training = public["training_config"]
    transform = training["dataloaders"]["test"]["transform"]
    required = {
        "sub_frames": 8, "num_views": 3, "dual_reference_mode": "dual_memory",
        "object_state_canvas_encoding": "lowfreq_bands_v1",
        "object_state_canvas_height": 32, "object_state_canvas_header_height": 8,
        "object_state_visual_canvas_mode": "full",
        "unified_wan_include_depth": False,
        "unified_wan_append_tactile_diff": True,
        "unified_wan_append_tactile_flow": True,
        "unified_wan_tactile_flow_encoding": "xyz_linear",
        "unified_wan_tactile_flow_linear_max_disp": 0.06,
    }
    for key, expected in required.items():
        if transform.get(key) != expected:
            raise ValueError(f"Unsupported public WM contract {key}={transform.get(key)!r}")
    if training["models"].get("flow_shift") != 5.0:
        raise ValueError("The published scheduler requires shift=5")
    for key, expected in (("tactile_flow_encoding", "xyz_linear"),
                          ("tactile_third_channel", "xyz"),
                          ("tactile_flow_linear_max_disp", .06)):
        if cfg.get(key, expected) != expected:
            raise ValueError(f"{key} must be {expected!r} for this checkpoint")
    if int(cfg.get("num_inference_steps", 30)) <= 0:
        raise ValueError("num_inference_steps must be positive")
    for name, expected in (("manifest.json", MANIFEST_SHA256),
                           ("config.json", MODEL_CONFIG_SHA256)):
        if sha256(checkpoint / name) != expected:
            raise ValueError(f"Not the published epoch30_step300 {name}")
    weight = checkpoint / "diffusion_pytorch_model.bin"
    if weight.stat().st_size != WEIGHTS_BYTES:
        raise ValueError("Published transformer size does not match")
    if hash_weights and sha256(weight) != WEIGHTS_SHA256:
        raise ValueError("Published transformer SHA256 does not match")
    for key in ("prompt_embedding", "pretrained_vae"):
        if not Path(str(cfg[key])).exists():
            raise FileNotFoundError(f"Missing {key}: {cfg[key]}")
    return {"checkpoint": str(checkpoint), "weights_sha256": WEIGHTS_SHA256,
            "hash_verified": hash_weights, "public_commit": PUBLIC_COMMIT,
            "visual_shape": [672, 672], "state_shape": [32, 672],
            "flow_encoding": "xyz_linear", "flow_max_disp": .06}


def configure_public_imports(cfg) -> None:
    source = str(Path(str(cfg["source_root"])).resolve())
    os.environ["PHYSCANVAS_EVAL_RUNTIME_SOURCE"] = source
    # This is the actual variable used by the public release's model_config.
    os.environ["CVPR2026_WM_PROMPT_EMBEDDING"] = str(cfg["prompt_embedding"])
    if source not in sys.path:
        sys.path.insert(0, source)


class PublishedWmRuntime:
    """One frozen transformer/VAE shared serially by independent WM sessions."""

    def __init__(self, cfg, device: str):
        self.cfg = cfg
        self.device = torch.device(device)
        self.dtype = torch.bfloat16
        self.loaded = False
        self.resident = False
        configure_public_imports(cfg)

    def load(self) -> None:
        if self.loaded:
            self.onload()
            return
        self.identity = validate_public_contract(self.cfg)
        self.api = importlib.import_module("evaluation.model_eval")
        from diffusers import AutoencoderKLWan, FlowMatchEulerDiscreteScheduler
        # Load on CPU first: avoid a transient FP32 5B model on a 36GB GPU.
        self.transformer = self.api.load_transformer(
            self.cfg["checkpoint"], "cpu", self.dtype
        ).requires_grad_(False)
        self.vae = AutoencoderKLWan.from_pretrained(
            self.cfg["pretrained_vae"], subfolder="vae", torch_dtype=self.dtype
        ).eval().requires_grad_(False)
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            self.cfg["pretrained_vae"], subfolder="scheduler"
        )
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
            scheduler.config, shift=5.0
        )
        if (int(self.vae.config.scale_factor_temporal), int(self.vae.config.z_dim)) != (4, 48):
            raise ValueError("The published WM requires the Wan 48-channel/4x temporal VAE")
        self.mean = torch.tensor(self.vae.config.latents_mean).view(1, 48, 1, 1, 1)
        self.scale = 1 / torch.tensor(self.vae.config.latents_std).view(1, 48, 1, 1, 1)
        prompt = torch.load(self.cfg["prompt_embedding"], map_location="cpu", weights_only=True)
        if isinstance(prompt, dict):
            prompt = prompt["prompt_embeds"]
        if not torch.is_tensor(prompt):
            raise TypeError("WM prompt embedding must be a tensor, not policy prompt text")
        self.prompt = prompt.unsqueeze(0) if prompt.ndim == 2 else prompt
        if self.prompt.ndim != 3 or self.prompt.shape[0] != 1:
            raise ValueError(f"Unexpected WM prompt embedding {self.prompt.shape}")
        self.loaded = True
        self.onload()

    def onload(self) -> None:
        if not self.loaded or self.resident:
            return
        self.transformer.to(self.device)
        self.vae.to(self.device)
        self.mean = self.mean.to(device=self.device, dtype=self.dtype)
        self.scale = self.scale.to(device=self.device, dtype=self.dtype)
        self.prompt = self.prompt.to(device=self.device, dtype=self.dtype)
        self.resident = True

    def offload(self) -> None:
        if not self.loaded or not self.resident:
            return
        self.transformer.to("cpu")
        self.vae.to("cpu")
        self.mean, self.scale, self.prompt = [x.cpu() for x in (self.mean, self.scale, self.prompt)]
        self.resident = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @torch.inference_mode()
    def predict(self, session, replay_visual: torch.Tensor):
        """Return 9 decoded frames; only the first local frame is conditioned."""
        self.load()
        api, vae = self.api, self.vae

        def encode(x):
            return api.forward_vae(vae, x.to(self.device), self.mean, self.scale)

        visual_ref = encode(session.visual)
        state_ref = encode(session.state_canvas)
        if visual_ref.shape[-2:] != (42, 42) or state_ref.shape[-2:] != (2, 42):
            raise ValueError("Unexpected separate-VAE canvas latent shapes")
        reference = torch.cat([visual_ref, state_ref], dim=-2)
        motion = replay_visual[:, :, :, :224, :]
        replay_state = F.interpolate(
            motion.flatten(0, 1), size=(32, 672), mode="bilinear", align_corners=False
        ).reshape(1, 9, 3, 32, 672)
        replay = torch.cat([encode(replay_visual), encode(replay_state)], dim=-2)
        shape_like = torch.empty((1, 48, 3, 44, 42), device=self.device, dtype=self.dtype)
        mask = torch.ones((1, 1, 3, 44, 42), device=self.device, dtype=self.dtype)
        mask[:, :, 0] = 0
        if session.anchor_latents is None:
            session.anchor_latents = encode(session.anchor_rgb).cpu()
        condition = {
            "ref_latents": reference, "replay_latents": replay,
            "first_frame_mask": mask, "prompt_embeds": self.prompt,
            "reference_fusion_mode": "dual_memory", "state_latent_height": 2,
            "global_reference_latents": session.anchor_latents.to(self.device),
            "global_reference_mask": torch.tensor([[float(session.frames > 0)]], device=self.device),
            "global_reference_role_ids": torch.zeros(1, device=self.device, dtype=torch.long),
            "global_reference_time_offsets": torch.tensor([[session.frames]], device=self.device, dtype=self.dtype),
            "clip_reference_role_id": torch.ones(1, device=self.device, dtype=torch.long),
        }
        generator = torch.Generator(device=self.device)
        if session.rng_state is None:
            generator.manual_seed(session.seed)
        else:
            generator.set_state(session.rng_state)
        latents = api.denoise_pipeline(
            self.transformer, shape_like, condition, self.scheduler,
            int(self.cfg.get("num_inference_steps", 30)), generator
        )
        session.rng_state = generator.get_state().cpu()
        visual = api.decode_vae(vae, latents[..., :42, :], self.mean, self.scale).float().cpu()
        state = api.decode_vae(vae, latents[..., 42:, :], self.mean, self.scale).float().cpu()
        if visual.shape != (1, 9, 3, 672, 672) or state.shape != (1, 9, 3, 32, 672):
            raise RuntimeError(f"Unexpected WM decoded shapes: {visual.shape}, {state.shape}")
        if not torch.isfinite(visual).all() or not torch.isfinite(state).all():
            raise FloatingPointError("WM generated nonfinite values")
        return visual, state
