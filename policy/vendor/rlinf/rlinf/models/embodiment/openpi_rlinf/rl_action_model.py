# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import dataclasses
import random
from typing import Any, Literal

import torch

from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.models.embodiment.modules.value_head import ValueHead
from rlinf.models.embodiment.openpi_rlinf.eval_action_model import (
    OpenPiPytorchEvalActionModel,
)
from rlinf.models.embodiment.openpi_rlinf.pi0_model import model as pi0_model_module
from rlinf.models.embodiment.openpi_rlinf.pi0_model.model import Observation
from rlinf.models.embodiment.openpi_rlinf.pi0_model.pi0 import Pi0
from rlinf.models.embodiment.openpi_rlinf.utils import rl_sampler


@dataclasses.dataclass(frozen=True)
class OpenPiPytorchRLConfig:
    """Static RL knobs read from ``actor.model.openpi`` in YAML."""

    add_value_head: bool = False
    noise_method: str = "flow_ode"
    noise_level: float = 0.0
    joint_logprob: bool = False
    ignore_last: bool = False
    value_after_vlm: bool = False
    value_vlm_mode: str = "mean_token"
    detach_critic_input: bool = False
    train_expert_only: bool = False
    config_name: str = ""
    hold_gt_until_state_move: bool = False
    state_move_threshold: float = 0.001
    hold_gt_max_policy_chunks: int = 4

    def __post_init__(self) -> None:
        if self.state_move_threshold <= 0:
            raise ValueError("state_move_threshold must be positive")
        if self.hold_gt_max_policy_chunks <= 0:
            raise ValueError("hold_gt_max_policy_chunks must be positive")


def first_motion_action_indices(
    actions: torch.Tensor,
    states: torch.Tensor,
    *,
    threshold: float,
) -> torch.Tensor:
    """Return each trajectory's first moving action index, or ``-1``.

    Jimu actions are absolute 14-D targets. An action is moving when at least
    one dimension differs from the current state by more than ``threshold``.
    The returned index is gate/audit metadata only: the accepted 16-step chunk
    is executed unchanged.
    """

    if actions.ndim != 3:
        raise ValueError(f"actions must have shape [B,H,D], got {actions.shape}")
    if states.ndim != 2 or states.shape != (actions.shape[0], actions.shape[2]):
        raise ValueError(
            "states must have shape [B,D] matching actions, "
            f"got actions={actions.shape} states={states.shape}"
        )
    if threshold <= 0:
        raise ValueError(f"threshold must be positive, got {threshold}")

    moving = (actions - states[:, None, :]).abs().amax(dim=-1) > float(threshold)
    has_motion = moving.any(dim=1)
    first = moving.to(torch.int64).argmax(dim=1)
    return torch.where(has_motion, first, torch.full_like(first, -1))


def _gather_attempt_tree(
    attempts: list[Any], selected_attempts: torch.Tensor
) -> Any:
    """Select one batch row from each sampler attempt in a nested result."""

    first = attempts[0]
    if torch.is_tensor(first):
        if first.ndim == 0:
            return first
        batch_size = int(selected_attempts.numel())
        if first.shape[0] != batch_size:
            return first
        stacked = torch.stack(attempts, dim=0)
        batch_indices = torch.arange(batch_size, device=stacked.device)
        return stacked[selected_attempts.to(stacked.device), batch_indices]
    if isinstance(first, dict):
        return {
            key: _gather_attempt_tree(
                [attempt[key] for attempt in attempts], selected_attempts
            )
            for key in first
        }
    if first is None:
        return None
    return first


class OpenPiPytorchRLActionModel(OpenPiPytorchEvalActionModel):
    """Eval + PPO variant of :class:`OpenPiPytorchEvalActionModel`."""

    def __init__(
        self,
        pi0_model: Pi0,
        *,
        num_steps: int,
        action_chunk: int,
        action_env_dim: int,
        rl_cfg: OpenPiPytorchRLConfig,
        paligemma_width: int,
    ):
        super().__init__(
            pi0_model,
            num_steps=num_steps,
            action_env_dim=action_env_dim,
            action_chunk=action_chunk,
            config_name=rl_cfg.config_name,
        )
        # RL-only shape knobs for the SDE chain. ``action_chunk`` is already
        # stored on the base (used by :meth:`output_transform`); these two are
        # exclusive to PPO and size the chain tensor.
        self.action_horizon = pi0_model.action_horizon
        self.model_action_dim = pi0_model.action_dim
        self.rl_cfg = rl_cfg

        if rl_cfg.add_value_head:
            if not rl_cfg.value_after_vlm:
                raise NotImplementedError(
                    "OpenPiPytorchRLActionModel currently only supports the "
                    "value_after_vlm=True value head (BEHAVIOR pi05 config); "
                    "suffix-pooled (value_after_vlm=False) is not wired through."
                )
            # Same MLP shape the original PPO path uses for pi05 (1024, 512, 256).
            self.value_head = ValueHead(
                input_dim=paligemma_width,
                hidden_sizes=(1024, 512, 256),
                output_dim=1,
                activation="relu",
                bias_last=True,
            )
            # Match the wrapped Pi0's parameter dtype so the rollout side (no
            # FSDP, weights already cast to ``cfg.precision``) does the prefix
            # → value head pass in one dtype, and so the FSDP-wrapped actor's
            # MixedPrecisionPolicy(param_dtype=bf16) keeps the value head
            # consistent with the rest of the model.
            model_dtype = next(self.model.parameters()).dtype
            self.value_head.to(model_dtype)

    def set_global_step(self, global_step: int) -> None:
        """Noise-annealing hook — currently a no-op (constant noise_level)."""
        del global_step

    # ------------------------------------------------------------------ rollout

    @torch.no_grad()
    def predict_action_batch(
        self,
        env_obs: dict[str, Any],
        mode: Literal["train", "eval"] = "eval",
        compute_values: bool = False,
        *,
        noise: torch.Tensor | None = None,
        rng: torch.Generator | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Sample env actions, in eval (Euler ODE) or train (SDE chain) mode.

        ``mode="eval"`` delegates straight to the base
        :meth:`OpenPiPytorchActionModel.predict_action_batch` (deterministic
        Euler sampler + openpi output transform). ``mode="train"`` runs the
        PPO chain-collecting SDE sampler implemented in :meth:`_predict_train`.
        """
        del kwargs
        if mode not in ("train", "eval"):
            raise ValueError(f"Unknown predict mode: {mode!r}")

        gate_mask = self._initial_motion_gate_mask(env_obs)
        if gate_mask is None or not bool(gate_mask.any()):
            if mode == "eval":
                return super().predict_action_batch(
                    env_obs, mode=mode, noise=noise, rng=rng
                )
            self._ensure_wrappers()
            repacked = self._repack_env_obs(env_obs)
            processed = self.input_transform(repacked, transpose=False)
            observation = self._observation_dict_to_device(processed)
            return self._predict_train(
                observation, noise=noise, rng=rng, compute_values=compute_values
            )

        if noise is not None:
            raise ValueError(
                "Explicit noise is incompatible with the initial motion gate: "
                "the gate requires a fresh sample for each attempt."
            )

        self._ensure_wrappers()
        repacked = self._repack_env_obs(env_obs)
        processed = self.input_transform(repacked, transpose=False)
        observation = self._observation_dict_to_device(processed)
        raw_states = torch.as_tensor(
            env_obs.get("states"), device=self.device, dtype=torch.float32
        )
        return self._predict_with_initial_motion_gate(
            observation,
            gate_mask=gate_mask.to(self.device),
            raw_states=raw_states,
            mode=mode,
            rng=rng,
            compute_values=compute_values,
        )

    def _initial_motion_gate_mask(
        self, env_obs: dict[str, Any]
    ) -> torch.Tensor | None:
        """Identify reset observations that require GT-hold resampling."""

        if not self.rl_cfg.hold_gt_until_state_move:
            return None
        chunk_indices = env_obs.get("policy_chunk_indices")
        if chunk_indices is None:
            return None
        chunk_indices = torch.as_tensor(chunk_indices, dtype=torch.long)
        if chunk_indices.ndim != 1:
            raise ValueError(
                "policy_chunk_indices must have shape [B], "
                f"got {tuple(chunk_indices.shape)}"
            )
        return chunk_indices == 0

    def _predict_with_initial_motion_gate(
        self,
        observation: Observation,
        *,
        gate_mask: torch.Tensor,
        raw_states: torch.Tensor,
        mode: Literal["train", "eval"],
        rng: torch.Generator | None,
        compute_values: bool,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Hold reset GT while sampling at most four complete policy chunks.

        Each batch row accepts its first chunk containing movement. A row with
        no movement after the configured maximum accepts the final sample. No
        action is sliced or padded, so the executed action, diffusion chain and
        stored log-probability remain exactly aligned for GRPO.
        """

        batch_size = int(observation.state.shape[0])
        if gate_mask.shape != (batch_size,):
            raise ValueError(
                f"gate_mask must have shape ({batch_size},), got {gate_mask.shape}"
            )
        if raw_states.shape != (batch_size, self.action_env_dim):
            raise ValueError(
                "states must provide raw absolute targets for the motion gate: "
                f"expected {(batch_size, self.action_env_dim)}, got {raw_states.shape}"
            )
        resolved = ~gate_mask.bool()
        selected_attempts = torch.zeros(
            batch_size, dtype=torch.long, device=self.device
        )
        action_attempts: list[torch.Tensor] = []
        result_attempts: list[dict[str, Any]] = []

        for attempt_index in range(self.rl_cfg.hold_gt_max_policy_chunks):
            if mode == "train":
                actions, result = self._predict_train(
                    observation,
                    noise=None,
                    rng=rng,
                    compute_values=compute_values,
                )
            else:
                actions, result = self._predict_eval(
                    observation, noise=None, rng=rng
                )
            action_attempts.append(actions)
            result_attempts.append(result)

            motion_indices = first_motion_action_indices(
                actions,
                raw_states.to(device=actions.device, dtype=actions.dtype),
                threshold=self.rl_cfg.state_move_threshold,
            )
            newly_resolved = (~resolved) & (motion_indices >= 0)
            selected_attempts[newly_resolved] = attempt_index
            resolved |= newly_resolved
            if bool(resolved.all()):
                break
            if attempt_index == self.rl_cfg.hold_gt_max_policy_chunks - 1:
                selected_attempts[~resolved] = attempt_index
                resolved[:] = True

        selected_actions = _gather_attempt_tree(
            action_attempts, selected_attempts
        ).contiguous()
        selected_result = _gather_attempt_tree(result_attempts, selected_attempts)
        selected_motion_indices = first_motion_action_indices(
            selected_actions,
            raw_states.to(
                device=selected_actions.device, dtype=selected_actions.dtype
            ),
            threshold=self.rl_cfg.state_move_threshold,
        )
        selected_result.update(
            {
                "motion_gate_applied": gate_mask.bool(),
                "motion_gate_attempts": selected_attempts + 1,
                "motion_start_action_indices": selected_motion_indices,
                "motion_gate_forced": gate_mask.bool()
                & (selected_motion_indices < 0),
            }
        )
        return selected_actions, selected_result

    def _predict_train(
        self,
        observation: Observation,
        *,
        noise: torch.Tensor | None,
        rng: torch.Generator | None,
        compute_values: bool,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        del compute_values  # always True for PPO; value head presence is the real gate
        rl_cfg = self.rl_cfg
        # joint_logprob=True / NFT collection are not wired through — surface
        # the configuration mismatch loudly so a misconfigured run does not
        # silently produce zero-logprob trajectories.
        if rl_cfg.joint_logprob:
            raise NotImplementedError(
                "openpi_rlinf RL port supports joint_logprob=False only "
                "(single random denoise step). Set actor.model.openpi.joint_logprob=False."
            )
        if not rl_cfg.add_value_head:
            raise RuntimeError(
                "predict_action_batch(mode='train') requires actor.model.add_value_head=True "
                "and openpi.value_after_vlm=True; check the experiment YAML."
            )

        device = self.device
        # Mirror Pi0.sample_actions' invariant: every prefix / suffix pass sees
        # the preprocessed observation (image resize/pad + default masks).
        observation = pi0_model_module.preprocess_observation(observation, train=False)
        B = observation.state.shape[0]
        num_steps = self.num_steps

        if noise is None:
            noise = torch.randn(
                B,
                self.action_horizon,
                self.model_action_dim,
                device=device,
                dtype=torch.float32,
                generator=rng,
            )

        prefix_out, prefix_mask, kv_cache = self.model.build_prefix_cache(observation)

        # VLM-pooled value (BEHAVIOR pi05): one value per sample, reused as the
        # rollout ``prev_values`` independent of the chosen denoise index.
        vlm_value = rl_sampler.value_from_prefix(
            self.value_head, prefix_out, prefix_mask, mode=rl_cfg.value_vlm_mode
        )

        # Single stochastic denoise step picked uniformly; the remaining steps
        # are deterministic flow_ode. ``denoise_inds[:, 0]`` is the read site
        # in default_forward (and matches the original RL contract where every
        # column of denoise_inds carries the same chosen index).
        hi = num_steps - 2 if rl_cfg.ignore_last else num_steps - 1
        chosen = random.randint(0, hi)
        denoise_inds = torch.full(
            (B, num_steps), chosen, device=device, dtype=torch.long
        )
        idx_step = torch.empty(B, device=device, dtype=torch.long)
        timesteps = rl_sampler.get_timesteps(num_steps, device)

        chains = [noise]
        log_probs: list[torch.Tensor] = []
        x_t = noise

        for idx in range(num_steps):
            method = rl_cfg.noise_method if idx == chosen else "flow_ode"
            t_val = float(timesteps[idx].item())
            t_tensor = torch.full((B,), t_val, device=device, dtype=torch.float32)
            suffix_act = self.model.run_suffix(
                observation, x_t, t_tensor, kv_cache, prefix_mask
            )
            v_t = self.model.velocity_from_suffix(suffix_act).to(torch.float32)
            idx_step.fill_(idx)
            x_t_mean, x_t_std = rl_sampler.sample_mean_var(
                x_t.to(torch.float32),
                v_t,
                idx_step,
                noise_method=method,
                noise_level=rl_cfg.noise_level,
                num_steps=num_steps,
            )
            step_noise = torch.randn(
                x_t.shape, device=device, dtype=torch.float32, generator=rng
            )
            x_t = x_t_mean + step_noise * x_t_std
            log_probs.append(rl_sampler.gaussian_logprob(x_t, x_t_mean, x_t_std))
            chains.append(x_t)

        x_0 = x_t
        chains_tensor = torch.stack(chains, dim=1).contiguous()  # [B, N+1, H, A]
        log_probs_stacked = torch.stack(log_probs, dim=1)  # [B, N, H, A]
        log_probs_picked = log_probs_stacked[
            torch.arange(B, device=device), denoise_inds[:, 0]
        ][:, : self.action_chunk, : self.action_env_dim]
        log_probs_picked = log_probs_picked.float().contiguous()

        prev_values = vlm_value[:, None].float().contiguous()  # [B, 1]

        env_outputs = self.output_transform(
            {"actions": x_0, "state": observation.state}
        )
        # openpi Unnormalize runs in float64; cast env actions to float32 to match
        # the eval path + legacy contract (and the env/rollout action dtype).
        actions = env_outputs["actions"].to(device=device, dtype=torch.float32)

        forward_inputs: dict[str, torch.Tensor] = {
            "chains": chains_tensor,
            "denoise_inds": denoise_inds,
            "obs_state": observation.state.contiguous(),
            "tokenized_prompt": observation.tokenized_prompt.contiguous(),
            "tokenized_prompt_mask": observation.tokenized_prompt_mask.contiguous(),
            "action": actions.reshape(B, -1).contiguous(),
            "model_action": x_0.reshape(B, -1).contiguous(),
        }
        # Flat keys: the rollout layer's `_split_policy_output` uses a flat
        # torch.split over forward_inputs values, so nested image / mask dicts
        # are unsupported. Encode them with prefixed scalar keys here and decode
        # symmetrically in default_forward.
        for k, v in observation.images.items():
            forward_inputs[f"obs_image__{k}"] = v.contiguous()
        for k, v in observation.image_masks.items():
            forward_inputs[f"obs_image_mask__{k}"] = v.contiguous()
        if observation.tactile is not None:
            forward_inputs["obs_tactile"] = observation.tactile.contiguous()
        if observation.tactile_mask is not None:
            forward_inputs["obs_tactile_mask"] = observation.tactile_mask.contiguous()

        return actions, {
            "prev_logprobs": log_probs_picked,
            "prev_values": prev_values,
            "forward_inputs": forward_inputs,
        }

    # --------------------------------------------------------------- training

    def forward(self, forward_type: ForwardType = ForwardType.DEFAULT, **kwargs):
        """Dispatch — RL variant only supports :attr:`ForwardType.DEFAULT`."""
        if forward_type != ForwardType.DEFAULT:
            raise NotImplementedError(
                f"{type(self).__name__} only supports ForwardType.DEFAULT; "
                f"got forward_type={forward_type!r}. "
                "Use the SFT subclass (actor.model.openpi.task='sft') for SFT."
            )
        return self.default_forward(**kwargs)

    def default_forward(self, forward_inputs: dict[str, torch.Tensor], **kwargs):
        """PPO-time recompute of logprobs and values from the stored chain."""
        rl_cfg = self.rl_cfg
        if rl_cfg.joint_logprob:
            raise NotImplementedError(
                "openpi_rlinf RL port supports joint_logprob=False only."
            )

        compute_values = kwargs.get("compute_values", True)

        chains = forward_inputs["chains"]
        denoise_inds = forward_inputs["denoise_inds"]
        B = chains.shape[0]
        device = chains.device

        images: dict[str, torch.Tensor] = {}
        image_masks: dict[str, torch.Tensor] = {}
        for key, value in forward_inputs.items():
            if key.startswith("obs_image__"):
                images[key[len("obs_image__") :]] = value
            elif key.startswith("obs_image_mask__"):
                image_masks[key[len("obs_image_mask__") :]] = value
        observation = Observation(
            images=images,
            image_masks=image_masks,
            state=forward_inputs["obs_state"],
            tokenized_prompt=forward_inputs["tokenized_prompt"],
            tokenized_prompt_mask=forward_inputs["tokenized_prompt_mask"],
            tactile=forward_inputs.get("obs_tactile"),
            tactile_mask=forward_inputs.get("obs_tactile_mask"),
        )

        # Grad-enabled prefix pass — the VLM value head reads ``prefix_out``
        # so gradients must flow through paligemma here.
        if rl_cfg.train_expert_only:
            with torch.no_grad():
                prefix_out, prefix_mask, kv_cache = self.model.build_prefix_cache(
                    observation
                )
        else:
            prefix_out, prefix_mask, kv_cache = self.model.build_prefix_cache(
                observation
            )

        idx0 = denoise_inds[:, 0].to(torch.long)
        arange_B = torch.arange(B, device=device)
        chains_pre = chains[arange_B, idx0]  # x_t   at the chosen step
        chains_next = chains[arange_B, idx0 + 1]  # x_{t-dt} actually drawn at rollout

        timesteps = rl_sampler.get_timesteps(self.num_steps, device)
        t_input = timesteps[idx0].to(torch.float32)

        suffix_act = self.model.run_suffix(
            observation, chains_pre, t_input, kv_cache, prefix_mask
        )
        v_t = self.model.velocity_from_suffix(suffix_act).to(torch.float32)

        x_t_mean, x_t_std = rl_sampler.sample_mean_var(
            chains_pre.to(torch.float32),
            v_t,
            idx0,
            noise_method=rl_cfg.noise_method,
            noise_level=rl_cfg.noise_level,
            num_steps=self.num_steps,
        )
        log_probs = rl_sampler.gaussian_logprob(
            chains_next.to(torch.float32), x_t_mean, x_t_std
        )
        log_probs = (
            log_probs[:, : self.action_chunk, : self.action_env_dim]
            .float()
            .contiguous()
        )

        if compute_values and rl_cfg.add_value_head and rl_cfg.value_after_vlm:
            values = rl_sampler.value_from_prefix(
                self.value_head, prefix_out, prefix_mask, mode=rl_cfg.value_vlm_mode
            )
        else:
            values = torch.zeros(B, device=device, dtype=torch.float32)

        entropy = torch.zeros((B, 1), device=device, dtype=torch.float32)
        return {
            "logprobs": log_probs,
            "values": values.float(),
            "entropy": entropy,
        }

    def freeze_vlm(self) -> int:
        """Freeze the PaliGemma VLM (vision + expert-0 of the LLM) for PPO.
        Returns the count of newly-frozen parameter tensors (for logging).
        """
        frozen = 0
        # SigLIP vision encoder — entirely belongs to the PaliGemma side.
        for p in self.model.img.parameters():
            if p.requires_grad:
                p.requires_grad = False
                frozen += 1
        # The tactile encoder is another observation-prefix encoder. Keep it
        # frozen with the VLM when train_expert_only=True, matching the frozen
        # perception-prefix contract used by this PPO/GRPO path.
        if self.model.use_tactile:
            for p in self.model.tactile_encoder.parameters():
                if p.requires_grad:
                    p.requires_grad = False
                    frozen += 1
        # Multi-expert gemma: expert index 0 = paligemma_2b LLM, expert 1 = action expert.
        llm = self.model.llm
        # Shared token embedder feeds the paligemma side of the LLM only.
        for p in llm.embedder.parameters():
            if p.requires_grad:
                p.requires_grad = False
                frozen += 1
        # Per-Block per-expert submodules: ModuleLists indexed by expert.
        for block in llm.layers:
            for sub in (
                block.pre_attention_norms[0],
                block.pre_ffw_norms[0],
                block.mlps[0],
            ):
                for p in sub.parameters():
                    if p.requires_grad:
                        p.requires_grad = False
                        frozen += 1
            attn = block.attn
            for proj_list in (attn.q_proj, attn.k_proj, attn.v_proj, attn.o_proj):
                proj = proj_list[0]
                if proj is None:
                    continue
                for p in proj.parameters():
                    if p.requires_grad:
                        p.requires_grad = False
                        frozen += 1
        # Per-expert final norms.
        if llm.final_norms[0] is not None:
            for p in llm.final_norms[0].parameters():
                if p.requires_grad:
                    p.requires_grad = False
                    frozen += 1
        return frozen
