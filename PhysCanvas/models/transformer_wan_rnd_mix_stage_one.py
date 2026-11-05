import math
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.utils import USE_PEFT_BACKEND, deprecate, logging, scale_lora_layers, unscale_lora_layers
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models._modeling_parallel import ContextParallelInput, ContextParallelOutput
from diffusers.models.attention import AttentionMixin, AttentionModuleMixin, FeedForward
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.embeddings import PixArtAlphaTextProjection, TimestepEmbedding, Timesteps, get_1d_rotary_pos_embed
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import FP32LayerNorm

from .tranformer_wan import WanRotaryPosEmbed, WanTimeTextImageEmbedding, WanTransformerBlock, logger

class WanConditionRNDMixStageOneModel(
    ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin, CacheMixin, AttentionMixin
):
    r"""
    A Transformer model for video-like data used in the Wan model.

    Args:
        patch_size (`Tuple[int]`, defaults to `(1, 2, 2)`):
            3D patch dimensions for video embedding (t_patch, h_patch, w_patch).
        num_attention_heads (`int`, defaults to `40`):
            Fixed length for text embeddings.
        attention_head_dim (`int`, defaults to `128`):
            The number of channels in each head.
        in_channels (`int`, defaults to `16`):
            The number of channels in the input.
        out_channels (`int`, defaults to `16`):
            The number of channels in the output.
        text_dim (`int`, defaults to `512`):
            Input dimension for text embeddings.
        freq_dim (`int`, defaults to `256`):
            Dimension for sinusoidal time embeddings.
        ffn_dim (`int`, defaults to `13824`):
            Intermediate dimension in feed-forward network.
        num_layers (`int`, defaults to `40`):
            The number of layers of transformer blocks to use.
        window_size (`Tuple[int]`, defaults to `(-1, -1)`):
            Window size for local attention (-1 indicates global attention).
        cross_attn_norm (`bool`, defaults to `True`):
            Enable cross-attention normalization.
        qk_norm (`bool`, defaults to `True`):
            Enable query/key normalization.
        eps (`float`, defaults to `1e-6`):
            Epsilon value for normalization layers.
        add_img_emb (`bool`, defaults to `False`):
            Whether to use img_emb.
        added_kv_proj_dim (`int`, *optional*, defaults to `None`):
            The number of channels to use for the added key and value projections. If `None`, no projection is used.
    """

    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["patch_embedding", "replay_embedding", "reference_embedding", "condition_embedder", "norm"]
    _no_split_modules = ["WanTransformerBlock"]
    _keep_in_fp32_modules = ["time_embedder", "scale_shift_table", "norm1", "norm2", "norm3"]
    _keys_to_ignore_on_load_unexpected = ["norm_added_q"]
    _repeated_blocks = ["WanTransformerBlock"]
    _cp_plan = {
        "rope": {
            0: ContextParallelInput(split_dim=1, expected_dims=4, split_output=True),
            1: ContextParallelInput(split_dim=1, expected_dims=4, split_output=True),
        },
        "blocks.0": {
            "hidden_states": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
        },
        "blocks.*": {
            "encoder_hidden_states": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
        },
        "proj_out": ContextParallelOutput(gather_dim=1, expected_dims=3),
        "": {
            "timestep": ContextParallelInput(split_dim=1, expected_dims=2, split_output=False),
        },
    }

    @register_to_config
    def __init__(
        self,
        patch_size: Tuple[int, ...] = (1, 2, 2),
        num_attention_heads: int = 24,
        attention_head_dim: int = 128,
        in_channels: int = 48,
        out_channels: int = 48,
        text_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 14336,
        num_layers: int = 30,
        cross_attn_norm: bool = True,
        qk_norm: Optional[str] = "rms_norm_across_heads",
        eps: float = 1e-6,
        image_dim: Optional[int] = None,
        added_kv_proj_dim: Optional[int] = None,
        rope_max_seq_len: int = 1024,
        pos_embed_seq_len: Optional[int] = None,
        reference_pool_size: int = 2,
        reference_position_scale: float = 0.1,
        reference_time_offset_scale: float = 0.1,
        reference_time_offset_norm: float = 32.0,
        dual_reference_version: int = 1,
    ) -> None:
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels

        # 1. Patch & position embedding
        self.rope = WanRotaryPosEmbed(attention_head_dim, patch_size, rope_max_seq_len)
        self.patch_embedding = nn.Conv3d(in_channels, inner_dim, kernel_size=patch_size, stride=patch_size)

        # replay-only condition patch embedding
        self.replay_embedding = nn.Conv3d(in_channels, inner_dim, kernel_size=patch_size, stride=patch_size)
        self.reference_embedding = nn.Conv3d(in_channels, inner_dim, kernel_size=patch_size, stride=patch_size)
        self.reference_role_embedding = nn.Embedding(2, inner_dim)
        self.reference_gate_logit = nn.Parameter(torch.tensor(-2.0))
        self.reference_pool_size = max(1, int(reference_pool_size))
        self.reference_position_scale = float(reference_position_scale)
        self.reference_time_offset_scale = float(reference_time_offset_scale)
        self.reference_time_offset_norm = max(float(reference_time_offset_norm), 1.0)
        nn.init.zeros_(self.reference_role_embedding.weight)

        # 2. Condition embeddings
        # image_embedding_dim=1280 for I2V model
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=freq_dim,
            time_proj_dim=inner_dim * 6,
            text_embed_dim=text_dim,
            image_embed_dim=image_dim,
            pos_embed_seq_len=pos_embed_seq_len,
        )

        # 3. Transformer blocks
        self.blocks = nn.ModuleList(
            [
                WanTransformerBlock(
                    inner_dim, ffn_dim, num_attention_heads, qk_norm, cross_attn_norm, eps, added_kv_proj_dim
                )
                for _ in range(num_layers)
            ]
        )

        # 4. Output norm & projection
        self.norm_out = FP32LayerNorm(inner_dim, eps, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels * math.prod(patch_size))
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, inner_dim) / inner_dim**0.5)

        self.gradient_checkpointing = False

    @torch.no_grad()
    def initialize_reference_embedding_from_patch(self) -> None:
        if self.reference_embedding.weight.is_meta:
            patch = self.patch_embedding
            self.reference_embedding = nn.Conv3d(
                patch.in_channels,
                patch.out_channels,
                kernel_size=patch.kernel_size,
                stride=patch.stride,
                padding=patch.padding,
                dilation=patch.dilation,
                groups=patch.groups,
                bias=patch.bias is not None,
                device=patch.weight.device,
                dtype=patch.weight.dtype,
            )
        if self.reference_role_embedding.weight.is_meta:
            self.reference_role_embedding = nn.Embedding(
                2,
                self.patch_embedding.out_channels,
                device=self.patch_embedding.weight.device,
                dtype=self.patch_embedding.weight.dtype,
            )
        if self.reference_gate_logit.is_meta:
            self.reference_gate_logit = nn.Parameter(
                torch.tensor(-2.0, device=self.patch_embedding.weight.device, dtype=torch.float32)
            )
        self.reference_embedding.load_state_dict(self.patch_embedding.state_dict())
        self.reference_role_embedding.weight.zero_()
        self.reference_gate_logit.fill_(-2.0)

    def _reference_memory_tokens(
        self,
        global_reference_latents: torch.Tensor,
        global_reference_mask: Optional[torch.Tensor],
        global_reference_role_ids: Optional[torch.Tensor],
        use_role_embedding: bool,
        use_gate: bool,
        global_reference_time_offsets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        reference_states = self.reference_embedding(global_reference_latents)
        if self.reference_pool_size > 1:
            reference_states = F.avg_pool3d(
                reference_states,
                kernel_size=(1, self.reference_pool_size, self.reference_pool_size),
                stride=(1, self.reference_pool_size, self.reference_pool_size),
            )
        batch_size, inner_dim, num_frames, height, width = reference_states.shape
        if self.reference_position_scale:
            coords = [
                torch.linspace(
                    -1.0,
                    1.0,
                    steps=max(size, 1),
                    device=reference_states.device,
                    dtype=reference_states.dtype,
                )
                for size in (num_frames, height, width)
            ]
            grid = torch.stack(torch.meshgrid(*coords, indexing="ij"), dim=0)
            num_frequencies = max(1, math.ceil(inner_dim / 6))
            frequencies = torch.arange(
                1, num_frequencies + 1, device=reference_states.device, dtype=reference_states.dtype
            ).view(-1, 1, 1, 1)
            position_parts = []
            for axis in grid:
                position_parts.extend(
                    [torch.sin(math.pi * frequencies * axis), torch.cos(math.pi * frequencies * axis)]
                )
            position = torch.cat(position_parts, dim=0)
            if position.shape[0] < inner_dim:
                position = F.pad(position, (0, 0, 0, 0, 0, 0, 0, inner_dim - position.shape[0]))
            reference_states = reference_states + self.reference_position_scale * position[:inner_dim].unsqueeze(0)
        if use_role_embedding and global_reference_role_ids is not None:
            role_ids = global_reference_role_ids.to(device=reference_states.device, dtype=torch.long)
            if role_ids.ndim == 1:
                role_ids = role_ids[:, None]
            if role_ids.shape[1] == 1 and num_frames != 1:
                role_ids = role_ids.expand(-1, num_frames)
            if role_ids.shape != (batch_size, num_frames):
                raise ValueError(
                    "global_reference_role_ids must be [B] or [B,T], "
                    f"got {tuple(role_ids.shape)} for reference T={num_frames}"
                )
            role = self.reference_role_embedding(role_ids).transpose(1, 2).unsqueeze(-1).unsqueeze(-1)
            reference_states = reference_states + role
        if self.reference_time_offset_scale and global_reference_time_offsets is not None:
            offsets = global_reference_time_offsets.to(device=reference_states.device, dtype=reference_states.dtype)
            if offsets.ndim == 1:
                offsets = offsets[:, None]
            if offsets.shape[1] == 1 and num_frames != 1:
                offsets = offsets.expand(-1, num_frames)
            if offsets.shape != (batch_size, num_frames):
                raise ValueError(
                    "global_reference_time_offsets must be [B] or [B,T], "
                    f"got {tuple(offsets.shape)} for reference T={num_frames}"
                )
            half_dim = max(1, inner_dim // 2)
            frequencies = torch.exp(
                -math.log(10000.0)
                * torch.arange(half_dim, device=reference_states.device, dtype=reference_states.dtype)
                / max(half_dim - 1, 1)
            )
            phase = offsets[..., None] / self.reference_time_offset_norm * frequencies[None, None, :]
            time_embedding = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)
            if time_embedding.shape[-1] < inner_dim:
                time_embedding = F.pad(time_embedding, (0, inner_dim - time_embedding.shape[-1]))
            time_embedding = time_embedding[..., :inner_dim].transpose(1, 2).unsqueeze(-1).unsqueeze(-1)
            reference_states = reference_states + self.reference_time_offset_scale * time_embedding
        if global_reference_mask is not None:
            mask = global_reference_mask.to(device=reference_states.device, dtype=reference_states.dtype)
            if mask.ndim == 1:
                mask = mask[:, None]
            if mask.shape[1] == 1 and num_frames != 1:
                mask = mask.expand(-1, num_frames)
            if mask.shape != (batch_size, num_frames):
                raise ValueError(
                    "global_reference_mask must be [B] or [B,T], "
                    f"got {tuple(mask.shape)} for reference T={num_frames}"
                )
            reference_states = reference_states * mask[:, None, :, None, None]
        if use_gate:
            gate = torch.sigmoid(self.reference_gate_logit).to(dtype=reference_states.dtype)
            reference_states = reference_states * gate
        return reference_states.flatten(2).transpose(1, 2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        front_depth_condition_latents=None,
        replay_condition_latents=None,
        replay_latents=None,
        global_reference_latents=None,
        global_reference_mask=None,
        global_reference_role_ids=None,
        global_reference_time_offsets=None,
        clip_reference_patch_mask=None,
        clip_reference_role_id=None,
        reference_fusion_mode: str = "clip_only",
        state_latent_height: int = 0,
        mask_state_replay: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        rotary_emb = self.rope(hidden_states)

        hidden_states = self.patch_embedding(hidden_states)
        if replay_condition_latents is None and replay_latents is not None:
            replay_condition_latents = replay_latents
        if replay_condition_latents is not None:
            replay_states = self.replay_embedding(replay_condition_latents)
            if mask_state_replay:
                state_rows = int(state_latent_height) // int(p_h)
                if state_rows <= 0 or state_rows >= replay_states.shape[-2]:
                    raise ValueError(
                        f"Invalid state replay rows={state_rows} for {replay_states.shape[-2]}"
                    )
                replay_states = replay_states.clone()
                replay_states[..., -state_rows:, :] = 0
            hidden_states = hidden_states + replay_states

        if reference_fusion_mode == "dual_memory" and clip_reference_patch_mask is not None:
            clip_mask = clip_reference_patch_mask.to(device=hidden_states.device, dtype=hidden_states.dtype)
            if clip_mask.shape[0] == 1 and batch_size != 1:
                clip_mask = clip_mask.expand(batch_size, -1, -1, -1, -1)
            role_id = clip_reference_role_id
            if role_id is None:
                role_id = torch.ones(batch_size, device=hidden_states.device, dtype=torch.long)
            role_id = role_id.to(device=hidden_states.device, dtype=torch.long).reshape(batch_size)
            clip_role = self.reference_role_embedding(role_id)[:, :, None, None, None]
            hidden_states = hidden_states + clip_mask * clip_role

        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        if timestep.ndim == 2 and timestep.shape[1] != hidden_states.shape[1]:
            target_seq_len = hidden_states.shape[1]
            current_seq_len = timestep.shape[1]
            if current_seq_len % target_seq_len == 0:
                ratio = current_seq_len // target_seq_len
                timestep = timestep.reshape(timestep.shape[0], target_seq_len, ratio).amax(dim=2)
            elif target_seq_len % current_seq_len == 0:
                ratio = target_seq_len // current_seq_len
                timestep = timestep.repeat_interleave(ratio, dim=1)
            else:
                timestep_dtype = timestep.dtype
                timestep = F.interpolate(
                    timestep.unsqueeze(1).float(), size=target_seq_len, mode="nearest"
                ).squeeze(1).to(dtype=timestep_dtype)

        # timestep shape: batch_size, or batch_size, seq_len (wan 2.2 ti2v)
        if timestep.ndim == 2:
            ts_batch_size = timestep.shape[0]
            ts_seq_len = timestep.shape[1]
            timestep = timestep.flatten()  # batch_size * seq_len
        else:
            ts_batch_size = None
            ts_seq_len = None

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image, timestep_seq_len=ts_seq_len
        )
        if ts_seq_len is not None:
            # batch_size, seq_len, 6, inner_dim
            temb = temb.view(ts_batch_size, ts_seq_len, -1)
            timestep_proj = timestep_proj.view(ts_batch_size, ts_seq_len, -1)
            timestep_proj = timestep_proj.unflatten(2, (6, -1))
        else:
            # batch_size, 6, inner_dim
            timestep_proj = timestep_proj.unflatten(1, (6, -1))

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        if global_reference_latents is not None and reference_fusion_mode != "clip_only":
            reference_tokens = self._reference_memory_tokens(
                global_reference_latents,
                global_reference_mask,
                global_reference_role_ids,
                use_role_embedding=reference_fusion_mode == "dual_memory",
                use_gate=reference_fusion_mode == "dual_memory",
                global_reference_time_offsets=global_reference_time_offsets,
            ).to(dtype=encoder_hidden_states.dtype, device=encoder_hidden_states.device)
            encoder_hidden_states = torch.cat([reference_tokens, encoder_hidden_states], dim=1)

        # 4. Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.blocks:
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb
                )
        else:
            for block in self.blocks:
                hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)

        # 5. Output norm, projection & unpatchify
        if temb.ndim == 3:
            # batch_size, seq_len, inner_dim (wan 2.2 ti2v)
            shift, scale = (self.scale_shift_table.unsqueeze(0).to(temb.device) + temb.unsqueeze(2)).chunk(2, dim=2)
            shift = shift.squeeze(2)
            scale = scale.squeeze(2)
        else:
            # batch_size, inner_dim
            shift, scale = (self.scale_shift_table.to(temb.device) + temb.unsqueeze(1)).chunk(2, dim=1)

        # Move the shift and scale tensors to the same device as hidden_states.
        # When using multi-GPU inference via accelerate these will be on the
        # first device rather than the last device, which hidden_states ends up
        # on.
        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)

        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
