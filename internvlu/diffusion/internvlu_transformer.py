# --------------------------------------------------------
# InternVL-U
# Modifications Copyright (c) 2026 OpenGVLab
# This file includes code from Qwen-Image and HuggingFace,
# licensed under the Apache License, Version 2.0.
# --------------------------------------------------------
# Copyright 2025 Qwen-Image Team, The HuggingFace Team. All rights reserved.
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

import inspect
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from transformers.utils import (
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
)
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.utils import (
    USE_PEFT_BACKEND,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.attention import FeedForward
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.transformers.transformer_qwenimage import (
    QwenTimestepProjEmbeddings,
)
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import RMSNorm, FP32LayerNorm
from diffusers.models.modeling_outputs import dataclass, BaseOutput

from .attention_processor import AttentionVE

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

    _flash_supports_window_size = "window_size" in list(
        inspect.signature(flash_attn_func).parameters
    )

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class Transformer2DModelOutput(BaseOutput):
    """
    The output of [`Transformer2DModel`].

    Args:
        sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)` or
        `(batch size, num_vector_embeds - 1, num_latent_pixels)` if [`Transformer2DModel`] is discrete):
            The hidden states output conditioned on the `encoder_hidden_states` input. If discrete, returns probability
            distributions for the unnoised latent pixels.
    """

    sample: "torch.Tensor"  # noqa: F821


def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


def _basic_init(module):
    if isinstance(module, nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Conv2d):
        w = module.weight
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
    elif isinstance(module, RMSNorm):
        if module.weight is not None:
            nn.init.constant_(module.weight, 1)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


def apply_rotary_emb_ms(
    x: torch.Tensor,
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]],
    use_real: bool = True,
    use_real_unbind_dim: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor. This function applies rotary embeddings
    to the given query or key 'x' tensors using the provided frequency tensor 'freqs_cis'. The input tensors are
    reshaped as complex numbers, and the frequency tensor is reshaped for broadcasting compatibility. The resulting
    tensors contain rotary embeddings and are returned as real tensors.

    Args:
        x (`torch.Tensor`):
            Query or key tensor to apply rotary embeddings. [B, S, H, D] xk (torch.Tensor): Key tensor to apply
        freqs_cis (`Tuple[torch.Tensor]`): Precomputed frequency tensor for complex exponentials. ([S, D], [S, D],)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    if use_real:
        cos, sin = freqs_cis
        cos = cos[None, None]
        sin = sin[None, None]
        cos, sin = cos.to(x.device), sin.to(x.device)

        if use_real_unbind_dim == -1:
            x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
            x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        elif use_real_unbind_dim == -2:
            x_real, x_imag = x.reshape(*x.shape[:-1], 2, -1).unbind(-2)
            x_rotated = torch.cat([-x_imag, x_real], dim=-1)
        else:
            raise ValueError(
                f"`use_real_unbind_dim={use_real_unbind_dim}` but should be -1 or -2."
            )

        out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)

        return out
    else:
        x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.unsqueeze(-2)
        x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)

        return x_out.type_as(x)


class UnifiedMSRoPE(nn.Module):
    """
    Unified multi-scale RoPE module for 3D position encodings.

    Args:
        theta (`int`): Base frequency for rotary embeddings.
        axes_dim (`List[int]`): Per-axis embedding dimensions (frame, height, width).
        scale_rope (`bool`, *optional*, defaults to `False`): Whether to apply scaling to cosine/sine outputs.
    """

    def __init__(self, theta: int, axes_dim: List[int], scale_rope=False):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        self.scale_rope = scale_rope

        inv_freqs_list = []
        self.axis_offsets = [0]

        for dim in self.axes_dim:
            assert dim % 2 == 0, f"Dimension {dim} must be even"
            inv_freq = 1.0 / torch.pow(
                theta, torch.arange(0, dim, 2).to(torch.float32).div(dim)
            )
            inv_freqs_list.append(inv_freq)
            self.axis_offsets.append(self.axis_offsets[-1] + len(inv_freq))

        self.all_inv_freqs = torch.cat(inv_freqs_list, dim=0).float()

    def get_inv_freq(self, axis_idx, device):
        start_idx = self.axis_offsets[axis_idx]
        end_idx = self.axis_offsets[axis_idx + 1]
        return self.all_inv_freqs[start_idx:end_idx].to(device)

    def rope_params(self, positions, axis_idx):
        """
        Args:
            positions (`torch.Tensor`):
                Position tensor of shape `[N]` (supports float or negative values).
            axis_idx (`int`):
                Axis index (0: frame, 1: height, 2: width).

        Returns:
            `torch.Tensor`: Complex frequencies of shape `[N, dim // 2]`.
        """
        positions = positions.float()
        inv_freq = self.get_inv_freq(axis_idx, positions.device).float()
        freqs = torch.outer(positions, inv_freq)
        freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_complex

    def forward(self, position_ids_3d, device=None):
        if device is None:
            device = position_ids_3d.device

        position_ids_3d = position_ids_3d.to(device)
        L = position_ids_3d.shape[0]

        total_dim = sum(self.axes_dim) // 2
        freqs_result = torch.zeros(L, total_dim, dtype=torch.complex64, device=device)

        dim_offset = 0
        for axis_idx in range(3):
            axis_dim = self.axes_dim[axis_idx] // 2
            axis_positions = position_ids_3d[:, axis_idx]

            axis_freqs = self.rope_params(axis_positions, axis_idx)
            freqs_result[:, dim_offset : dim_offset + axis_dim] = axis_freqs

            dim_offset += axis_dim

        return freqs_result

    def get_cos_sin(self, position_ids_3d, device=None):
        freqs_complex = self.forward(position_ids_3d, device)

        cos_freqs = freqs_complex.real
        sin_freqs = freqs_complex.imag

        cos = torch.cat([cos_freqs, cos_freqs], dim=-1)
        sin = torch.cat([sin_freqs, sin_freqs], dim=-1)

        if self.scale_rope:
            attention_scaling = 1.0
            cos = cos * attention_scaling
            sin = sin * attention_scaling

        return cos, sin


def get_video_scale_factors(video_scale_factor, batch_size, video_fhw):
    """Convert video_scale_factor to per-video scale factors"""
    if video_scale_factor is None:
        return [
            [1.0] * len(sample_videos) if sample_videos else [1.0]
            for sample_videos in (video_fhw or [[]] * batch_size)
        ]

    if isinstance(video_scale_factor, (int, float)):
        return [
            (
                [video_scale_factor] * len(sample_videos)
                if sample_videos
                else [video_scale_factor]
            )
            for sample_videos in (video_fhw or [[]] * batch_size)
        ]

    if isinstance(video_scale_factor, list):
        if len(video_scale_factor) == 0:
            return [
                [1.0] * len(sample_videos) if sample_videos else [1.0]
                for sample_videos in (video_fhw or [[]] * batch_size)
            ]

        if isinstance(video_scale_factor[0], list):
            result = []
            for batch_idx in range(batch_size):
                if batch_idx < len(video_scale_factor):
                    result.append(video_scale_factor[batch_idx])
                else:
                    sample_videos = (
                        video_fhw[batch_idx]
                        if video_fhw and batch_idx < len(video_fhw)
                        else []
                    )
                    if not isinstance(sample_videos, list) or (
                        len(sample_videos) > 0
                        and not isinstance(sample_videos[0], (list, tuple))
                    ):
                        sample_videos = [sample_videos] if sample_videos else []
                    result.append([1.0] * len(sample_videos))
            return result
        else:
            result = []
            for batch_idx in range(batch_size):
                batch_scale = (
                    video_scale_factor[batch_idx]
                    if batch_idx < len(video_scale_factor)
                    else 1.0
                )
                sample_videos = (
                    video_fhw[batch_idx]
                    if video_fhw and batch_idx < len(video_fhw)
                    else []
                )
                if not isinstance(sample_videos, list) or (
                    len(sample_videos) > 0
                    and not isinstance(sample_videos[0], (list, tuple))
                ):
                    sample_videos = [sample_videos] if sample_videos else []
                result.append([batch_scale] * len(sample_videos))
            return result

    return [
        [1.0] * len(sample_videos) if sample_videos else [1.0]
        for sample_videos in (video_fhw or [[]] * batch_size)
    ]


def create_position_ids_3d_v2(
    video_fhw: Optional[Union[List, List[List]]] = None,
    input_token_mask: Optional[torch.Tensor] = None,
    scale_rope: bool = True,
    video_scale_factor: Optional[Union[float, List[float], List[List[float]]]] = None,
    device: Optional[torch.device] = None,
):
    """
    Optimized helper method to create 3D position IDs for interleaved text-video sequences
    Uses tensor operations instead of loops for better performance

    Args:
        video_fhw: batch of video specs, where each element is a list of [frame, height, width] for each video
                  Shape: [batch_size, num_videos_per_sample, 3] or [batch_size, 3] for single video per sample
        input_token_mask: batch of token type masks, where True indicates video token, False indicates text token
                         Shape: [batch_size, sequence_length]
                         The number of True values should match the total f*h*w from video_fhw
        scale_rope: bool, whether to use centered indexing for position IDs
        video_scale_factor: float, list of floats, or nested list of floats for scaling video position IDs
                           - float: same scale for all videos
                           - List[float]: scale per batch (all videos in a batch use same scale)
                           - List[List[float]]: scale per video (each video can have different scale)
        device: torch device for the output tensor

    Returns:
        position_ids_3d: Tensor of shape [total_tokens, 3] where each row is [frame_idx, height_idx, width_idx]
    """
    if video_fhw is None and input_token_mask is None:
        return torch.empty(0, 3, dtype=torch.float32, device=device)

    batch_size = (
        len(input_token_mask) if input_token_mask is not None else len(video_fhw)
    )

    video_scale_factors_per_video = get_video_scale_factors(
        video_scale_factor, batch_size, video_fhw
    )

    batch_position_ids = []

    for batch_idx in range(batch_size):
        if input_token_mask is not None and batch_idx < len(input_token_mask):
            token_mask = input_token_mask[batch_idx]
        else:
            total_video_tokens = sum(
                f * h * w for f, h, w in (video_fhw[batch_idx] if video_fhw else [])
            )
            token_mask = torch.ones(total_video_tokens, dtype=torch.bool, device=device)

        seq_len = len(token_mask)

        sample_video_fhw = []
        if video_fhw is not None and batch_idx < len(video_fhw):
            sample_video_fhw = video_fhw[batch_idx]

            if not isinstance(sample_video_fhw, list):
                sample_video_fhw = [sample_video_fhw]

            if len(sample_video_fhw) > 0 and not isinstance(
                sample_video_fhw[0], (list, tuple)
            ):
                sample_video_fhw = [sample_video_fhw]

        current_batch_scales = (
            video_scale_factors_per_video[batch_idx]
            if batch_idx < len(video_scale_factors_per_video)
            else [1.0]
        )

        video_position_blocks = []
        video_token_counts = []

        if sample_video_fhw:
            for video_idx, fhw in enumerate(sample_video_fhw):
                frame, height, width = fhw
                video_token_counts.append(frame * height * width)

                current_video_scale = (
                    current_batch_scales[video_idx]
                    if video_idx < len(current_batch_scales)
                    else 1.0
                )

                video_pos = _create_single_video_positions(
                    frame,
                    height,
                    width,
                    cum_frame=0,
                    scale_rope=scale_rope,
                    scale_factor=current_video_scale,
                    device=device,
                )
                video_position_blocks.append(video_pos)

        video_mask = token_mask

        position_ids = torch.zeros(seq_len, 3, dtype=torch.float32, device=device)

        if len(video_position_blocks) > 0:
            mask_diff = torch.diff(
                video_mask.float(), prepend=torch.tensor([0.0], device=device)
            )
            video_starts = torch.where(mask_diff == 1)[0]
            video_ends = torch.where(mask_diff == -1)[0]

            if len(video_ends) < len(video_starts):
                video_ends = torch.cat(
                    [video_ends, torch.tensor([seq_len], device=device)]
                )

            current_text_pos = 0
            video_block_idx = 0

            if len(video_starts) == 0 or video_starts[0] > 0:
                first_video_start = (
                    video_starts[0] if len(video_starts) > 0 else seq_len
                )
                text_positions = torch.arange(
                    current_text_pos,
                    current_text_pos + first_video_start,
                    dtype=torch.float32,
                    device=device,
                )
                position_ids[:first_video_start] = text_positions.unsqueeze(1).expand(
                    -1, 3
                )
                current_text_pos += first_video_start

            for seg_idx in range(len(video_starts)):
                video_start = video_starts[seg_idx]
                video_end = video_ends[seg_idx]
                video_length = video_end - video_start

                if video_block_idx < len(video_position_blocks):
                    video_pos = video_position_blocks[video_block_idx]
                    if len(video_pos) == video_length:
                        adjusted_video_pos = video_pos.clone()
                        adjusted_video_pos[:, 0] += current_text_pos
                        position_ids[video_start:video_end] = adjusted_video_pos

                        max_pos = torch.max(adjusted_video_pos).item()
                        current_text_pos = int(max_pos) + 1
                    else:
                        fallback_pos = torch.arange(
                            current_text_pos,
                            current_text_pos + video_length,
                            dtype=torch.float32,
                            device=device,
                        )
                        position_ids[video_start:video_end] = fallback_pos.unsqueeze(
                            1
                        ).expand(-1, 3)
                        current_text_pos += video_length

                    video_block_idx += 1

                next_video_start = (
                    video_starts[seg_idx + 1]
                    if seg_idx + 1 < len(video_starts)
                    else seq_len
                )
                text_segment_length = next_video_start - video_end

                if text_segment_length > 0:
                    text_positions = torch.arange(
                        current_text_pos,
                        current_text_pos + text_segment_length,
                        dtype=torch.float32,
                        device=device,
                    )
                    position_ids[video_end:next_video_start] = text_positions.unsqueeze(
                        1
                    ).expand(-1, 3)
                    current_text_pos += text_segment_length

        else:
            text_positions = torch.arange(seq_len, dtype=torch.float32, device=device)
            position_ids = text_positions.unsqueeze(1).expand(-1, 3)

        batch_position_ids.append(position_ids)

    if batch_position_ids:
        result = torch.cat(batch_position_ids, dim=0)
    else:
        result = torch.empty(0, 3, dtype=torch.float32, device=device)

    if device is not None:
        result = result.to(device)

    return result


def create_position_ids_3d_v3(
    video_fhw: Optional[Union[List, List[List], List[List[List]]]] = None,
    input_token_mask: Optional[torch.Tensor] = None,
    scale_rope: bool = True,
    video_scale_factor: Optional[
        Union[float, List[float], List[List[float]], List[List[List[float]]]]
    ] = None,
    device: Optional[torch.device] = None,
):
    """
    Optimized helper method to create 3D position IDs for interleaved text-video sequences
    Uses tensor operations instead of loops for better performance

    Args:
        video_fhw: batch of video specs, supports multiple formats:
                  - [batch_size, 3]: single video per sample
                  - [batch_size, num_videos_per_sample, 3]: multiple videos per sample
                  - [batch_size, num_videos_per_sample, num_flips_per_video, 3]: multiple videos with flips per sample
        input_token_mask: batch of token type masks, where True indicates video token, False indicates text token
                         Shape: [batch_size, sequence_length]
                         The number of True values should match the total f*h*w from video_fhw
        scale_rope: bool, whether to use centered indexing for position IDs
        video_scale_factor: scaling factors, supports multiple formats:
                           - float: same scale for all videos
                           - List[float]: scale per batch
                           - List[List[float]]: scale per video
                           - List[List[List[float]]]: scale per flip
        device: torch device for the output tensor

    Returns:
        position_ids_3d: Tensor of shape [total_tokens, 3] where each row is [frame_idx, height_idx, width_idx]
    """
    if video_fhw is None and input_token_mask is None:
        return torch.empty(0, 3, dtype=torch.float32, device=device)

    batch_size = (
        len(input_token_mask) if input_token_mask is not None else len(video_fhw)
    )

    video_scale_factors_per_video = get_video_scale_factors_with_flips(
        video_scale_factor, batch_size, video_fhw
    )

    batch_position_ids = []

    for batch_idx in range(batch_size):
        if input_token_mask is not None and batch_idx < len(input_token_mask):
            token_mask = input_token_mask[batch_idx]
        else:
            total_video_tokens = calculate_total_video_tokens(
                video_fhw[batch_idx] if video_fhw else []
            )
            token_mask = torch.ones(total_video_tokens, dtype=torch.bool, device=device)

        seq_len = len(token_mask)

        sample_video_fhw = []
        if video_fhw is not None and batch_idx < len(video_fhw):
            sample_video_fhw = video_fhw[batch_idx]

            if not isinstance(sample_video_fhw, list):
                sample_video_fhw = [sample_video_fhw]

            if len(sample_video_fhw) > 0 and not isinstance(
                sample_video_fhw[0], (list, tuple)
            ):
                sample_video_fhw = [sample_video_fhw]

        current_batch_scales = (
            video_scale_factors_per_video[batch_idx]
            if batch_idx < len(video_scale_factors_per_video)
            else [[[1.0]]]
        )

        video_position_blocks = []

        if sample_video_fhw:
            for video_idx, video_data in enumerate(sample_video_fhw):
                video_scales = (
                    current_batch_scales[video_idx]
                    if video_idx < len(current_batch_scales)
                    else [[1.0]]
                )

                if (
                    isinstance(video_data, list)
                    and len(video_data) > 0
                    and isinstance(video_data[0], list)
                    and len(video_data[0]) == 3
                ):

                    video_flip_positions = []

                    cum_frame = 0
                    for flip_idx, fhw in enumerate(video_data):
                        frame, height, width = fhw

                        flip_scale = (
                            video_scales[flip_idx]
                            if flip_idx < len(video_scales)
                            else 1.0
                        )
                        if isinstance(flip_scale, list):
                            flip_scale = flip_scale[0] if len(flip_scale) > 0 else 1.0

                        flip_pos = _create_single_video_positions(
                            frame,
                            height,
                            width,
                            cum_frame=cum_frame,
                            scale_rope=scale_rope,
                            scale_factor=flip_scale,
                            device=device,
                        )
                        video_flip_positions.append(flip_pos)
                        cum_frame += frame

                    if video_flip_positions:
                        video_pos = torch.cat(video_flip_positions, dim=0)
                        video_position_blocks.append(video_pos)

                else:
                    frame, height, width = video_data

                    video_scale = video_scales[0] if len(video_scales) > 0 else 1.0
                    if isinstance(video_scale, list):
                        video_scale = video_scale[0] if len(video_scale) > 0 else 1.0
                        if isinstance(video_scale, list):
                            video_scale = (
                                video_scale[0] if len(video_scale) > 0 else 1.0
                            )

                    video_pos = _create_single_video_positions(
                        frame,
                        height,
                        width,
                        cum_frame=0,
                        scale_rope=scale_rope,
                        scale_factor=video_scale,
                        device=device,
                    )
                    video_position_blocks.append(video_pos)

        video_mask = token_mask

        position_ids = torch.zeros(seq_len, 3, dtype=torch.float32, device=device)

        if len(video_position_blocks) > 0:
            mask_diff = torch.diff(
                video_mask.float(), prepend=torch.tensor([0.0], device=device)
            )
            video_starts = torch.where(mask_diff == 1)[0]
            video_ends = torch.where(mask_diff == -1)[0]

            if len(video_ends) < len(video_starts):
                video_ends = torch.cat(
                    [video_ends, torch.tensor([seq_len], device=device)]
                )

            current_text_pos = 0
            video_block_idx = 0

            if len(video_starts) == 0 or video_starts[0] > 0:
                first_video_start = (
                    video_starts[0] if len(video_starts) > 0 else seq_len
                )
                text_positions = torch.arange(
                    current_text_pos,
                    current_text_pos + first_video_start,
                    dtype=torch.float32,
                    device=device,
                )
                position_ids[:first_video_start] = text_positions.unsqueeze(1).expand(
                    -1, 3
                )
                current_text_pos += first_video_start

            for seg_idx in range(len(video_starts)):
                video_start = video_starts[seg_idx]
                video_end = video_ends[seg_idx]
                video_length = video_end - video_start

                if video_block_idx < len(video_position_blocks):
                    video_pos = video_position_blocks[video_block_idx]
                    if len(video_pos) == video_length:
                        adjusted_video_pos = video_pos.clone()
                        adjusted_video_pos[:, 0] += current_text_pos
                        position_ids[video_start:video_end] = adjusted_video_pos

                        max_pos = torch.max(adjusted_video_pos).item()
                        current_text_pos = int(max_pos) + 1
                    else:
                        fallback_pos = torch.arange(
                            current_text_pos,
                            current_text_pos + video_length,
                            dtype=torch.float32,
                            device=device,
                        )
                        position_ids[video_start:video_end] = fallback_pos.unsqueeze(
                            1
                        ).expand(-1, 3)
                        current_text_pos += video_length

                    video_block_idx += 1

                next_video_start = (
                    video_starts[seg_idx + 1]
                    if seg_idx + 1 < len(video_starts)
                    else seq_len
                )
                text_segment_length = next_video_start - video_end

                if text_segment_length > 0:
                    text_positions = torch.arange(
                        current_text_pos,
                        current_text_pos + text_segment_length,
                        dtype=torch.float32,
                        device=device,
                    )
                    position_ids[video_end:next_video_start] = text_positions.unsqueeze(
                        1
                    ).expand(-1, 3)
                    current_text_pos += text_segment_length

        else:
            text_positions = torch.arange(seq_len, dtype=torch.float32, device=device)
            position_ids = text_positions.unsqueeze(1).expand(-1, 3)

        batch_position_ids.append(position_ids)

    if batch_position_ids:
        result = torch.cat(batch_position_ids, dim=0)
    else:
        result = torch.empty(0, 3, dtype=torch.float32, device=device)

    if device is not None:
        result = result.to(device)

    return result


def get_video_scale_factors_with_flips(video_scale_factor, batch_size, video_fhw):
    """
    Helper function to handle video scale factors with flip support
    """
    if video_scale_factor is None:
        return [[[1.0]] for _ in range(batch_size)]

    if isinstance(video_scale_factor, (int, float)):
        return [[[video_scale_factor]] for _ in range(batch_size)]

    if isinstance(video_scale_factor, list):
        if len(video_scale_factor) == 0:
            return [[[1.0]] for _ in range(batch_size)]

        if isinstance(video_scale_factor[0], (int, float)):
            result = []
            for batch_idx in range(batch_size):
                scale = (
                    video_scale_factor[batch_idx]
                    if batch_idx < len(video_scale_factor)
                    else 1.0
                )
                result.append([[scale]])
            return result

        elif isinstance(video_scale_factor[0], list):
            if len(video_scale_factor[0]) > 0 and isinstance(
                video_scale_factor[0][0], (int, float)
            ):
                result = []
                for batch_idx in range(batch_size):
                    batch_scales = (
                        video_scale_factor[batch_idx]
                        if batch_idx < len(video_scale_factor)
                        else [1.0]
                    )
                    video_scales = []
                    for video_scale in batch_scales:
                        video_scales.append([video_scale])
                    result.append(video_scales)
                return result

            elif len(video_scale_factor[0]) > 0 and isinstance(
                video_scale_factor[0][0], list
            ):
                return video_scale_factor

    return [[[1.0]] for _ in range(batch_size)]


def calculate_total_video_tokens(sample_video_fhw):
    """
    Calculate total number of video tokens for a sample
    """
    if not sample_video_fhw:
        return 0

    total = 0

    if not isinstance(sample_video_fhw, list):
        return 0

    for video_data in sample_video_fhw:
        if (
            isinstance(video_data, list)
            and len(video_data) > 0
            and isinstance(video_data[0], list)
            and len(video_data[0]) == 3
        ):
            for fhw in video_data:
                f, h, w = fhw
                total += f * h * w
        elif isinstance(video_data, list) and len(video_data) == 3:
            f, h, w = video_data
            total += f * h * w

    return total


def _create_single_video_positions(
    frame,
    height,
    width,
    cum_frame=0,
    scale_rope=True,
    scale_factor=1.0,
    device=None,
):
    """Helper function to create position IDs for a single video"""

    if scale_rope:
        h_neg_count = height - height // 2
        w_neg_count = width - width // 2

        h_indices = torch.cat(
            [
                torch.arange(-h_neg_count, 0, dtype=torch.float32, device=device),
                torch.arange(0, height // 2, dtype=torch.float32, device=device),
            ]
        )
        w_indices = torch.cat(
            [
                torch.arange(-w_neg_count, 0, dtype=torch.float32, device=device),
                torch.arange(0, width // 2, dtype=torch.float32, device=device),
            ]
        )

        if scale_factor != 1.0:
            target_h_neg_count = int(h_neg_count / scale_factor)
            target_h_pos_count = int((height // 2) / scale_factor)
            target_w_neg_count = int(w_neg_count / scale_factor)
            target_w_pos_count = int((width // 2) / scale_factor)

            h_min_orig, h_max_orig = -h_neg_count, height // 2 - 1
            h_min_target, h_max_target = -target_h_neg_count, target_h_pos_count - 1
            if h_max_orig != h_min_orig:
                h_indices = h_min_target + (h_indices - h_min_orig) * (
                    h_max_target - h_min_target
                ) / (h_max_orig - h_min_orig)

            w_min_orig, w_max_orig = -w_neg_count, width // 2 - 1
            w_min_target, w_max_target = -target_w_neg_count, target_w_pos_count - 1
            if w_max_orig != w_min_orig:
                w_indices = w_min_target + (w_indices - w_min_orig) * (
                    w_max_target - w_min_target
                ) / (w_max_orig - w_min_orig)

    else:
        h_indices = torch.arange(height, dtype=torch.float32, device=device)
        w_indices = torch.arange(width, dtype=torch.float32, device=device)

        if scale_factor != 1.0:
            target_height = int(height / scale_factor)
            target_width = int(width / scale_factor)

            h_min_orig, h_max_orig = 0, height - 1
            h_min_target, h_max_target = 0, target_height - 1
            if h_max_orig != h_min_orig:
                h_indices = h_min_target + (h_indices - h_min_orig) * (
                    h_max_target - h_min_target
                ) / (h_max_orig - h_min_orig)

            w_min_orig, w_max_orig = 0, width - 1
            w_min_target, w_max_target = 0, target_width - 1
            if w_max_orig != w_min_orig:
                w_indices = w_min_target + (w_indices - w_min_orig) * (
                    w_max_target - w_min_target
                ) / (w_max_orig - w_min_orig)

    f_indices = torch.arange(
        cum_frame, cum_frame + frame, dtype=torch.float32, device=device
    )

    f_grid, h_grid, w_grid = torch.meshgrid(
        f_indices, h_indices, w_indices, indexing="ij"
    )

    video_positions = torch.stack(
        [f_grid.flatten(), h_grid.flatten(), w_grid.flatten()], dim=1
    )

    return video_positions


@maybe_allow_in_graph
class InternVLUTransformerBlock(nn.Module):
    """
    Dual-stream transformer block used by InternVLU diffusion models.

    The block applies modulation, joint attention, and feed-forward layers with separate
    normalization paths for image and text tokens.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        qk_norm: str = "rms_norm",
        eps: float = 1e-6,
    ):
        super().__init__()

        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim

        self.img_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True),
        )
        self.img_norm1 = FP32LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.attn = AttentionVE(
            query_dim=dim,
            cross_attention_dim=None,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=False,
            bias=True,
            processor=(InternVLUDoubleStreamFlashAttnProcessor()if is_flash_attn_2_available() else InternVLUDoubleStreamTorchAttnProcessor()),
            qk_norm=qk_norm,
            eps=eps,
        )
        self.img_norm2 = FP32LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.img_mlp = FeedForward(
            dim=dim, dim_out=dim, activation_fn="gelu-approximate"
        )

        self.txt_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True),
        )
        self.txt_norm1 = FP32LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.txt_norm2 = FP32LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.txt_mlp = FeedForward(
            dim=dim, dim_out=dim, activation_fn="gelu-approximate"
        )

    def _modulate(self, x, mod_params):
        """Apply modulation to input tensor"""
        shift, scale, gate = mod_params.chunk(3, dim=-1)
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1), gate.unsqueeze(1)

    def _forward_ve(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        enc_token_mask: Optional[torch.Tensor] = None,
        padding_type: str = "pad",
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        batch_size, seq_len = hidden_states.shape[:2]

        img_mod_params = self.img_mod(temb)
        txt_mod_params = self.txt_mod(temb)

        enc_mask = enc_token_mask.unsqueeze(-1).to(hidden_states.dtype)
        img_mask = 1.0 - enc_mask

        if padding_type == "pack":
            per_seq_lens = attention_mask[0, 1:] - attention_mask[0, :-1]
            img_mod_params = img_mod_params.repeat_interleave(per_seq_lens, dim=0)[None]
            txt_mod_params = txt_mod_params.repeat_interleave(per_seq_lens, dim=0)[None]
            mod_params = img_mod_params * img_mask + txt_mod_params * enc_mask
        elif padding_type == "pad":
            img_expanded = img_mod_params.unsqueeze(1).expand(-1, seq_len, -1)
            txt_expanded = txt_mod_params.unsqueeze(1).expand(-1, seq_len, -1)
            mod_params = img_expanded * img_mask + txt_expanded * enc_mask

        mod1, mod2 = mod_params.chunk(2, dim=-1)
        shift1, scale1, gate1 = mod1.chunk(3, dim=-1)
        shift2, scale2, gate2 = mod2.chunk(3, dim=-1)

        img_normed = self.img_norm1(hidden_states)
        txt_normed = self.txt_norm1(hidden_states)

        normed_states = img_normed * img_mask + txt_normed * enc_mask
        modulated_states = normed_states * (1 + scale1) + shift1

        joint_attention_kwargs = joint_attention_kwargs or {}
        attn_output = self.attn(
            hidden_states=modulated_states,
            attention_mask=attention_mask,
            image_rotary_emb=image_rotary_emb,
            enc_token_mask=enc_token_mask,
            attn_mode="ve",
            padding_type=padding_type,
            **joint_attention_kwargs,
        )

        gated_attn = gate1 * attn_output
        hidden_states = hidden_states + gated_attn

        img_normed = self.img_norm2(hidden_states)
        txt_normed = self.txt_norm2(hidden_states)

        normed_states = img_normed * img_mask + txt_normed * enc_mask
        modulated_states = normed_states * (1 + scale2) + shift2

        img_mlp_out = self.img_mlp(modulated_states)
        txt_mlp_out = self.txt_mlp(modulated_states)

        mlp_output = img_mlp_out * img_mask + txt_mlp_out * enc_mask
        hidden_states = hidden_states + gate2 * mlp_output

        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        return hidden_states

    def _forward_ori(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_mask: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        img_mod_params = self.img_mod(temb)
        txt_mod_params = self.txt_mod(temb)

        img_mod1, img_mod2 = img_mod_params.chunk(2, dim=-1)
        txt_mod1, txt_mod2 = txt_mod_params.chunk(2, dim=-1)

        img_normed = self.img_norm1(hidden_states)
        img_modulated, img_gate1 = self._modulate(img_normed, img_mod1)

        txt_normed = self.txt_norm1(encoder_hidden_states)
        txt_modulated, txt_gate1 = self._modulate(txt_normed, txt_mod1)

        joint_attention_kwargs = joint_attention_kwargs or {}
        attn_output = self.attn(
            hidden_states=img_modulated,
            encoder_hidden_states=txt_modulated,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            image_rotary_emb=image_rotary_emb,
            attn_mode="default",
            **joint_attention_kwargs,
        )

        img_attn_output, txt_attn_output = attn_output

        hidden_states = hidden_states + img_gate1 * img_attn_output
        encoder_hidden_states = encoder_hidden_states + txt_gate1 * txt_attn_output

        img_normed2 = self.img_norm2(hidden_states)
        img_modulated2, img_gate2 = self._modulate(img_normed2, img_mod2)
        img_mlp_output = self.img_mlp(img_modulated2)
        hidden_states = hidden_states + img_gate2 * img_mlp_output

        txt_normed2 = self.txt_norm2(encoder_hidden_states)
        txt_modulated2, txt_gate2 = self._modulate(txt_normed2, txt_mod2)
        txt_mlp_output = self.txt_mlp(txt_modulated2)
        encoder_hidden_states = encoder_hidden_states + txt_gate2 * txt_mlp_output

        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        enc_token_mask: Optional[torch.Tensor] = None,
        padding_type: str = "pad",
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        output_ve = self._forward_ve(
            hidden_states,
            temb,
            attention_mask=attention_mask,
            image_rotary_emb=image_rotary_emb,
            enc_token_mask=enc_token_mask,
            padding_type=padding_type,
            joint_attention_kwargs=joint_attention_kwargs,
        )

        return output_ve


class InternVLUDoubleStreamFlashAttnProcessor:
    """
    Flash Attention 2 processor for InternVLU double-stream architecture, matching DoubleStreamLayerMegatron logic.
    This processor implements joint attention computation where text and image streams are processed together.
    """

    _attention_backend = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "InternVLUDoubleStreamFlashAttnProcessor requires PyTorch 2.0, please upgrade PyTorch to 2.0."
            )
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()
        self.is_causal = False

    def _call_ve(
        self,
        attn: AttentionVE,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        enc_token_mask: Optional[torch.Tensor] = None,
        padding_type: str = "pad",
    ):
        dtype = attn.to_v.weight.dtype

        enc_mask = enc_token_mask.unsqueeze(-1).to(dtype)
        img_mask = 1.0 - enc_mask

        q_enc = attn.add_q_proj(hidden_states).unflatten(-1, (attn.heads, -1))
        k_enc = attn.add_k_proj(hidden_states).unflatten(-1, (attn.heads, -1))
        v_enc = attn.add_v_proj(hidden_states)

        q_img = attn.to_q(hidden_states).unflatten(-1, (attn.heads, -1))
        k_img = attn.to_k(hidden_states).unflatten(-1, (attn.heads, -1))
        v_img = attn.to_v(hidden_states)

        bsz, q_len = q_enc.shape[:2]
        head_dim = attn.out_dim // attn.heads

        q_enc, gate_score_enc = torch.split(q_enc, [head_dim, head_dim], dim=-1)
        gate_score_enc = gate_score_enc.reshape(bsz, q_len, -1)
        q_img, gate_score_img = torch.split(q_img, [head_dim, head_dim], dim=-1)
        gate_score_img = gate_score_img.reshape(bsz, q_len, -1)

        q_enc = attn.norm_added_q(q_enc)
        k_enc = attn.norm_added_k(k_enc)
        q_img = attn.norm_q(q_img)
        k_img = attn.norm_k(k_img)

        joint_query = (
            q_enc * enc_mask.unsqueeze(-1) + q_img * img_mask.unsqueeze(-1)
        ).to(dtype)
        joint_key = (
            k_enc * enc_mask.unsqueeze(-1) + k_img * img_mask.unsqueeze(-1)
        ).to(dtype)
        joint_value = (
            (v_enc * enc_mask + v_img * img_mask)
            .unflatten(-1, (attn.heads, -1))
            .to(dtype)
        )

        if image_rotary_emb is not None:
            joint_freqs = image_rotary_emb
            joint_query = apply_rotary_emb_ms(joint_query, joint_freqs, use_real=False)
            joint_key = apply_rotary_emb_ms(joint_key, joint_freqs, use_real=False)

        bsz, q_len = joint_query.shape[:2]

        attn_dtype = torch.bfloat16
        joint_hidden_states = self._flash_attention_forward(
            joint_query.to(dtype=attn_dtype),
            joint_key.to(dtype=attn_dtype),
            joint_value.to(dtype=attn_dtype),
            attention_mask=attention_mask,
            query_length=q_len,
            dropout=0.0,
            use_sliding_windows=False,
            padding_type=padding_type,
        )

        joint_hidden_states = joint_hidden_states.reshape(bsz, q_len, -1).contiguous()
        joint_hidden_states = joint_hidden_states.to(joint_query.dtype)

        enc_output = attn.to_add_out(joint_hidden_states)

        img_output = attn.to_out[0](joint_hidden_states)
        if len(attn.to_out) > 1:
            img_output = attn.to_out[1](img_output)

        attn_output = enc_output * enc_mask + img_output * img_mask

        gate_score = gate_score_enc * enc_mask + gate_score_img * img_mask
        attn_output = attn_output * torch.sigmoid(gate_score)

        return attn_output

    def _call_ori(
        self,
        attn: AttentionVE,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        encoder_hidden_states_mask: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        padding_type: str = "pad",
    ) -> torch.FloatTensor:
        if encoder_hidden_states is None:
            raise ValueError(
                "InternVLUDoubleStreamFlashAttnProcessor requires encoder_hidden_states (text stream)"
            )

        seq_txt = encoder_hidden_states.shape[1]

        img_query = attn.to_q(hidden_states)
        img_key = attn.to_k(hidden_states)
        img_value = attn.to_v(hidden_states)

        txt_query = attn.add_q_proj(encoder_hidden_states)
        txt_key = attn.add_k_proj(encoder_hidden_states)
        txt_value = attn.add_v_proj(encoder_hidden_states)

        img_query = img_query.unflatten(-1, (attn.heads, -1))
        img_key = img_key.unflatten(-1, (attn.heads, -1))
        img_value = img_value.unflatten(-1, (attn.heads, -1))

        txt_query = txt_query.unflatten(-1, (attn.heads, -1))
        txt_key = txt_key.unflatten(-1, (attn.heads, -1))
        txt_value = txt_value.unflatten(-1, (attn.heads, -1))

        if attn.norm_q is not None:
            img_query = attn.norm_q(img_query)
        if attn.norm_k is not None:
            img_key = attn.norm_k(img_key)
        if attn.norm_added_q is not None:
            txt_query = attn.norm_added_q(txt_query)
        if attn.norm_added_k is not None:
            txt_key = attn.norm_added_k(txt_key)

        if image_rotary_emb is not None:
            img_freqs, txt_freqs = image_rotary_emb
            img_query = apply_rotary_emb_ms(img_query, img_freqs, use_real=False)
            img_key = apply_rotary_emb_ms(img_key, img_freqs, use_real=False)
            txt_query = apply_rotary_emb_ms(txt_query, txt_freqs, use_real=False)
            txt_key = apply_rotary_emb_ms(txt_key, txt_freqs, use_real=False)

        joint_query = torch.cat([txt_query, img_query], dim=1)
        joint_key = torch.cat([txt_key, img_key], dim=1)
        joint_value = torch.cat([txt_value, img_value], dim=1)

        q_len = joint_query.shape[1]

        joint_hidden_states = self._flash_attention_forward(
            joint_query,
            joint_key,
            joint_value,
            attention_mask=None,
            query_length=q_len,
            dropout=0.0,
            use_sliding_windows=False,
            padding_type=padding_type,
        )
        joint_hidden_states = joint_hidden_states.flatten(2, 3)
        joint_hidden_states = joint_hidden_states.to(joint_query.dtype)

        txt_attn_output = joint_hidden_states[:, :seq_txt, :]
        img_attn_output = joint_hidden_states[:, seq_txt:, :]

        img_attn_output = attn.to_out[0](img_attn_output)
        if len(attn.to_out) > 1:
            img_attn_output = attn.to_out[1](img_attn_output)

        txt_attn_output = attn.to_add_out(txt_attn_output)

        return img_attn_output, txt_attn_output

    def __call__(
        self,
        attn: AttentionVE,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        encoder_hidden_states_mask: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        enc_token_mask: Optional[torch.Tensor] = None,
        padding_type: str = "pad",
        attn_mode: str = "default",
    ) -> torch.FloatTensor:

        if attn_mode == "default":
            output = self._call_ori(
                attn,
                hidden_states,
                encoder_hidden_states,
                encoder_hidden_states_mask,
                attention_mask,
                image_rotary_emb,
                padding_type,
            )
            return output
        elif attn_mode == "ve":
            output_ve = self._call_ve(
                attn,
                hidden_states,
                attention_mask,
                image_rotary_emb,
                enc_token_mask,
                padding_type,
            )
            return output_ve

    def _flash_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        query_length,
        dropout=0.0,
        softmax_scale=None,
        use_sliding_windows=False,
        padding_type="pad",
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`int`, *optional*):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
            use_sliding_windows (`bool`, *optional*):
                Whether to activate sliding window attention.
        """
        if padding_type == "pad":
            attn_output = self._flash_attention_forward_pad(
                query_states,
                key_states,
                value_states,
                attention_mask,
                query_length,
                dropout=dropout,
                softmax_scale=softmax_scale,
                use_sliding_windows=use_sliding_windows,
            )
        elif padding_type == "pack":
            attn_output = self._flash_attention_forward_pack(
                query_states,
                key_states,
                value_states,
                attention_mask,
                query_length,
                dropout=dropout,
                softmax_scale=softmax_scale,
                use_sliding_windows=use_sliding_windows,
            )
        else:
            raise ValueError(
                f"padding_type should be either `pad` or `pack`, got {padding_type}"
            )
        return attn_output

    def _flash_attention_forward_pad(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        query_length,
        dropout=0.0,
        softmax_scale=None,
        use_sliding_windows=False,
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`float`):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
            use_sliding_windows (`bool`, *optional*):
                Whether to activate sliding window attention.
        """
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            causal = self.is_causal and query_length != 1

        if use_sliding_windows and self.layer_idx >= self.config.max_window_layers:
            use_sliding_windows = False

        if attention_mask is not None:
            batch_size = query_states.shape[0]
            (
                query_states,
                key_states,
                value_states,
                indices_q,
                cu_seq_lens,
                max_seq_lens,
            ) = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            if not use_sliding_windows:
                attn_output_unpad = flash_attn_varlen_func(
                    query_states,
                    key_states,
                    value_states,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_in_batch_q,
                    max_seqlen_k=max_seqlen_in_batch_k,
                    dropout_p=dropout,
                    softmax_scale=softmax_scale,
                    causal=causal,
                )
            else:
                attn_output_unpad = flash_attn_varlen_func(
                    query_states,
                    key_states,
                    value_states,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_in_batch_q,
                    max_seqlen_k=max_seqlen_in_batch_k,
                    dropout_p=dropout,
                    softmax_scale=softmax_scale,
                    causal=causal,
                    window_size=(
                        self.config.sliding_window,
                        self.config.sliding_window,
                    ),
                )

            attn_output = pad_input(
                attn_output_unpad, indices_q, batch_size, query_length
            )
        else:
            if not use_sliding_windows:
                attn_output = flash_attn_func(
                    query_states,
                    key_states,
                    value_states,
                    dropout,
                    softmax_scale=softmax_scale,
                    causal=causal,
                )
            else:
                attn_output = flash_attn_func(
                    query_states,
                    key_states,
                    value_states,
                    dropout,
                    softmax_scale=softmax_scale,
                    causal=causal,
                    window_size=(
                        self.config.sliding_window,
                        self.config.sliding_window,
                    ),
                )

        return attn_output

    def _flash_attention_forward_pack(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        query_length,
        dropout=0.0,
        softmax_scale=None,
        use_sliding_windows=False,
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`int`, *optional*):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
            use_sliding_windows (`bool`, *optional*):
                Whether to activate sliding window attention.
        """
        assert query_states.size(0) == key_states.size(0) == value_states.size(0) == 1
        query_states = query_states.squeeze(0)
        key_states = key_states.squeeze(0)
        value_states = value_states.squeeze(0)
        cu_seqlens = attention_mask.squeeze(0).to(dtype=torch.int32)

        with torch.no_grad():
            max_seqlen = max(
                [
                    cu_seqlens[idx + 1] - cu_seqlens[idx]
                    for idx in range(cu_seqlens.size(0) - 1)
                ]
            ).item()

        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            causal = self.is_causal and query_length != 1

        if use_sliding_windows and self.layer_idx >= self.config.max_window_layers:
            use_sliding_windows = False

        if not use_sliding_windows:
            attn_output = flash_attn_varlen_func(
                q=query_states,
                k=key_states,
                v=value_states,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )
        else:
            attn_output = flash_attn_varlen_func(
                q=query_states,
                k=key_states,
                v=value_states,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=(self.config.sliding_window, self.config.sliding_window),
            )

        query_states = query_states.unsqueeze(0)
        key_states = key_states.unsqueeze(0)
        value_states = value_states.unsqueeze(0)
        return attn_output

    def _upad_input(
        self, query_layer, key_layer, value_layer, attention_mask, query_length
    ):
        batch_size, kv_seq_len, num_heads, head_dim = key_layer.shape

        if kv_seq_len != attention_mask.shape[-1]:
            attention_mask_num_tokens = attention_mask.shape[-1]
            attention_mask = attention_mask[:, attention_mask_num_tokens - kv_seq_len :]

        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k
        )

        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim),
                indices_k,
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(
                query_layer, attention_mask
            )

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


class InternVLUDoubleStreamTorchAttnProcessor:
    """
    Torch SDPA processor for InternVLU double-stream architecture.
    Structure matches InternVLUDoubleStreamFlashAttnProcessor, but does not use flash-attn.
    """

    _attention_backend = "torch_sdpa"

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "InternVLUDoubleStreamTorchAttnProcessor requires PyTorch 2.0+ "
                "(torch.nn.functional.scaled_dot_product_attention)."
            )
        self.is_causal = False

    def _call_ve(
        self,
        attn: AttentionVE,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        enc_token_mask: Optional[torch.Tensor] = None,
        padding_type: str = "pad",
    ):
        dtype = attn.to_v.weight.dtype

        enc_mask = enc_token_mask.unsqueeze(-1).to(dtype)
        img_mask = 1.0 - enc_mask

        q_enc = attn.add_q_proj(hidden_states).unflatten(-1, (attn.heads, -1))
        k_enc = attn.add_k_proj(hidden_states).unflatten(-1, (attn.heads, -1))
        v_enc = attn.add_v_proj(hidden_states)

        q_img = attn.to_q(hidden_states).unflatten(-1, (attn.heads, -1))
        k_img = attn.to_k(hidden_states).unflatten(-1, (attn.heads, -1))
        v_img = attn.to_v(hidden_states)

        bsz, q_len = q_enc.shape[:2]
        head_dim = attn.out_dim // attn.heads

        q_enc, gate_score_enc = torch.split(q_enc, [head_dim, head_dim], dim=-1)
        gate_score_enc = gate_score_enc.reshape(bsz, q_len, -1)
        q_img, gate_score_img = torch.split(q_img, [head_dim, head_dim], dim=-1)
        gate_score_img = gate_score_img.reshape(bsz, q_len, -1)

        q_enc = attn.norm_added_q(q_enc)
        k_enc = attn.norm_added_k(k_enc)
        q_img = attn.norm_q(q_img)
        k_img = attn.norm_k(k_img)

        joint_query = (
            q_enc * enc_mask.unsqueeze(-1) + q_img * img_mask.unsqueeze(-1)
        ).to(dtype)
        joint_key = (
            k_enc * enc_mask.unsqueeze(-1) + k_img * img_mask.unsqueeze(-1)
        ).to(dtype)
        joint_value = (
            (v_enc * enc_mask + v_img * img_mask)
            .unflatten(-1, (attn.heads, -1))
            .to(dtype)
        )

        if image_rotary_emb is not None:
            joint_query = apply_rotary_emb_ms(
                joint_query, image_rotary_emb, use_real=False
            )
            joint_key = apply_rotary_emb_ms(
                joint_key, image_rotary_emb, use_real=False
            )

        joint_hidden_states = self._torch_attention_forward(
            joint_query,
            joint_key,
            joint_value,
            attention_mask=attention_mask,
            query_length=q_len,
            dropout=0.0,
            padding_type=padding_type,
            training=attn.training,
        )

        joint_hidden_states = joint_hidden_states.reshape(bsz, q_len, -1).contiguous()

        enc_output = attn.to_add_out(joint_hidden_states)
        img_output = attn.to_out[0](joint_hidden_states)
        if len(attn.to_out) > 1:
            img_output = attn.to_out[1](img_output)

        attn_output = enc_output * enc_mask + img_output * img_mask
        gate_score = gate_score_enc * enc_mask + gate_score_img * img_mask
        attn_output = attn_output * torch.sigmoid(gate_score)

        return attn_output

    def _call_ori(
        self,
        attn: AttentionVE,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        encoder_hidden_states_mask: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        padding_type: str = "pad",
    ):
        if encoder_hidden_states is None:
            raise ValueError(
                "InternVLUDoubleStreamTorchAttnProcessor requires encoder_hidden_states (text stream)"
            )

        seq_txt = encoder_hidden_states.shape[1]

        img_query = attn.to_q(hidden_states).unflatten(-1, (attn.heads, -1))
        img_key = attn.to_k(hidden_states).unflatten(-1, (attn.heads, -1))
        img_value = attn.to_v(hidden_states).unflatten(-1, (attn.heads, -1))

        txt_query = attn.add_q_proj(encoder_hidden_states).unflatten(-1, (attn.heads, -1))
        txt_key = attn.add_k_proj(encoder_hidden_states).unflatten(-1, (attn.heads, -1))
        txt_value = attn.add_v_proj(encoder_hidden_states).unflatten(-1, (attn.heads, -1))

        if attn.norm_q is not None:
            img_query = attn.norm_q(img_query)
        if attn.norm_k is not None:
            img_key = attn.norm_k(img_key)
        if attn.norm_added_q is not None:
            txt_query = attn.norm_added_q(txt_query)
        if attn.norm_added_k is not None:
            txt_key = attn.norm_added_k(txt_key)

        if image_rotary_emb is not None:
            img_freqs, txt_freqs = image_rotary_emb
            img_query = apply_rotary_emb_ms(img_query, img_freqs, use_real=False)
            img_key = apply_rotary_emb_ms(img_key, img_freqs, use_real=False)
            txt_query = apply_rotary_emb_ms(txt_query, txt_freqs, use_real=False)
            txt_key = apply_rotary_emb_ms(txt_key, txt_freqs, use_real=False)

        joint_query = torch.cat([txt_query, img_query], dim=1)
        joint_key = torch.cat([txt_key, img_key], dim=1)
        joint_value = torch.cat([txt_value, img_value], dim=1)

        q_len = joint_query.shape[1]
        joint_hidden_states = self._torch_attention_forward(
            joint_query,
            joint_key,
            joint_value,
            attention_mask=None,
            query_length=q_len,
            dropout=0.0,
            padding_type=padding_type,
            training=attn.training,
        )
        joint_hidden_states = joint_hidden_states.flatten(2, 3).to(joint_query.dtype)

        txt_attn_output = joint_hidden_states[:, :seq_txt, :]
        img_attn_output = joint_hidden_states[:, seq_txt:, :]

        img_attn_output = attn.to_out[0](img_attn_output)
        if len(attn.to_out) > 1:
            img_attn_output = attn.to_out[1](img_attn_output)

        txt_attn_output = attn.to_add_out(txt_attn_output)

        return img_attn_output, txt_attn_output

    def __call__(
        self,
        attn: AttentionVE,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        encoder_hidden_states_mask: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        enc_token_mask: Optional[torch.Tensor] = None,
        padding_type: str = "pad",
        attn_mode: str = "default",
    ):
        if attn_mode == "default":
            return self._call_ori(
                attn,
                hidden_states,
                encoder_hidden_states,
                encoder_hidden_states_mask,
                attention_mask,
                image_rotary_emb,
                padding_type,
            )
        if attn_mode == "ve":
            return self._call_ve(
                attn,
                hidden_states,
                attention_mask,
                image_rotary_emb,
                enc_token_mask,
                padding_type,
            )
        raise ValueError(f"Unknown attn_mode: {attn_mode}")

    def _torch_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        query_length,
        dropout=0.0,
        padding_type="pad",
        training=False,
    ):
        if padding_type == "pad":
            return self._torch_attention_forward_pad(
                query_states=query_states,
                key_states=key_states,
                value_states=value_states,
                attention_mask=attention_mask,
                query_length=query_length,
                dropout=dropout,
                training=training,
            )
        if padding_type == "pack":
            return self._torch_attention_forward_pack(
                query_states=query_states,
                key_states=key_states,
                value_states=value_states,
                attention_mask=attention_mask,
                query_length=query_length,
                dropout=dropout,
                training=training,
            )
        raise ValueError(f"padding_type should be either `pad` or `pack`, got {padding_type}")

    def _torch_attention_forward_pad(
        self,
        query_states,   # [B, Sq, H, D]
        key_states,     # [B, Sk, H, D]
        value_states,   # [B, Sk, H, D]
        attention_mask, # [B, Sk] (1 valid / 0 pad) or None
        query_length,
        dropout=0.0,
        training=False,
    ):
        q = query_states.transpose(1, 2)  # [B, H, Sq, D]
        k = key_states.transpose(1, 2)    # [B, H, Sk, D]
        v = value_states.transpose(1, 2)  # [B, H, Sk, D]

        attn_mask_4d = None
        if attention_mask is not None:
            if attention_mask.shape[-1] != k.shape[-2]:
                attention_mask = attention_mask[:, -k.shape[-2]:]
            valid_k = attention_mask.to(torch.bool)  # [B, Sk]
            neg_inf = torch.finfo(q.dtype).min
            attn_mask_4d = torch.zeros(
                (q.shape[0], 1, q.shape[-2], k.shape[-2]),
                dtype=q.dtype,
                device=q.device,
            )
            attn_mask_4d = attn_mask_4d.masked_fill(~valid_k[:, None, None, :], neg_inf)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask_4d,
            dropout_p=dropout if training else 0.0,
            is_causal=self.is_causal,
        )  # [B, H, Sq, D]

        return out.transpose(1, 2).contiguous()  # [B, Sq, H, D]

    def _torch_attention_forward_pack(
        self,
        query_states,   # [1, S, H, D]
        key_states,     # [1, S, H, D]
        value_states,   # [1, S, H, D]
        attention_mask, # [1, n_seq+1] cumulative lengths
        query_length,
        dropout=0.0,
        training=False,
    ):
        assert query_states.size(0) == key_states.size(0) == value_states.size(0) == 1
        cu = attention_mask.squeeze(0).to(dtype=torch.long)  # [n_seq+1]

        outputs = []
        for i in range(cu.numel() - 1):
            s = cu[i].item()
            e = cu[i + 1].item()
            if e <= s:
                continue

            q_seg = query_states[:, s:e]  # [1, Li, H, D]
            k_seg = key_states[:, s:e]
            v_seg = value_states[:, s:e]

            out_seg = self._torch_attention_forward_pad(
                query_states=q_seg,
                key_states=k_seg,
                value_states=v_seg,
                attention_mask=None,
                query_length=e - s,
                dropout=dropout,
                training=training,
            )
            outputs.append(out_seg)

        if len(outputs) == 0:
            return torch.empty_like(query_states)

        return torch.cat(outputs, dim=1)


class AdaLayerNormContinuous(nn.Module):
    r"""
    Adaptive normalization layer with a norm layer (layer_norm or rms_norm).

    Args:
        embedding_dim (`int`): Embedding dimension to use during projection.
        conditioning_embedding_dim (`int`): Dimension of the input condition.
        elementwise_affine (`bool`, defaults to `True`):
            Boolean flag to denote if affine transformation should be applied.
        eps (`float`, defaults to 1e-5): Epsilon factor.
        bias (`bias`, defaults to `True`): Boolean flag to denote if bias should be use.
        norm_type (`str`, defaults to `"layer_norm"`):
            Normalization layer to use. Values supported: "layer_norm", "rms_norm".
    """

    def __init__(
        self,
        embedding_dim: int,
        conditioning_embedding_dim: int,
        elementwise_affine=True,
        eps=1e-5,
        bias=True,
        norm_type="layer_norm",
    ):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(
            conditioning_embedding_dim, embedding_dim * 2, bias=bias
        )
        if norm_type == "layer_norm":
            self.norm = FP32LayerNorm(embedding_dim, eps, elementwise_affine, bias)
        elif norm_type == "rms_norm":
            self.norm = RMSNorm(embedding_dim, eps, elementwise_affine)
        else:
            raise ValueError(f"unknown norm_type {norm_type}")

    def forward(
        self,
        x: torch.Tensor,
        conditioning_embedding: torch.Tensor,
        per_num_image_token: torch.Tensor = None,
    ) -> torch.Tensor:

        dtype = x.dtype
        emb = self.linear(self.silu(conditioning_embedding).to(x.dtype))
        if per_num_image_token is not None:
            emb = emb.float()
            x = x.float()
            emb = emb.repeat_interleave(per_num_image_token, dim=0)
            scale, shift = torch.chunk(emb, 2, dim=1)
            x = self.norm(x) * (1 + scale) + shift
            x = x.to(dtype=dtype)
        else:
            emb = emb.float()
            x = x.float()
            scale, shift = torch.chunk(emb, 2, dim=1)
            x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
            x = x.to(dtype=dtype)
        return x


class InternVLUTransformer2DModel(
    ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin, CacheMixin
):
    """
    Transformer backbone used for InternVLU diffusion generation.

    Args:
        patch_size (`int`, defaults to `2`):
            Patch size to turn the input data into small patches.
        in_channels (`int`, defaults to `64`):
            The number of channels in the input.
        out_channels (`int`, *optional*, defaults to `None`):
            The number of channels in the output. If not specified, it defaults to `in_channels`.
        num_layers (`int`, defaults to `60`):
            The number of layers of dual stream DiT blocks to use.
        attention_head_dim (`int`, defaults to `128`):
            The number of dimensions to use for each attention head.
        num_attention_heads (`int`, defaults to `24`):
            The number of attention heads to use.
        joint_attention_dim (`int`, defaults to `3584`):
            The number of dimensions to use for the joint attention (embedding/channel dimension of
            `encoder_hidden_states`).
        axes_dims_rope (`Tuple[int]`, defaults to `(16, 56, 56)`):
            The dimensions to use for the rotary positional embeddings.
        video_position_scale_factor (`int`, defaults to `1`):
            Scaling factor applied to video position indices.
        vlm_cond_position_scale_factor (`int`, defaults to `1`):
            Scaling factor applied to conditioning position indices.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = [
        "InternVLUTransformerBlock",
        "UnifiedMSRoPE",
        "QwenTimestepProjEmbeddings",
        "AdaLayerNormContinuous",
    ]
    _skip_layerwise_casting_patterns = ["pos_embed", "norm"]
    _repeated_blocks = ["InternVLUTransformerBlock"]

    @register_to_config
    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 64,
        out_channels: Optional[int] = 16,
        num_layers: int = 60,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 3584,
        axes_dims_rope: Tuple[int, int, int] = (16, 56, 56),
        video_position_scale_factor: int = 1,
        vlm_cond_position_scale_factor: int = 1,
    ):
        super().__init__()
        self.out_channels = out_channels or in_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        self.pos_embed = UnifiedMSRoPE(
            theta=10000, axes_dim=list(axes_dims_rope), scale_rope=True
        )

        self.time_text_embed = QwenTimestepProjEmbeddings(embedding_dim=self.inner_dim)

        self.txt_norm = RMSNorm(joint_attention_dim, eps=1e-6)

        self.img_in = nn.Linear(in_channels, self.inner_dim)
        self.txt_in = nn.Linear(joint_attention_dim, self.inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                InternVLUTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm_out = AdaLayerNormContinuous(
            self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6
        )
        self.proj_out = nn.Linear(
            self.inner_dim, patch_size * patch_size * self.out_channels, bias=True
        )

        self.gradient_checkpointing = False

        self.patch_size = patch_size
        self.video_position_scale_factor = video_position_scale_factor
        self.vlm_cond_position_scale_factor = vlm_cond_position_scale_factor

    def initialize_weights(self):
        self.apply(_basic_init)

    def _prepare_hidden_inputs_anyres(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        image_fhw_cond: Union[torch.Tensor, List[torch.Tensor]] = None,
        conditional_input: Optional[List[torch.Tensor]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        encoder_image_token_mask: Optional[torch.Tensor] = None,
        padding_type: str = "pad",
        image_grid_thw_gen: torch.Tensor = None,
        image_grid_thw_gen_cond: torch.Tensor = None,
    ):
        batch_size, num_channels, height, width = hidden_states.shape
        p = self.patch_size

        post_image_grid_thw_gen = image_grid_thw_gen.clone()
        post_image_grid_thw_gen[:, 1] = post_image_grid_thw_gen[:, 1] // p
        post_image_grid_thw_gen[:, 2] = post_image_grid_thw_gen[:, 2] // p
        min_post_patch_height = post_image_grid_thw_gen[:, 1].min()
        min_post_patch_width = post_image_grid_thw_gen[:, 2].min()
        num_post_image_token_gen = post_image_grid_thw_gen.prod(dim=-1)

        hidden_states_pack = []
        for b in range(batch_size):
            _hs = hidden_states[
                b, :, : image_grid_thw_gen[b, 1], : image_grid_thw_gen[b, 2]
            ]
            _hs = rearrange(_hs, "c (h ph) (w pw) -> (h w) (c ph pw)", ph=p, pw=p)
            hidden_states_pack.append(_hs)
        hidden_states_pack = torch.cat(hidden_states_pack)
        hidden_states = self.img_in(hidden_states_pack)

        num_conds_max = 0
        add_vae_condition = conditional_input is not None
        if add_vae_condition:
            num_conds = [cond.shape[0] for cond in conditional_input]
            num_conds_max = max(num_conds)
            num_conds = torch.tensor(num_conds, device=hidden_states.device)
            assert (
                num_conds_max > 0
            ), "At least one conditional input is required for VE model."
            assert num_conds_max < int(
                min(min_post_patch_height, min_post_patch_width)
                // 2
                / self.video_position_scale_factor
            ), (
                f"num_conds_max ({num_conds_max}) must be less than "
                f"{int(min(min_post_patch_height, min_post_patch_width) // 2 / self.video_position_scale_factor)}"
            )
            cond_hidden_states = torch.cat(conditional_input, dim=0)
            image_grid_thw_gen_cond = torch.cat(image_grid_thw_gen_cond, dim=0)

            post_image_grid_thw_gen_cond = image_grid_thw_gen_cond.clone()
            post_image_grid_thw_gen_cond[:, 1] = post_image_grid_thw_gen_cond[:, 1] // p
            post_image_grid_thw_gen_cond[:, 2] = post_image_grid_thw_gen_cond[:, 2] // p
            num_post_image_token_gen_cond = post_image_grid_thw_gen_cond.prod(dim=-1)

            cond_hidden_states_pack = []
            for b in range(cond_hidden_states.shape[0]):
                _hs = cond_hidden_states[
                    b,
                    :,
                    : image_grid_thw_gen_cond[b, 1],
                    : image_grid_thw_gen_cond[b, 2],
                ]
                _hs = rearrange(_hs, "c (h ph) (w pw) -> (h w) (c ph pw)", ph=p, pw=p)
                cond_hidden_states_pack.append(_hs)
            cond_hidden_states_pack = torch.cat(cond_hidden_states_pack)
            cond_hidden_states = self.img_in(cond_hidden_states_pack)

        assert (
            encoder_attention_mask is not None
        ), "encoder_attention_mask cannot be None when using `pack` padding_type"
        txt_seq_lens = encoder_attention_mask.sum(dim=1)
        txt_seq_lens = txt_seq_lens.tolist()

        hidden_states_pack = []
        attention_mask_pack = [0]
        enc_token_mask_pack = []
        cond_token_mask_pack = []

        img_shapes = []
        img_scale_factors = []
        input_token_mask_batch = []

        (
            cur_num_cond,
            cur_num_post_image_token_gen_cond,
            cur_num_post_image_token_gen,
        ) = (0, 0, 0)
        for b in range(batch_size):
            num_image_token = post_image_grid_thw_gen[b].prod()
            img_scale_factors_b = [[self.video_position_scale_factor]]
            img_shape_b = [[post_image_grid_thw_gen[b].tolist()]]
            if add_vae_condition:
                num_image_token += (
                    post_image_grid_thw_gen_cond[
                        cur_num_cond : cur_num_cond + num_conds[b]
                    ]
                    .prod(dim=-1)
                    .sum()
                )
                img_shape_b[0].extend(
                    post_image_grid_thw_gen_cond[
                        cur_num_cond : cur_num_cond + num_conds[b]
                    ].tolist()
                )
                img_scale_factors_b[0].extend(
                    [self.video_position_scale_factor] * num_conds[b]
                )

            cond_img_shape = image_fhw_cond[b].tolist()
            cond_img_shape = [[c] for c in cond_img_shape]
            img_shape_b.extend(cond_img_shape)
            img_scale_factors_b += [[self.vlm_cond_position_scale_factor]] * len(
                cond_img_shape
            )
            img_shapes.append(img_shape_b)
            img_scale_factors.append(img_scale_factors_b)
            input_token_mask = torch.cat(
                [
                    torch.ones(
                        (num_image_token,),
                        device=encoder_image_token_mask.device,
                        dtype=encoder_image_token_mask.dtype,
                    ),
                    encoder_image_token_mask[b, : txt_seq_lens[b]],
                ],
                dim=0,
            )
            input_token_mask_batch.append(input_token_mask)

            hidden_states_b = hidden_states[
                cur_num_post_image_token_gen : cur_num_post_image_token_gen
                + num_post_image_token_gen[b],
                :,
            ]
            hidden_states_pack.append(hidden_states_b)
            enc_token_mask_pack.append(
                torch.zeros(
                    len(hidden_states_b), dtype=torch.bool, device=hidden_states.device
                )
            )
            cond_token_mask_pack.append(
                torch.zeros(
                    len(hidden_states_b), dtype=torch.bool, device=hidden_states.device
                )
            )
            cur_num_post_image_token_gen += num_post_image_token_gen[b]

            if add_vae_condition and num_conds[b] > 0:
                cond_hidden_states_b = cond_hidden_states[
                    cur_num_post_image_token_gen_cond : cur_num_post_image_token_gen_cond
                    + num_post_image_token_gen_cond[
                        cur_num_cond : cur_num_cond + num_conds[b]
                    ]
                ]
                hidden_states_pack.append(cond_hidden_states_b)
                enc_token_mask_pack.append(
                    torch.zeros(
                        len(cond_hidden_states_b),
                        dtype=torch.bool,
                        device=hidden_states.device,
                    )
                )
                cond_token_mask_pack.append(
                    torch.ones(
                        len(cond_hidden_states_b),
                        dtype=torch.bool,
                        device=hidden_states.device,
                    )
                )

                cur_num_post_image_token_gen_cond += num_post_image_token_gen_cond[
                    cur_num_cond : cur_num_cond + num_conds[b]
                ]
                cur_num_cond += num_conds[b]

            encoder_hidden_states_b = encoder_hidden_states[b, : txt_seq_lens[b], :]
            hidden_states_pack.append(encoder_hidden_states_b)
            enc_token_mask_pack.append(
                torch.ones(
                    txt_seq_lens[b], dtype=torch.bool, device=hidden_states.device
                )
            )
            cond_token_mask_pack.append(
                torch.ones(
                    txt_seq_lens[b], dtype=torch.bool, device=hidden_states.device
                )
            )

            cu_seq_len = (
                attention_mask_pack[-1] + len(hidden_states_b) + txt_seq_lens[b]
            )
            if add_vae_condition and num_conds[b] > 0:
                cu_seq_len += len(cond_hidden_states_b)
            attention_mask_pack.append(cu_seq_len)

        position_3ds = create_position_ids_3d_v3(
            img_shapes,
            input_token_mask_batch,
            video_scale_factor=img_scale_factors,
            device=hidden_states.device,
        )

        image_rotary_emb = self.pos_embed(position_3ds, device=hidden_states.device)

        hidden_states = torch.cat(hidden_states_pack, dim=0)[None]
        attention_mask = torch.tensor(attention_mask_pack, device=hidden_states.device)[
            None
        ]
        enc_token_mask = torch.cat(enc_token_mask_pack, dim=0)[None]
        cond_token_mask = torch.cat(cond_token_mask_pack, dim=0)[None]

        return (
            hidden_states,
            attention_mask,
            image_rotary_emb,
            enc_token_mask,
            cond_token_mask,
        )

    def _prepare_hidden_inputs(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        image_fhw_cond: Union[torch.Tensor, List[torch.Tensor]] = None,
        conditional_input: Optional[List[torch.Tensor]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        encoder_image_token_mask: Optional[torch.Tensor] = None,
        padding_type: str = "pad",
    ):
        batch_size, num_channels, height, width = hidden_states.shape
        p = self.patch_size
        post_patch_height, post_patch_width = height // p, width // p
        hidden_states = rearrange(
            hidden_states, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=p, pw=p
        )

        hidden_states = self.img_in(hidden_states)

        num_conds_max = 0
        add_vae_condition = conditional_input is not None
        if add_vae_condition:
            num_conds = [cond.shape[0] for cond in conditional_input]
            num_conds_max = max(num_conds)
            num_conds = torch.tensor(num_conds, device=hidden_states.device)
            assert (
                num_conds_max > 0
            ), "At least one conditional input is required for VE model."
            assert num_conds_max < int(
                min(post_patch_height, post_patch_width)
                // 2
                / self.video_position_scale_factor
            ), (
                f"num_conds_max ({num_conds_max}) must be less than "
                f"{int(min(post_patch_height, post_patch_width) // 2 / self.video_position_scale_factor)}"
            )

            cond_hidden_states = torch.cat(conditional_input, dim=0)
            _, _, cond_height, cond_width = cond_hidden_states.shape
            post_cond_height, post_cond_width = cond_height // p, cond_width // p
            assert (
                post_patch_height == post_cond_height
                and post_patch_width == post_cond_width
            ), (
                f"post_patch_height ({post_patch_height}) must be equal to post_cond_height ({post_cond_height}), "
                f"post_patch_width ({post_patch_width}) must be equal to post_cond_width ({post_cond_width})"
            )

            cond_hidden_states = rearrange(
                cond_hidden_states, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=p, pw=p
            )
            cond_hidden_states = self.img_in(cond_hidden_states)
            cond_hidden_states_pad = torch.zeros(
                hidden_states.shape[0],
                hidden_states.shape[1] * num_conds_max,
                hidden_states.shape[-1],
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            for b in range(batch_size):
                cond_hidden_states_pad[b, : num_conds[b] * hidden_states.shape[1]] = (
                    cond_hidden_states[
                        sum(num_conds[:b]) : sum(num_conds[: b + 1])
                    ].flatten(0, 1)
                )

        if padding_type == "pad":
            txt_seq_lens = (
                torch.ones_like(encoder_attention_mask.sum(dim=1))
                * encoder_attention_mask.shape[-1]
            )
            txt_seq_lens = txt_seq_lens.tolist()

            img_shapes = []
            img_scale_factors = []
            for b in range(batch_size):
                img_scale_factors_b = [self.video_position_scale_factor]
                img_shape_b = [(num_conds_max + 1, post_patch_height, post_patch_width)]
                cond_img_shape = image_fhw_cond[b].cpu().tolist()
                img_shape_b.extend(cond_img_shape)
                img_scale_factors_b += [self.vlm_cond_position_scale_factor] * len(
                    cond_img_shape
                )
                img_shapes.append(img_shape_b)
                img_scale_factors.append(img_scale_factors_b)
            assert len(img_scale_factors) == len(
                img_shapes
            ), f"len(img_scale_factors) ({len(img_scale_factors)}) must be equal to len(img_shapes): {len(img_shapes)}"

            num_image_token = post_patch_height * post_patch_width * (num_conds_max + 1)
            input_token_mask = torch.cat(
                [
                    torch.ones(
                        (batch_size, num_image_token),
                        device=encoder_image_token_mask.device,
                        dtype=encoder_image_token_mask.dtype,
                    ),
                    encoder_image_token_mask,
                ],
                dim=1,
            )

            position_3ds = create_position_ids_3d_v2(
                img_shapes,
                input_token_mask,
                video_scale_factor=img_scale_factors,
                device=hidden_states.device,
            )
            image_rotary_emb = self.pos_embed(position_3ds, device=hidden_states.device)
            image_rotary_emb = image_rotary_emb.reshape(
                batch_size, -1, image_rotary_emb.shape[-1]
            )

            encoder_attention_mask = encoder_attention_mask[:, : max(txt_seq_lens)]
            encoder_hidden_states = encoder_hidden_states[:, : max(txt_seq_lens)]
            seq_txt = encoder_hidden_states.shape[1]
            seq_img = hidden_states.shape[1]
            attention_mask = torch.ones_like(hidden_states[:, :, 0])
            if add_vae_condition:
                hidden_states = torch.cat(
                    [hidden_states, cond_hidden_states_pad], dim=1
                )
                attention_mask_cond = torch.zeros_like(cond_hidden_states_pad[:, :, 0])
                positions = torch.arange(
                    attention_mask_cond.shape[1], device=attention_mask_cond.device
                )[None]
                mask = positions < num_conds[:, None] * seq_img
                attention_mask_cond[mask] = 1
                attention_mask = torch.cat([attention_mask, attention_mask_cond], dim=1)

            hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)
            attention_mask = torch.cat([attention_mask, encoder_attention_mask], dim=1)
            enc_token_mask = torch.zeros(
                hidden_states.shape[0],
                hidden_states.shape[1],
                device=encoder_hidden_states.device,
            )
            enc_token_mask[:, -seq_txt:] = 1
            enc_token_mask = enc_token_mask.bool()

            return (
                hidden_states,
                attention_mask,
                image_rotary_emb,
                enc_token_mask,
                seq_img,
                seq_txt,
            )

        elif padding_type == "pack":
            assert (
                encoder_attention_mask is not None
            ), "encoder_attention_mask cannot be None when using `pack` padding_type"
            txt_seq_lens = encoder_attention_mask.sum(dim=1)
            txt_seq_lens = txt_seq_lens.tolist()

            hidden_states_pack = []
            attention_mask_pack = [0]
            enc_token_mask_pack = []
            cond_token_mask_pack = []

            img_shapes = []
            img_scale_factors = []
            input_token_mask_batch = []

            for b in range(batch_size):
                img_scale_factors_b = []
                img_shape_b = (
                    [(num_conds[b] + 1, post_patch_height, post_patch_width)]
                    if add_vae_condition
                    else [(1, post_patch_height, post_patch_width)]
                )
                img_scale_factors_b.append(self.video_position_scale_factor)

                cond_img_shape = image_fhw_cond[b].cpu().tolist()
                img_shape_b.extend(cond_img_shape)
                img_scale_factors_b += [self.vlm_cond_position_scale_factor] * len(
                    cond_img_shape
                )
                img_shapes.append(img_shape_b)
                img_scale_factors.append(img_scale_factors_b)
                num_image_token = (
                    post_patch_height
                    * post_patch_width
                    * (num_conds[b] + 1 if add_vae_condition else 1)
                )
                input_token_mask = torch.cat(
                    [
                        torch.ones(
                            (num_image_token,),
                            device=encoder_image_token_mask.device,
                            dtype=encoder_image_token_mask.dtype,
                        ),
                        encoder_image_token_mask[b, : txt_seq_lens[b]],
                    ],
                    dim=0,
                )
                input_token_mask_batch.append(input_token_mask)

                hidden_states_b = hidden_states[b, :, :]
                hidden_states_pack.append(hidden_states_b)
                enc_token_mask_pack.append(
                    torch.zeros(
                        len(hidden_states_b),
                        dtype=torch.bool,
                        device=hidden_states.device,
                    )
                )
                cond_token_mask_pack.append(
                    torch.zeros(
                        len(hidden_states_b),
                        dtype=torch.bool,
                        device=hidden_states.device,
                    )
                )

                if add_vae_condition and num_conds[b] > 0:
                    cond_hidden_states_b = cond_hidden_states_pad[
                        b, : num_conds[b] * hidden_states.shape[1], :
                    ]
                    hidden_states_pack.append(cond_hidden_states_b)
                    enc_token_mask_pack.append(
                        torch.zeros(
                            len(cond_hidden_states_b),
                            dtype=torch.bool,
                            device=hidden_states.device,
                        )
                    )
                    cond_token_mask_pack.append(
                        torch.ones(
                            len(cond_hidden_states_b),
                            dtype=torch.bool,
                            device=hidden_states.device,
                        )
                    )

                encoder_hidden_states_b = encoder_hidden_states[b, : txt_seq_lens[b], :]
                hidden_states_pack.append(encoder_hidden_states_b)
                enc_token_mask_pack.append(
                    torch.ones(
                        txt_seq_lens[b], dtype=torch.bool, device=hidden_states.device
                    )
                )
                cond_token_mask_pack.append(
                    torch.ones(
                        txt_seq_lens[b], dtype=torch.bool, device=hidden_states.device
                    )
                )

                cu_seq_len = (
                    attention_mask_pack[-1] + len(hidden_states_b) + txt_seq_lens[b]
                )
                if add_vae_condition and num_conds[b] > 0:
                    cu_seq_len += len(cond_hidden_states_b)
                attention_mask_pack.append(cu_seq_len)

            position_3ds = create_position_ids_3d_v2(
                img_shapes,
                input_token_mask_batch,
                video_scale_factor=img_scale_factors,
                device=hidden_states.device,
            )

            image_rotary_emb = self.pos_embed(position_3ds, device=hidden_states.device)

            hidden_states = torch.cat(hidden_states_pack, dim=0)[None]
            attention_mask = torch.tensor(
                attention_mask_pack, device=hidden_states.device
            )[None]
            enc_token_mask = torch.cat(enc_token_mask_pack, dim=0)[None]
            cond_token_mask = torch.cat(cond_token_mask_pack, dim=0)[None]

            return (
                hidden_states,
                attention_mask,
                image_rotary_emb,
                enc_token_mask,
                cond_token_mask,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        conditional_input: Optional[List[torch.Tensor]] = None,
        guidance: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        encoder_image_token_mask: Optional[torch.Tensor] = None,
        image_fhw_cond: Union[torch.Tensor, List[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        padding_type: str = "pad",
        image_grid_thw_gen: torch.Tensor = None,
        image_grid_thw_gen_cond: torch.Tensor = None,
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:
        """
        The [`InternVLUTransformer2DModel`] forward method.

        Args:
            hidden_states: [B, C, H, W]
                Input `hidden_states`.
            encoder_hidden_states (`torch.Tensor` of shape `(batch_size, text_sequence_length, joint_attention_dim)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            encoder_attention_mask (`torch.Tensor` of shape `(batch_size, text_sequence_length)`):
                Mask of the input conditions.
            timestep ( `torch.LongTensor`):
                Used to indicate denoising step.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).  # noqa: E501
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)
        else:
            if (
                attention_kwargs is not None
                and attention_kwargs.get("scale", None) is not None
            ):
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )

        timestep = timestep.to(hidden_states.dtype)
        encoder_hidden_states = self.txt_norm(encoder_hidden_states)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)

        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000

        temb = (
            self.time_text_embed(timestep, hidden_states)
            if guidance is None
            else self.time_text_embed(timestep, guidance, hidden_states)
        )

        if not torch.is_grad_enabled():
            padding_type = "pad"

        assert (
            encoder_image_token_mask is not None
        ), "encoder_image_token_mask cannot be None"

        anyres_mode_enabled = image_grid_thw_gen is not None
        batch_size, num_channels, height, width = hidden_states.shape
        p = self.patch_size
        if anyres_mode_enabled:
            padding_type = "pack"
            (
                hidden_states,
                attention_mask,
                image_rotary_emb,
                enc_token_mask,
                cond_token_mask,
            ) = self._prepare_hidden_inputs_anyres(
                hidden_states,
                encoder_hidden_states,
                image_fhw_cond=image_fhw_cond,
                conditional_input=conditional_input,
                encoder_attention_mask=encoder_attention_mask,
                encoder_image_token_mask=encoder_image_token_mask,
                padding_type=padding_type,
                image_grid_thw_gen=image_grid_thw_gen,
                image_grid_thw_gen_cond=image_grid_thw_gen_cond,
            )
        else:
            post_patch_height, post_patch_width = height // p, width // p
            add_vae_condition = conditional_input is not None
            hidden_inputs = self._prepare_hidden_inputs(
                hidden_states,
                encoder_hidden_states,
                image_fhw_cond=image_fhw_cond,
                conditional_input=conditional_input,
                encoder_attention_mask=encoder_attention_mask,
                encoder_image_token_mask=encoder_image_token_mask,
                padding_type=padding_type,
            )
            if padding_type == "pad":
                (
                    hidden_states,
                    attention_mask,
                    image_rotary_emb,
                    enc_token_mask,
                    seq_img,
                    seq_txt,
                ) = hidden_inputs
            elif padding_type == "pack":
                (
                    hidden_states,
                    attention_mask,
                    image_rotary_emb,
                    enc_token_mask,
                    cond_token_mask,
                ) = hidden_inputs

        import os

        enable_fp32_mode = int(os.environ.get("FP32_MODE_ENABLED", "0")) == 1
        if enable_fp32_mode:
            self.transformer_blocks[-1].to(dtype=torch.float)
            self.norm_out.to(dtype=torch.float)
            self.proj_out.to(dtype=torch.float)

        for index_block, block in enumerate(self.transformer_blocks):
            from contextlib import nullcontext
            from torch import autocast

            if enable_fp32_mode and index_block == len(self.transformer_blocks) - 1:
                block.to(dtype=torch.float)
                hidden_states = hidden_states.float()
                temb = temb.float()
                if padding_type == "pad":
                    attention_mask = attention_mask.float()

            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    temb,
                    attention_mask,
                    image_rotary_emb,
                    enc_token_mask,
                    padding_type,
                )
            else:
                hidden_states = block(
                    hidden_states=hidden_states,
                    temb=temb,
                    attention_mask=attention_mask,
                    image_rotary_emb=image_rotary_emb,
                    enc_token_mask=enc_token_mask,
                    padding_type=padding_type,
                    joint_attention_kwargs=attention_kwargs,
                )

        if anyres_mode_enabled:
            hidden_states = hidden_states[~cond_token_mask]
            per_num_image_token = image_grid_thw_gen.prod(dim=-1) // (p**2)
            hidden_states = self.norm_out(
                hidden_states, temb, per_num_image_token=per_num_image_token
            )
        else:
            if padding_type == "pad":
                hidden_states, encoder_hidden_states = (
                    hidden_states[:, :-seq_txt, :],
                    hidden_states[:, -seq_txt:, :],
                )
                if add_vae_condition:
                    hidden_states, cond_hidden_states = (
                        hidden_states[:, :seq_img, :],
                        hidden_states[:, seq_img:, :],
                    )
            elif padding_type == "pack":
                hidden_states = hidden_states[~cond_token_mask]
                hidden_states = hidden_states.reshape(
                    batch_size, -1, hidden_states.shape[-1]
                )
            hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if anyres_mode_enabled:
            output_pad = torch.zeros(
                batch_size,
                num_channels,
                height,
                width,
                device=output.device,
                dtype=output.dtype,
            )
            cum_image_token = 0
            for b in range(batch_size):
                cur_output = output[
                    cum_image_token : cum_image_token + per_num_image_token[b]
                ].view(image_grid_thw_gen[b, 1] // p, image_grid_thw_gen[b, 2] // p, -1)
                cur_output = rearrange(
                    cur_output, "h w (c ph pw) -> c (h ph) (w pw)", ph=p, pw=p
                )
                output_pad[
                    b, :, : image_grid_thw_gen[b, 1], : image_grid_thw_gen[b, 2]
                ] = cur_output
                cum_image_token += per_num_image_token[b]
            output = output_pad
        else:
            output = rearrange(
                output,
                "b (h w) (c ph pw) -> b c (h ph) (w pw)",
                h=post_patch_height,
                w=post_patch_width,
                ph=p,
                pw=p,
            )

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
