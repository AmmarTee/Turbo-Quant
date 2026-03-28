"""
TurboQuant KV Cache

Integrates TurboQuant compression into the transformer KV cache.
Keys are quantized using TurboQuant_prod (unbiased inner product) and
values use standard per-group asymmetric quantization.

Memory reduction:
    - Keys: 16-bit -> 3-bit (5.3x reduction)
    - Values: 16-bit -> 2-bit with group quantization
    - Overall: >4.5x compression at near-zero quality loss
"""

import math
import torch
import torch.nn as nn
from typing import Optional, Tuple, List

from .turbo_quant import TurboQuantProd, TurboQuantMSE
from .qjl import QJL


class ValueQuantizer:
    """Standard per-group asymmetric quantization for values.

    Values don't need unbiased inner product estimation (they're multiplied
    by attention weights, not used for attention score computation), so
    simple min-max quantization suffices.
    """

    def __init__(self, bit_width: int = 2, group_size: int = 32):
        self.bit_width = bit_width
        self.group_size = group_size
        self.num_levels = 2 ** bit_width

    def quantize(self, values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize values with per-group asymmetric quantization.

        Args:
            values: (batch, heads, seq, dim)

        Returns:
            (quantized_uint8, scales, zero_points)
        """
        b, h, s, d = values.shape
        # Reshape into groups
        num_groups = (d + self.group_size - 1) // self.group_size
        pad = num_groups * self.group_size - d
        if pad > 0:
            values = torch.nn.functional.pad(values, (0, pad))

        grouped = values.reshape(b, h, s, num_groups, self.group_size)

        # Per-group min/max
        vmin = grouped.min(dim=-1, keepdim=True).values
        vmax = grouped.max(dim=-1, keepdim=True).values

        # Compute scale and zero point
        scale = (vmax - vmin) / (self.num_levels - 1)
        scale = scale.clamp(min=1e-8)
        zero_point = vmin

        # Quantize
        quantized = ((grouped - zero_point) / scale).round().clamp(0, self.num_levels - 1)
        quantized = quantized.to(torch.uint8)

        return quantized, scale.squeeze(-1), zero_point.squeeze(-1)

    def dequantize(
        self,
        quantized: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        original_dim: int,
    ) -> torch.Tensor:
        """Dequantize values.

        Args:
            quantized: uint8 tensor (batch, heads, seq, num_groups, group_size)
            scale: (batch, heads, seq, num_groups)
            zero_point: (batch, heads, seq, num_groups)
            original_dim: original dimension before padding

        Returns:
            Dequantized values (batch, heads, seq, original_dim)
        """
        result = quantized.float() * scale.unsqueeze(-1) + zero_point.unsqueeze(-1)
        result = result.reshape(*result.shape[:3], -1)
        return result[..., :original_dim]


class TurboQuantKVCache:
    """KV Cache with TurboQuant compression.

    Maintains a compressed key-value cache for efficient long-context inference.
    Keys use TurboQuant_prod for unbiased attention score estimation.
    Values use standard group quantization.

    Architecture:
        - Full-precision buffer for recent tokens (buffer_size)
        - Quantized storage for older tokens
        - Streaming quantization during autoregressive generation
    """

    def __init__(
        self,
        head_dim: int,
        num_heads: int,
        num_kv_heads: int,
        key_bit_width: int = 3,
        value_bit_width: int = 2,
        value_group_size: int = 32,
        buffer_size: int = 128,
        device: str = "cuda",
    ):
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_heads // num_kv_heads
        self.key_bit_width = key_bit_width
        self.value_bit_width = value_bit_width
        self.buffer_size = buffer_size
        self.device = device

        # Key quantizer: TurboQuant_prod per head
        self.key_quantizer = TurboQuantProd(head_dim, key_bit_width, device=device)

        # Value quantizer: standard group quantization
        self.value_quantizer = ValueQuantizer(value_bit_width, value_group_size)

        # State
        self.seq_len = 0
        self._key_quant_state = None       # Quantized keys
        self._value_quant = None            # Quantized values
        self._value_scale = None
        self._value_zp = None
        self._key_buffer = None             # Full-precision key buffer (recent)
        self._value_buffer = None           # Full-precision value buffer (recent)

    @property
    def effective_bits(self) -> float:
        return (self.key_bit_width + self.value_bit_width) / 2

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """Repeat KV heads for GQA/MQA. Handles 4D and 5D tensors."""
        if self.num_kv_groups == 1:
            return x
        b, h = x.shape[:2]
        rest = x.shape[2:]
        x = x.unsqueeze(2).expand(b, h, self.num_kv_groups, *rest)
        return x.reshape(b, h * self.num_kv_groups, *rest)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> None:
        """Add new key-value pairs to the cache.

        On first call (prefill), processes the full prompt.
        On subsequent calls (decode), appends one token at a time.

        Args:
            key_states: (batch, kv_heads, seq, head_dim)
            value_states: (batch, kv_heads, seq, head_dim)
        """
        new_len = key_states.shape[2]
        self.seq_len += new_len

        if self._key_buffer is None:
            # First call - initialize buffer
            self._key_buffer = key_states
            self._value_buffer = value_states
        else:
            # Append to buffer
            self._key_buffer = torch.cat([self._key_buffer, key_states], dim=2)
            self._value_buffer = torch.cat([self._value_buffer, value_states], dim=2)

        # If buffer exceeds threshold, quantize older tokens
        if self._key_buffer.shape[2] > self.buffer_size:
            to_quantize_len = self._key_buffer.shape[2] - self.buffer_size

            keys_to_quant = self._key_buffer[:, :, :to_quantize_len, :]
            vals_to_quant = self._value_buffer[:, :, :to_quantize_len, :]

            # Quantize keys with TurboQuant
            new_key_state = self.key_quantizer.quantize(keys_to_quant)

            # Quantize values with group quantization
            v_quant, v_scale, v_zp = self.value_quantizer.quantize(vals_to_quant)

            if self._key_quant_state is None:
                self._key_quant_state = new_key_state
                self._value_quant = v_quant
                self._value_scale = v_scale
                self._value_zp = v_zp
            else:
                # Concatenate with existing quantized data
                for k in self._key_quant_state:
                    if isinstance(self._key_quant_state[k], torch.Tensor):
                        self._key_quant_state[k] = torch.cat(
                            [self._key_quant_state[k], new_key_state[k]], dim=2
                        )
                self._value_quant = torch.cat([self._value_quant, v_quant], dim=2)
                self._value_scale = torch.cat([self._value_scale, v_scale], dim=2)
                self._value_zp = torch.cat([self._value_zp, v_zp], dim=2)

            # Keep only the buffer
            self._key_buffer = self._key_buffer[:, :, to_quantize_len:, :]
            self._value_buffer = self._value_buffer[:, :, to_quantize_len:, :]

    def get_attention_output(
        self,
        query_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute attention output using the compressed KV cache.

        Combines scores from:
        1. Quantized keys (TurboQuant inner product estimation)
        2. Full-precision buffer keys (standard dot product)

        Then applies attention weights to combined value representation.

        Args:
            query_states: (batch, num_heads, seq_q, head_dim)
            attention_mask: Optional mask (batch, 1, seq_q, seq_total)

        Returns:
            Attention output (batch, num_heads, seq_q, head_dim)
        """
        b, h, sq, d = query_states.shape

        # Expand KV for GQA if needed
        key_buffer = self._repeat_kv(self._key_buffer)
        value_buffer = self._repeat_kv(self._value_buffer)
        buffer_len = key_buffer.shape[2]

        if self._key_quant_state is not None:
            # Compute attention scores for quantized portion
            # We need to expand quant state for GQA too
            scores_quant = self.key_quantizer.estimate_attention_score(
                query_states, self._expand_quant_state_for_gqa(self._key_quant_state)
            )
            quant_len = scores_quant.shape[-1]

            # Compute attention scores for buffer portion
            scores_buffer = torch.matmul(query_states, key_buffer.transpose(-2, -1))

            # Concatenate scores: [quantized | buffer]
            attn_weights = torch.cat([scores_quant, scores_buffer], dim=-1)
            attn_weights = attn_weights / math.sqrt(d)

            # Apply attention mask
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask

            attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
            attn_weights = attn_weights.to(query_states.dtype)

            # Compute weighted sum of values
            # Quantized values
            value_quant_expanded = self._repeat_kv_quant(
                self._value_quant, self._value_scale, self._value_zp
            )
            v_dequant = self.value_quantizer.dequantize(
                value_quant_expanded[0],
                value_quant_expanded[1],
                value_quant_expanded[2],
                d,
            )

            attn_out_quant = torch.matmul(attn_weights[..., :quant_len], v_dequant)
            attn_out_buffer = torch.matmul(attn_weights[..., quant_len:], value_buffer)

            return attn_out_quant + attn_out_buffer
        else:
            # All tokens are in the buffer (short context)
            scores = torch.matmul(query_states, key_buffer.transpose(-2, -1))
            scores = scores / math.sqrt(d)

            if attention_mask is not None:
                scores = scores + attention_mask

            attn_weights = torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float32)
            attn_weights = attn_weights.to(query_states.dtype)

            return torch.matmul(attn_weights, value_buffer)

    def _expand_quant_state_for_gqa(self, quant_state: dict) -> dict:
        """Expand quantized key state for grouped query attention."""
        if self.num_kv_groups == 1:
            return quant_state

        expanded = {}
        for k, v in quant_state.items():
            if isinstance(v, torch.Tensor) and v.dim() >= 3:
                b, h = v.shape[:2]
                rest = v.shape[2:]
                v = v.unsqueeze(2).expand(b, h, self.num_kv_groups, *rest)
                expanded[k] = v.reshape(b, h * self.num_kv_groups, *rest)
            else:
                expanded[k] = v
        return expanded

    def _repeat_kv_quant(self, quant, scale, zp):
        """Repeat quantized value tensors for GQA."""
        if self.num_kv_groups == 1:
            return quant, scale, zp
        return (
            self._repeat_kv(quant.float()).to(quant.dtype),
            self._repeat_kv(scale),
            self._repeat_kv(zp),
        )

    def get_seq_length(self) -> int:
        return self.seq_len

    def memory_stats(self) -> dict:
        """Report memory usage statistics."""
        stats = {"seq_len": self.seq_len, "buffer_len": 0, "quantized_len": 0}

        if self._key_buffer is not None:
            stats["buffer_len"] = self._key_buffer.shape[2]
            buffer_bytes = (
                self._key_buffer.nelement() * self._key_buffer.element_size()
                + self._value_buffer.nelement() * self._value_buffer.element_size()
            )
            stats["buffer_bytes"] = buffer_bytes

        if self._key_quant_state is not None:
            quant_seq = 0
            quant_bytes = 0
            for k, v in self._key_quant_state.items():
                if isinstance(v, torch.Tensor):
                    quant_bytes += v.nelement() * v.element_size()
                    if v.dim() >= 3:
                        quant_seq = max(quant_seq, v.shape[2])
            if self._value_quant is not None:
                quant_bytes += self._value_quant.nelement() * self._value_quant.element_size()
                quant_bytes += self._value_scale.nelement() * self._value_scale.element_size()
                quant_bytes += self._value_zp.nelement() * self._value_zp.element_size()
            stats["quantized_len"] = quant_seq
            stats["quantized_bytes"] = quant_bytes

        # Estimate what full precision would cost
        fp_bytes = self.seq_len * self.num_kv_heads * self.head_dim * 2 * 2  # keys + values, fp16
        stats["fp16_equivalent_bytes"] = fp_bytes
        total_bytes = stats.get("buffer_bytes", 0) + stats.get("quantized_bytes", 0)
        stats["compression_ratio"] = fp_bytes / max(total_bytes, 1)

        return stats


class TurboQuantKVCacheCollection:
    """Collection of TurboQuant KV caches for all layers."""

    def __init__(
        self,
        num_layers: int,
        head_dim: int,
        num_heads: int,
        num_kv_heads: int,
        key_bit_width: int = 3,
        value_bit_width: int = 2,
        value_group_size: int = 32,
        buffer_size: int = 128,
        device: str = "cuda",
    ):
        self.num_layers = num_layers
        self.caches: List[TurboQuantKVCache] = []
        for _ in range(num_layers):
            self.caches.append(
                TurboQuantKVCache(
                    head_dim=head_dim,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    key_bit_width=key_bit_width,
                    value_bit_width=value_bit_width,
                    value_group_size=value_group_size,
                    buffer_size=buffer_size,
                    device=device,
                )
            )

    def __getitem__(self, layer_idx: int) -> TurboQuantKVCache:
        return self.caches[layer_idx]

    def get_seq_length(self) -> int:
        if self.caches:
            return self.caches[0].get_seq_length()
        return 0

    def memory_stats(self) -> dict:
        total = {"layers": self.num_layers}
        layer_stats = [c.memory_stats() for c in self.caches]
        total["total_buffer_bytes"] = sum(s.get("buffer_bytes", 0) for s in layer_stats)
        total["total_quantized_bytes"] = sum(s.get("quantized_bytes", 0) for s in layer_stats)
        total["total_fp16_equivalent"] = sum(s.get("fp16_equivalent_bytes", 0) for s in layer_stats)
        total_actual = total["total_buffer_bytes"] + total["total_quantized_bytes"]
        total["overall_compression_ratio"] = total["total_fp16_equivalent"] / max(total_actual, 1)
        total["seq_len"] = self.get_seq_length()
        return total
