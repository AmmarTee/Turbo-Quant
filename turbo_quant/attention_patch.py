"""
Attention Patching for HuggingFace Models

Monkey-patches the attention mechanism of HuggingFace transformer models
to use TurboQuant-compressed KV caches. Supports:
    - Llama family (Llama-2, Llama-3, Llama-3.1, CodeLlama)
    - Mistral / Mixtral
    - Gemma family
    - Phi-3
    - Qwen2

The patching replaces the standard KV cache with TurboQuantKVCache during
autoregressive generation, enabling >4x memory reduction for long contexts.
"""

import math
import warnings
from typing import Optional, Tuple, Dict, Any
from functools import wraps

import torch
import torch.nn as nn
import torch.nn.functional as F

from .kv_cache import TurboQuantKVCache, TurboQuantKVCacheCollection


class TurboQuantAttentionWrapper:
    """Wraps a HuggingFace attention module to use TurboQuant KV cache.

    During generation with use_cache=True, this wrapper:
    1. Intercepts the key/value states after projection + RoPE
    2. Stores them in a TurboQuant KV cache
    3. Computes attention scores using the compressed cache
    """

    def __init__(
        self,
        original_attention: nn.Module,
        kv_cache: TurboQuantKVCache,
        layer_idx: int,
    ):
        self.original = original_attention
        self.kv_cache = kv_cache
        self.layer_idx = layer_idx

    def forward_with_turbo_cache(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        """Modified forward that uses TurboQuant KV cache."""
        attn = self.original
        bsz, q_len, _ = hidden_states.size()

        # Compute Q, K, V projections (same as original)
        query_states = attn.q_proj(hidden_states)
        key_states = attn.k_proj(hidden_states)
        value_states = attn.v_proj(hidden_states)

        # Reshape to (batch, heads, seq, head_dim)
        num_heads = getattr(attn, "num_heads", attn.config.num_attention_heads if hasattr(attn, "config") else query_states.shape[-1] // attn.head_dim)
        num_kv_heads = getattr(attn, "num_key_value_heads", num_heads)
        head_dim = attn.head_dim

        query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)

        # Apply RoPE
        if hasattr(attn, "rotary_emb"):
            cos, sin = attn.rotary_emb(value_states, position_ids)
            query_states, key_states = _apply_rotary_pos_emb(
                query_states, key_states, cos, sin
            )

        # Update TurboQuant KV cache
        self.kv_cache.update(key_states, value_states)

        # Compute attention using compressed cache
        kv_seq_len = self.kv_cache.get_seq_length()

        # Build proper attention mask for the full sequence
        if attention_mask is not None and attention_mask.shape[-1] < kv_seq_len:
            # Extend mask if needed
            pad_len = kv_seq_len - attention_mask.shape[-1]
            attention_mask = F.pad(attention_mask, (pad_len, 0), value=0)

        attn_output = self.kv_cache.get_attention_output(query_states, attention_mask)

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, num_heads * head_dim)

        # Output projection
        attn_output = attn.o_proj(attn_output)

        return attn_output, None, self.kv_cache if use_cache else None


def _apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary position embeddings to query and key tensors."""
    # Handle different RoPE output formats across HF model versions
    if isinstance(cos, tuple):
        cos, sin = cos

    # Ensure correct shape: (batch, 1, seq, dim) or (1, 1, seq, dim)
    if cos.dim() == 2:
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
    elif cos.dim() == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


def _rotate_half(x):
    """Rotate half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def patch_model_for_turbo_quant(
    model: nn.Module,
    key_bit_width: int = 3,
    value_bit_width: int = 2,
    value_group_size: int = 32,
    buffer_size: int = 128,
    device: str = "cuda",
) -> Tuple[nn.Module, TurboQuantKVCacheCollection]:
    """Patch a HuggingFace model to use TurboQuant KV cache.

    This replaces the forward method of each attention layer to intercept
    KV states and store them in compressed TurboQuant caches.

    Supports: Llama, Mistral, Gemma, Phi, Qwen model families.

    Args:
        model: HuggingFace CausalLM model
        key_bit_width: Bits for key quantization (2-4, default 3)
        value_bit_width: Bits for value quantization (2-4, default 2)
        value_group_size: Group size for value quantization
        buffer_size: Number of recent tokens kept in full precision
        device: Target device

    Returns:
        (patched_model, kv_cache_collection) tuple
    """
    config = model.config
    num_layers = config.num_hidden_layers
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    num_heads = config.num_attention_heads
    num_kv_heads = getattr(config, "num_key_value_heads", num_heads)

    # Create cache collection
    kv_caches = TurboQuantKVCacheCollection(
        num_layers=num_layers,
        head_dim=head_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        key_bit_width=key_bit_width,
        value_bit_width=value_bit_width,
        value_group_size=value_group_size,
        buffer_size=buffer_size,
        device=device,
    )

    # Find and patch attention layers
    attention_layers = _find_attention_layers(model)

    for idx, attn_module in enumerate(attention_layers):
        wrapper = TurboQuantAttentionWrapper(attn_module, kv_caches[idx], idx)

        # Replace the forward method
        original_forward = attn_module.forward

        def make_patched_forward(w):
            @wraps(original_forward)
            def patched_forward(hidden_states, **kwargs):
                return w.forward_with_turbo_cache(hidden_states, **kwargs)
            return patched_forward

        attn_module.forward = make_patched_forward(wrapper)
        attn_module._turbo_quant_wrapper = wrapper

    print(f"[TurboQuant] Patched {len(attention_layers)} attention layers")
    print(f"[TurboQuant] Key quantization: {key_bit_width}-bit, Value quantization: {value_bit_width}-bit")
    print(f"[TurboQuant] Buffer size: {buffer_size} tokens")
    effective_bits = (key_bit_width + value_bit_width) / 2
    compression = 16 / effective_bits
    print(f"[TurboQuant] Effective: {effective_bits:.1f} bits/value, ~{compression:.1f}x compression")

    return model, kv_caches


def _find_attention_layers(model: nn.Module) -> list:
    """Find all attention sub-modules in a HuggingFace model."""
    attention_layers = []

    # Common patterns across HF model architectures
    model_body = None
    for attr in ["model", "transformer", "gpt_neox"]:
        if hasattr(model, attr):
            model_body = getattr(model, attr)
            break

    if model_body is None:
        model_body = model

    layers_container = None
    for attr in ["layers", "h", "block"]:
        if hasattr(model_body, attr):
            layers_container = getattr(model_body, attr)
            break

    if layers_container is None:
        raise ValueError(
            "Could not find transformer layers. Supported architectures: "
            "Llama, Mistral, Gemma, Phi, Qwen, GPT-NeoX"
        )

    for layer in layers_container:
        attn = None
        for attr in ["self_attn", "attention", "attn"]:
            if hasattr(layer, attr):
                attn = getattr(layer, attr)
                break
        if attn is not None:
            attention_layers.append(attn)

    return attention_layers


def unpatch_model(model: nn.Module) -> nn.Module:
    """Remove TurboQuant patches and restore original attention."""
    attention_layers = _find_attention_layers(model)
    for attn in attention_layers:
        if hasattr(attn, "_turbo_quant_wrapper"):
            # The original forward is stored by functools.wraps
            if hasattr(attn.forward, "__wrapped__"):
                attn.forward = attn.forward.__wrapped__
            delattr(attn, "_turbo_quant_wrapper")
    return model
