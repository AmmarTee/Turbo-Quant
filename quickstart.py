"""
Minimal script to test TurboQuant on an actual model.

Downloads a small model, patches it with TurboQuant, and generates text.
Picks a model that fits comfortably in VRAM (or runs on CPU if no GPU).

Usage:
    python quickstart.py
    python quickstart.py --model "mistralai/Mistral-7B-Instruct-v0.3"
    python quickstart.py --prompt "Write me a haiku about tensors"
    python quickstart.py --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # fast test
"""

import argparse
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from turbo_quant import patch_model_for_turbo_quant


# Small models that don't need much VRAM, good for quick testing
SMALL_MODELS = [
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",     # 1.1B, ~2.5 GB fp16
    "Qwen/Qwen2.5-0.5B-Instruct",              # 0.5B, ~1 GB fp16
    "microsoft/phi-2",                           # 2.7B, ~5.5 GB fp16
]

# Heavy models -- the actual targets
HEAVY_MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",        # needs ~16 GB fp16 or ~5 GB 4-bit
    "mistralai/Mistral-7B-Instruct-v0.3",       # needs ~14 GB fp16 or ~5 GB 4-bit
    "google/gemma-2-9b-it",                     # needs ~18 GB fp16
    "Qwen/Qwen2.5-7B-Instruct",                # needs ~14 GB fp16
]


def pick_default_model():
    """Pick the biggest model that will fit, or fall back to TinyLlama."""
    if not torch.cuda.is_available():
        print("[info] No GPU found, defaulting to TinyLlama (CPU mode)")
        return SMALL_MODELS[0], False

    vram_gb = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f"[info] GPU: {torch.cuda.get_device_name(0)}, VRAM: {vram_gb:.1f} GB")

    if vram_gb >= 18:
        return "meta-llama/Llama-3.1-8B-Instruct", False
    elif vram_gb >= 8:
        return "meta-llama/Llama-3.1-8B-Instruct", True   # 4-bit weights
    elif vram_gb >= 4:
        return SMALL_MODELS[0], False
    else:
        return SMALL_MODELS[1], False


def main():
    parser = argparse.ArgumentParser(description="Quick test of TurboQuant on a real model")
    parser.add_argument("--model", type=str, default=None, help="HuggingFace model name")
    parser.add_argument("--prompt", type=str, default="Explain what vector quantization is and why it matters, in two paragraphs.", help="Prompt to test with")
    parser.add_argument("--max-tokens", type=int, default=200, help="Tokens to generate")
    parser.add_argument("--key-bits", type=int, default=3)
    parser.add_argument("--value-bits", type=int, default=2)
    parser.add_argument("--load-in-4bit", action="store_true", help="Load weights in 4-bit (saves VRAM)")
    args = parser.parse_args()

    # Pick model
    use_4bit = args.load_in_4bit
    if args.model is None:
        model_name, use_4bit = pick_default_model()
    else:
        model_name = args.model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    # --- Load ---
    print(f"\n--- Loading {model_name} ---")
    if use_4bit:
        print("  (using 4-bit weight quantization to fit in VRAM)")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kw = {"torch_dtype": dtype, "device_map": "auto" if device == "cuda" else None}
    if use_4bit:
        from transformers import BitsAndBytesConfig
        load_kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
        )
        load_kw.pop("torch_dtype", None)

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kw)
    if device != "cuda":
        model = model.to(device)
    model.eval()

    if device == "cuda":
        print(f"  GPU memory after load: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # --- Patch with TurboQuant ---
    print(f"\n--- Patching with TurboQuant (keys={args.key_bits}b, values={args.value_bits}b) ---")
    model, kv_caches = patch_model_for_turbo_quant(
        model,
        key_bit_width=args.key_bits,
        value_bit_width=args.value_bits,
        buffer_size=128,
        device=device,
    )

    # --- Generate ---
    print(f"\n--- Generating ({args.max_tokens} tokens) ---")
    print(f"  Prompt: {args.prompt[:80]}{'...' if len(args.prompt) > 80 else ''}\n")

    # Build input with chat template if available
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        try:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": args.prompt}],
                tokenize=False, add_generation_prompt=True,
            )
        except Exception:
            text = args.prompt
    else:
        text = args.prompt

    inputs = tokenizer(text, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    new_tokens = out[0][input_len:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    tok_per_sec = len(new_tokens) / elapsed if elapsed > 0 else 0

    print(f"  {response}\n")
    print(f"--- Stats ---")
    print(f"  Generated {len(new_tokens)} tokens in {elapsed:.2f}s ({tok_per_sec:.1f} tok/s)")
    if device == "cuda":
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"  Peak GPU memory: {peak:.2f} GB")

    cache_stats = kv_caches.memory_stats()
    print(f"  KV cache: {cache_stats['seq_len']} tokens, ~{cache_stats.get('overall_compression_ratio', 0):.1f}x compression")
    print()


if __name__ == "__main__":
    main()
