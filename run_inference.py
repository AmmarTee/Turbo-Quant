"""
TurboQuant Inference Runner

Run heavy LLMs with TurboQuant KV cache compression for dramatically
reduced memory usage during long-context generation.

Usage:
    # Basic inference with Llama-3.1-8B
    python run_inference.py --model meta-llama/Llama-3.1-8B-Instruct --prompt "Explain quantum computing"

    # With custom quantization settings
    python run_inference.py --model mistralai/Mistral-7B-Instruct-v0.3 \
        --key-bits 3 --value-bits 2 --buffer-size 128

    # Long-context stress test
    python run_inference.py --model meta-llama/Llama-3.1-8B-Instruct \
        --prompt-file long_document.txt --max-new-tokens 512

    # Compare with and without TurboQuant
    python run_inference.py --model meta-llama/Llama-3.1-8B-Instruct \
        --prompt "Summarize the key ideas of relativity" --compare
"""

import argparse
import time
import sys
import os

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from turbo_quant import patch_model_for_turbo_quant


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run LLM inference with TurboQuant KV cache compression"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="HuggingFace model name or path",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Input prompt text",
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        default=None,
        help="File containing the input prompt",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default="You are a helpful assistant.",
        help="System prompt for chat models",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--key-bits",
        type=int,
        default=3,
        help="Bits for key quantization (2-4)",
    )
    parser.add_argument(
        "--value-bits",
        type=int,
        default=2,
        help="Bits for value quantization (2-4)",
    )
    parser.add_argument(
        "--value-group-size",
        type=int,
        default=32,
        help="Group size for value quantization",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=128,
        help="Number of recent tokens kept in full precision",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Model dtype",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: 'cuda', 'cpu', or 'auto'",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run both with and without TurboQuant for comparison",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling",
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Use sampling instead of greedy decoding",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code for custom models",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load model weights in 4-bit (bitsandbytes) to save GPU memory",
    )
    return parser.parse_args()


def get_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def get_dtype(dtype_arg: str):
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype_arg]


def load_model(args):
    """Load model and tokenizer from HuggingFace."""
    device = get_device(args.device)
    dtype = get_dtype(args.dtype)

    print(f"\n{'='*60}")
    print(f"Loading model: {args.model}")
    print(f"Device: {device}, Dtype: {args.dtype}")
    print(f"{'='*60}\n")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": args.trust_remote_code,
        "device_map": "auto" if device == "cuda" else None,
    }

    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        load_kwargs.pop("torch_dtype", None)

    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)

    if device != "cuda" or "device_map" not in load_kwargs or load_kwargs["device_map"] is None:
        model = model.to(device)

    model.eval()

    # Report GPU memory after loading
    if device == "cuda":
        mem_gb = torch.cuda.memory_allocated() / 1e9
        print(f"GPU memory after model load: {mem_gb:.2f} GB")

    return model, tokenizer, device


def prepare_prompt(args, tokenizer):
    """Build the input prompt."""
    if args.prompt_file:
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            user_text = f.read()
    elif args.prompt:
        user_text = args.prompt
    else:
        user_text = "Explain how vector quantization enables efficient AI model compression, and why it matters for deploying large language models."

    # Try to use chat template if available
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = [
            {"role": "system", "content": args.system_prompt},
            {"role": "user", "content": user_text},
        ]
        try:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            prompt = f"{args.system_prompt}\n\nUser: {user_text}\n\nAssistant:"
    else:
        prompt = f"{args.system_prompt}\n\nUser: {user_text}\n\nAssistant:"

    return prompt, user_text


def run_generation(model, tokenizer, prompt, args, device, label=""):
    """Run text generation and return output + stats."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    input_len = inputs["input_ids"].shape[1]

    gen_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature if args.do_sample else 1.0,
        top_p=args.top_p if args.do_sample else 1.0,
        pad_token_id=tokenizer.pad_token_id,
    )

    # Clear CUDA cache
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start_time = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            generation_config=gen_config,
        )
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time

    # Decode output
    new_tokens = outputs[0][input_len:]
    output_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    num_new_tokens = len(new_tokens)

    # Memory stats
    peak_mem = 0
    if device == "cuda":
        peak_mem = torch.cuda.max_memory_allocated() / 1e9

    stats = {
        "input_tokens": input_len,
        "output_tokens": num_new_tokens,
        "elapsed_seconds": elapsed,
        "tokens_per_second": num_new_tokens / elapsed if elapsed > 0 else 0,
        "peak_gpu_memory_gb": peak_mem,
    }

    return output_text, stats


def print_stats(stats, label=""):
    prefix = f"[{label}] " if label else ""
    print(f"\n{prefix}--- Generation Stats ---")
    print(f"{prefix}Input tokens:      {stats['input_tokens']}")
    print(f"{prefix}Output tokens:     {stats['output_tokens']}")
    print(f"{prefix}Time:              {stats['elapsed_seconds']:.2f}s")
    print(f"{prefix}Speed:             {stats['tokens_per_second']:.1f} tok/s")
    if stats["peak_gpu_memory_gb"] > 0:
        print(f"{prefix}Peak GPU memory:   {stats['peak_gpu_memory_gb']:.2f} GB")


def main():
    args = parse_args()

    # Load model
    model, tokenizer, device = load_model(args)

    # Prepare prompt
    prompt, user_text = prepare_prompt(args, tokenizer)
    print(f"\nPrompt ({len(prompt)} chars): {user_text[:200]}{'...' if len(user_text) > 200 else ''}\n")

    if args.compare:
        # --- Run WITHOUT TurboQuant ---
        print("=" * 60)
        print("Running BASELINE (no compression)")
        print("=" * 60)
        output_baseline, stats_baseline = run_generation(
            model, tokenizer, prompt, args, device, "Baseline"
        )
        print_stats(stats_baseline, "Baseline")
        print(f"\n[Baseline] Output:\n{output_baseline}\n")

    # --- Apply TurboQuant ---
    print("=" * 60)
    print("Applying TurboQuant KV Cache Compression")
    print(f"  Keys:   {args.key_bits}-bit TurboQuant_prod")
    print(f"  Values: {args.value_bits}-bit group quantization")
    print(f"  Buffer: {args.buffer_size} tokens full-precision")
    effective = (args.key_bits + args.value_bits) / 2
    print(f"  Effective: {effective:.1f} bits/value (~{16/effective:.1f}x compression)")
    print("=" * 60)

    model, kv_caches = patch_model_for_turbo_quant(
        model,
        key_bit_width=args.key_bits,
        value_bit_width=args.value_bits,
        value_group_size=args.value_group_size,
        buffer_size=args.buffer_size,
        device=device,
    )

    # --- Run WITH TurboQuant ---
    print("\nGenerating with TurboQuant...\n")
    output_turbo, stats_turbo = run_generation(
        model, tokenizer, prompt, args, device, "TurboQuant"
    )
    print_stats(stats_turbo, "TurboQuant")
    print(f"\n[TurboQuant] Output:\n{output_turbo}\n")

    # Print KV cache memory stats
    cache_stats = kv_caches.memory_stats()
    print("\n--- KV Cache Memory ---")
    print(f"Sequence length:     {cache_stats['seq_len']}")
    print(f"Compression ratio:   {cache_stats.get('overall_compression_ratio', 'N/A'):.1f}x")

    if args.compare:
        print("\n" + "=" * 60)
        print("COMPARISON SUMMARY")
        print("=" * 60)
        if stats_baseline["peak_gpu_memory_gb"] > 0 and stats_turbo["peak_gpu_memory_gb"] > 0:
            mem_saved = stats_baseline["peak_gpu_memory_gb"] - stats_turbo["peak_gpu_memory_gb"]
            mem_pct = (mem_saved / stats_baseline["peak_gpu_memory_gb"]) * 100
            print(f"Memory saved:  {mem_saved:.2f} GB ({mem_pct:.1f}%)")
        speedup = stats_turbo["tokens_per_second"] / max(stats_baseline["tokens_per_second"], 0.01)
        print(f"Speed:         {speedup:.2f}x")


if __name__ == "__main__":
    main()
