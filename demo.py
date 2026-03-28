"""
Quick sanity check -- runs entirely on CPU, no model download needed.

Tests the core TurboQuant quantization pipeline on synthetic vectors
to verify everything is wired up correctly.

Usage:
    python demo.py
"""

import torch
import time

from turbo_quant.qjl import QJL
from turbo_quant.codebook import LloydMaxCodebook
from turbo_quant.turbo_quant import TurboQuantMSE, TurboQuantProd
from turbo_quant.kv_cache import TurboQuantKVCache


def banner(text):
    print(f"\n{'='*50}")
    print(f"  {text}")
    print(f"{'='*50}")


def test_qjl():
    banner("QJL -- 1-bit Sign Quantization")
    dim = 128
    qjl = QJL(dim, device="cpu")

    x = torch.randn(4, 32, dim)
    x = x / x.norm(dim=-1, keepdim=True)

    z = qjl.quantize(x)              # -> sign bits {-1, +1}
    x_hat = qjl.dequantize(z)        # -> reconstruction

    mse = (x - x_hat).pow(2).mean().item()
    print(f"  dim={dim}, vectors=4x32")
    print(f"  quantized shape:  {z.shape} (sign bits)")
    print(f"  reconstruction MSE: {mse:.6f}")

    # unbiased inner product check
    y = torch.randn(4, 8, dim)
    true_ip = torch.matmul(y, x.transpose(-2, -1))
    est_ip = torch.matmul(y, x_hat.transpose(-2, -1))
    bias = (true_ip - est_ip).mean().item()
    print(f"  inner product bias: {bias:.6f}  (should be ~0)")
    return True


def test_codebook():
    banner("Lloyd-Max Codebook (1-4 bit)")
    dim = 128
    for b in range(1, 5):
        cb = LloydMaxCodebook(b, dim, device="cpu")
        vals = torch.randn(5000) / (dim ** 0.5)
        idx = cb.quantize(vals)
        rec = cb.dequantize(idx)
        mse = (vals - rec).pow(2).mean().item()
        print(f"  {b}-bit  |  levels={cb.num_levels:>2d}  |  MSE={mse:.7f}")
    return True


def test_turbo_quant():
    banner("TurboQuant MSE + Prod")
    dim = 128
    for b in [2, 3, 4]:
        tq_mse = TurboQuantMSE(dim, bit_width=b, device="cpu")
        tq_prod = TurboQuantProd(dim, bit_width=b, device="cpu")

        x = torch.randn(2, 16, dim)
        x = x / x.norm(dim=-1, keepdim=True)

        rec_mse, _, _ = tq_mse.quantize_dequantize(x)
        mse_err = (x - rec_mse).pow(2).mean().item()

        state = tq_prod.quantize(x)
        rec_prod = tq_prod.dequantize(state)
        y = torch.randn(2, 8, dim)
        true_ip = torch.matmul(y, x.transpose(-2, -1))
        est_ip = torch.matmul(y, rec_prod.transpose(-2, -1))
        ip_bias = (true_ip - est_ip).mean().item()

        print(f"  {b}-bit  |  MSE={mse_err:.7f}  |  IP bias={ip_bias:+.6f}")
    return True


def test_kv_cache():
    banner("KV Cache -- Simulated Transformer")
    batch, num_heads, num_kv_heads, head_dim = 1, 8, 4, 64
    buffer_size = 32

    cache = TurboQuantKVCache(
        head_dim=head_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        key_bit_width=3,
        value_bit_width=2,
        buffer_size=buffer_size,
        device="cpu",
    )

    # prefill 64 tokens
    k = torch.randn(batch, num_kv_heads, 64, head_dim)
    v = torch.randn(batch, num_kv_heads, 64, head_dim)
    cache.update(k, v)

    # decode 16 more
    for _ in range(16):
        cache.update(
            torch.randn(batch, num_kv_heads, 1, head_dim),
            torch.randn(batch, num_kv_heads, 1, head_dim),
        )

    q = torch.randn(batch, num_heads, 1, head_dim)
    out = cache.get_attention_output(q)

    stats = cache.memory_stats()
    print(f"  total tokens:      {stats['seq_len']}")
    print(f"  quantized tokens:  {stats['quantized_len']}")
    print(f"  buffer tokens:     {stats['buffer_len']}")
    print(f"  compression ratio: {stats.get('compression_ratio', 0):.2f}x")
    print(f"  attention output:  {out.shape}")
    return True


def main():
    print("TurboQuant -- Smoke Test")
    print("Running on CPU with synthetic data (no model download)\n")

    t0 = time.perf_counter()
    results = {
        "QJL": test_qjl(),
        "Codebook": test_codebook(),
        "TurboQuant": test_turbo_quant(),
        "KV Cache": test_kv_cache(),
    }
    elapsed = time.perf_counter() - t0

    banner("Results")
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  {name:.<30s} {status}")
    print(f"\n  Finished in {elapsed:.2f}s")

    if all(results.values()):
        print("\n  Everything works. You're ready to run a real model.\n")
    else:
        print("\n  Something broke -- check the output above.\n")


if __name__ == "__main__":
    main()
