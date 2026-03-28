# TurboQuant: Run Heavy LLMs with Extreme KV Cache Compression

Implementation of **TurboQuant** (ICLR 2026), a near-optimal vector quantization
algorithm that compresses the KV cache of large language models to ~3 bits per
value with zero accuracy loss.

Based on three papers from Google Research:
- [TurboQuant](https://arxiv.org/abs/2504.19874) - The combined system (ICLR 2026)
- [QJL](https://arxiv.org/abs/2406.03482) - 1-bit quantized JL transform (AAAI 2025)
- [PolarQuant](https://arxiv.org/abs/2502.02617) - Polar coordinate quantization (AISTATS 2026)

## What This Does

TurboQuant compresses the Key-Value cache during LLM inference, enabling:
- **>5x memory reduction** for the KV cache (16-bit to ~3-bit)
- **Up to 8x speedup** in attention computation (on H100 GPUs)
- **Zero accuracy loss** at 3.5 bits, marginal degradation at 2.5 bits
- **No training or fine-tuning** required -- works as a drop-in replacement

This lets you run larger models, process longer contexts, and serve more
concurrent users on the same hardware.

## Quick Start

### Install

```bash
pip install -r requirements.txt
```

### Run Inference

```bash
# Basic: Run Llama-3.1-8B with TurboQuant
python run_inference.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --prompt "Explain the theory of relativity in simple terms" \
    --max-new-tokens 256

# Memory-constrained: Use 4-bit model weights + TurboQuant KV cache
python run_inference.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --load-in-4bit \
    --key-bits 3 --value-bits 2 \
    --prompt "Write a Python function to sort a linked list"

# Compare with/without compression
python run_inference.py \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    --prompt "What are the key challenges in quantum computing?" \
    --compare

# Long document processing
python run_inference.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --prompt-file long_document.txt \
    --max-new-tokens 512 \
    --buffer-size 256
```

### Use as a Library

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from turbo_quant import patch_model_for_turbo_quant

# Load any HuggingFace model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

# Patch with TurboQuant -- one line!
model, kv_caches = patch_model_for_turbo_quant(
    model,
    key_bit_width=3,    # 3-bit keys (TurboQuant_prod)
    value_bit_width=2,  # 2-bit values (group quantization)
    buffer_size=128,    # Recent tokens in full precision
)

# Use normally -- compression is automatic
inputs = tokenizer("Hello, how are you?", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## How TurboQuant Works

### The Algorithm (Two-Stage Approach)

**Stage 1 -- MSE-Optimal Scalar Quantization:**
1. Randomly rotate the input vector: `y = Pi * x` (where Pi is a random orthogonal matrix)
2. After rotation, each coordinate follows a Beta distribution (converges to Gaussian in high dimensions)
3. Apply precomputed Lloyd-Max optimal scalar quantizers to each coordinate independently
4. This uses (b-1) bits and captures the main signal

**Stage 2 -- QJL Residual Correction (1 bit):**
1. Compute the residual: `r = x - dequantize(quantize(x))`
2. Apply the QJL transform: `sign(S * r)` where S is a random Gaussian matrix
3. This eliminates bias in inner product estimation

The result is an **unbiased** inner product estimator with near-optimal distortion:

$$D_{\text{prod}} \leq \frac{3\sqrt{\pi}}{2} \cdot \frac{\|y\|^2}{d} \cdot \frac{1}{4^b}$$

### Key Quantization (TurboQuant_prod)

Keys in the KV cache are used to compute attention scores via inner products
with queries. TurboQuant_prod provides **unbiased** estimation of these inner
products, which is critical for maintaining attention quality.

### Value Quantization (Group Quantization)

Values don't need unbiased inner product properties -- they are multiplied
by already-computed attention weights. Standard per-group min-max quantization
works well here with minimal overhead.

## Architecture

```
turbo_quant/
  __init__.py            -- Package exports
  qjl.py                -- QJL 1-bit quantizer (Johnson-Lindenstrauss)
  codebook.py            -- Lloyd-Max scalar quantizer codebooks
  turbo_quant.py         -- TurboQuant MSE and Prod quantizers
  kv_cache.py            -- KV cache integration layer
  attention_patch.py     -- HuggingFace model patcher
run_inference.py         -- CLI inference script
config.yaml              -- Default configuration
```

## Configuration Presets

| Preset | Key Bits | Value Bits | Effective | Compression | Quality |
| --- | --- | --- | --- | --- | --- |
| High Quality | 4 | 3 | 3.5 bit | ~4.6x | Lossless |
| Balanced | 3 | 2 | 2.5 bit | ~6.4x | Near-lossless |
| Max Compression | 2 | 2 | 2.0 bit | ~8.0x | Slight degradation |

## Supported Models

Any HuggingFace causal LM with standard attention. Tested architectures:
- **Llama** family (Llama-2, Llama-3, Llama-3.1, CodeLlama)
- **Mistral** / Mixtral
- **Gemma** family
- **Qwen2** family
- **Phi-3**

## Hardware Requirements

- **GPU**: Any NVIDIA GPU with >= 8GB VRAM (for 7-8B models with 4-bit weights)
- **For full fp16**: >= 16GB VRAM for 7B models, >= 40GB for 13B models
- **CPU**: Supported but significantly slower

## References

```bibtex
@inproceedings{zandieh2025turboquant,
  title={TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate},
  author={Zandieh, Amir and Daliri, Majid and Hadian, Majid and Mirrokni, Vahab},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026}
}

@inproceedings{zandieh2024qjl,
  title={QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero Overhead},
  author={Zandieh, Amir and Daliri, Majid and Han, Insu},
  booktitle={AAAI Conference on Artificial Intelligence},
  year={2025}
}

@inproceedings{han2025polarquant,
  title={PolarQuant: Quantizing KV Caches with Polar Transformation},
  author={Han, Insu and Kacham, Praneeth and Karbasi, Amin and Mirrokni, Vahab and Zandieh, Amir},
  booktitle={International Conference on Artificial Intelligence and Statistics (AISTATS)},
  year={2026}
}
```

## License

This is a research implementation for educational purposes. The original
TurboQuant, QJL, and PolarQuant algorithms are by Google Research.
