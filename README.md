# TurboQuant

Compress the KV cache of large language models down to ~3 bits during inference. No fine-tuning, no training -- just patch your HuggingFace model and go.

This is a from-scratch implementation of the TurboQuant paper (ICLR 2026) by Zandieh et al. The core idea: randomly rotate key vectors onto the unit sphere, quantize each coordinate with a Lloyd-Max codebook, then correct the residual with a 1-bit Johnson-Lindenstrauss sketch. The result is an unbiased inner product estimator that lets you shrink the KV cache by 5-8x without wrecking attention quality.

Papers this builds on:
- [TurboQuant](https://arxiv.org/abs/2504.19874) (ICLR 2026) -- the full system
- [QJL](https://arxiv.org/abs/2406.03482) (AAAI 2025) -- the 1-bit residual trick
- [PolarQuant](https://arxiv.org/abs/2502.02617) (AISTATS 2026) -- polar coordinate quantization

## Quick Start

Clone the repo and install dependencies:

```bash
git clone https://github.com/AmmarTee/Turbo-Quant.git
cd Turbo-Quant
pip install -r requirements.txt
```

### Sanity check (no GPU, no model download)

Run the smoke test first to make sure everything is wired up:

```bash
python demo.py
```

This tests QJL, the Lloyd-Max codebook, TurboQuant MSE/Prod quantizers, and the KV cache layer -- all on CPU with random vectors. Takes a few seconds.

### Run a real model

The quickstart script picks a model that fits your hardware automatically:

```bash
python quickstart.py
```

It checks your GPU VRAM and picks accordingly -- TinyLlama on CPU, Llama-3.1-8B with 4-bit weights on 8+ GB GPUs, full fp16 on 18+ GB. You can also specify a model directly:

```bash
# Small model, fast test
python quickstart.py --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Heavy model with 4-bit weights to fit in VRAM
python quickstart.py --model "meta-llama/Llama-3.1-8B-Instruct" --load-in-4bit

# Custom prompt
python quickstart.py --prompt "Write a short essay about black holes"
```

### Full CLI (more options)

For finer control over quantization settings, use `run_inference.py`:

```bash
python run_inference.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --key-bits 3 --value-bits 2 \
    --prompt "Explain the theory of relativity in simple terms"

# Compare output with and without compression
python run_inference.py \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    --prompt "What are the key challenges in quantum computing?" \
    --compare

# 4-bit model weights + TurboQuant KV cache (lowest memory)
python run_inference.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --load-in-4bit --key-bits 3 --value-bits 2 \
    --prompt "Write a Python function to sort a linked list"
```

### Use as a library

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from turbo_quant import patch_model_for_turbo_quant

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

# one line to patch
model, kv_caches = patch_model_for_turbo_quant(
    model,
    key_bit_width=3,
    value_bit_width=2,
    buffer_size=128,
)

inputs = tokenizer("Hello, how are you?", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## How it works

TurboQuant is a two-stage quantizer:

1. **Rotate + scalar quantize (b-1 bits):** Multiply the vector by a random orthogonal matrix so that each coordinate is roughly Gaussian, then quantize each one independently with a precomputed Lloyd-Max codebook. This captures most of the signal.

2. **QJL on the residual (1 bit):** Take the quantization error from stage 1, project it with a random Gaussian matrix, and store just the signs. This corrects the bias in inner product estimation and costs only 1 bit per dimension.

Keys use TurboQuant_prod (both stages) because attention scores are inner products and bias matters there. Values use plain per-group min-max quantization since they only get multiplied by already-computed attention weights.

## Presets

| Preset | Key bits | Value bits | Effective bits | Compression | Quality |
| --- | --- | --- | --- | --- | --- |
| high_quality | 4 | 3 | 3.5 | ~4.6x | Lossless |
| balanced | 3 | 2 | 2.5 | ~6.4x | Near-lossless |
| max_compression | 2 | 2 | 2.0 | ~8.0x | Slight degradation |

## Supported models

Works with any HuggingFace causal LM that uses standard attention. Currently handles Llama (2/3/3.1), Mistral, Gemma, Qwen2, and Phi-3 architectures. Adding a new architecture just means telling the patcher where to find the attention layers.

## Hardware

- **8+ GB VRAM:** 7B models with 4-bit weights + TurboQuant KV cache
- **16+ GB VRAM:** 7B models in fp16
- **CPU:** Works, just slow -- fine for testing with TinyLlama

## Project layout

```
turbo_quant/
  qjl.py              QJL 1-bit sign quantizer
  codebook.py          Lloyd-Max codebooks for Gaussian distributions
  turbo_quant.py       TurboQuant MSE and Prod quantizers
  kv_cache.py          Compressed KV cache with streaming updates
  attention_patch.py   Monkey-patches HuggingFace attention layers
quickstart.py          Auto-picks a model and runs it
run_inference.py       Full CLI with all the knobs
demo.py                CPU-only smoke test, no downloads needed
config.yaml            Default settings and presets
```

## References

- Zandieh, Daliri, Hadian, Mirrokni. *TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate.* ICLR 2026. [arXiv:2504.19874](https://arxiv.org/abs/2504.19874)
- Zandieh, Daliri, Han. *QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero Overhead.* AAAI 2025. [arXiv:2406.03482](https://arxiv.org/abs/2406.03482)
- Han, Kacham, Karbasi, Mirrokni, Zandieh. *PolarQuant: Quantizing KV Caches with Polar Transformation.* AISTATS 2026. [arXiv:2502.02617](https://arxiv.org/abs/2502.02617)

## License

MIT

## License

This is a research implementation for educational purposes. The original
TurboQuant, QJL, and PolarQuant algorithms are by Google Research.
