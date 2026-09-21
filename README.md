---
license: other
license_name: qwen-community-1.0
language:
- en
- zh
pipeline_tag: text-generation
tags:
- zen6
- zen6-coder
- agentic-coding
- moe
- qwen3.8-flash-next
- unsloth
- halogen
- strix-halo
- mtp
base_model:
- Qwen/Qwen3.8-Flash-Next
- unsloth/Qwen3.8-Flash-Next-GGUF
---

<div align="center">

# Zen6 Coder: 180B Frontier Agentic MoE

**125B Base (6B Active) | 51B N-Gram Embedding | 4B MTP Drafter | 62.5 SWE-bench Pro**

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-zenlm%2Fzen6--coder-blue)](https://huggingface.co/zenlm/zen6-coder)
[![GitHub](https://img.shields.io/badge/GitHub-zenlm%2Fzen6--coder-black)](https://github.com/zenlm/zen6-coder)

</div>

---

## Architectural Highlights

**Zen6 Coder** is built on the next-generation **Qwen3.8-Flash-Next** architecture, representing a fundamental redesign of modern agentic language models:

- **180 Billion Total Parameters**:
  - **125B Base Language Model** with only **6B Activated Parameters** per token (10 routed experts + 1 shared expert out of 512 total experts).
  - **51B N-Gram Embedding Table** (20,000,000 bigrams/trigrams injected at layer 2) enabling ultra-dense lexical memory without compute overhead.
  - **4B Multi-Token Prediction (MTP) Head** (1 dedicated layer trained with multi-step prediction) delivering 1.3x–1.7x speculative acceleration out of the box.
- **Hybrid Attention with QSA (Qwen Sparse Attention)**:
  - 48 Layers arranged as $12 \times [3 \times (\text{Gated DeltaNet} \to \text{MoE}) \to 1 \times (\text{QSA} \to \text{MoE})]$.
  - **Gated DeltaNet**: 48 linear attention heads for V, 16 heads for QK (head dim 128) handling constant-memory linear sequence progression.
  - **QSA**: 24 Query heads, 2 KV heads (head dim 256, RoPE dim 64) with an MQA Indexer (4 Query / 1 Shared Key, budget 512 micro-blocks / 2048 tokens).
- **Gated Residuals**: 4 residual branches modulated by data-dependent read and write gates with bottleneck rank 320.
- **Context Length**: 262,144 tokens native, extensible to 1,000,000 tokens via YaRN (`rope_theta: 10000000, factor: 4.0`).

---

## State-of-the-Art Coding & Agent Benchmarks

Zen6 Coder establishes new state-of-the-art benchmarks in real-world software engineering and agentic coding:

| Benchmark | Zen6 Coder (Qwen3.8-Flash-Next) | Claude-Opus-4.6 (Max) | DeepSeek-V4-Flash-0731 | Qwen3.8-27B |
| :--- | :---: | :---: | :---: | :---: |
| **SWE-bench Pro** | **62.5%** | 53.4% | 56.0% | 61.7% |
| **DeepSWE 1.1** | **58.7%** | — | 54.4% | 42.2% |
| **SWE-bench Multilingual** | **81.0%** | 77.5% | — | 73.8% |
| **LiveCodeBench v6** | **91.9%** | 88.8% | 90.6% | 90.3% |
| **NL2Repo-Bench** | **48.1%** | 47.6% | 54.2% | 42.3% |
| **GPQA Diamond** | **91.7%** | 91.3% | 90.8% | 89.2% |
| **Toolathlon Verified (Pass@1)** | **73.5%** | — | 70.3% | 67.1% |
| **CoWorkBench** | **73.9%** | 68.2% | 45.1% | 70.7% |

---

## Model Weights & Formats

This repository distributes Zen6 Coder in two primary formats:

### 1. Unsloth Dynamic GGUF (`UD-IQ4_XS`) + MTP
- **`UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf`** (10.9 MB)
- **`UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00003.gguf`** (49.8 GB)
- **`UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00003-of-00003.gguf`** (43.8 GB)
- **`MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf`** (Dedicated 4B MTP draft head)

### 2. Halogen W4B Format (AMD Strix Halo Native)
Optimized for AMD Ryzen AI Max+ 395 / Radeon 8060S (gfx1151) with ROCm and Halogen resumable prompt-state caching.

---

## Hardware Benchmarks

### AMD Strix Halo (8060S / 128GB Unified Memory)
| Context Length | Cold Prefill | Halogen Warm Resume | Speedup |
| :---: | :---: | :---: | :---: |
| **512 tokens** | 454.4 tok/s | **0.1 ms** | **7.89x** |
| **2,048 tokens** | 959.1 tok/s | **0.1 ms** | **14.08x** |
| **8,192 tokens** | 1,298.4 tok/s | **0.1 ms** | **36.31x** |
| **16,384 tokens** | 1,373.4 tok/s | **0.1 ms** | **60.47x** |
| **32,768 tokens** | 1,451.8 tok/s | **0.1 ms** | **79.45x** |

---

## Serving Instructions

### Option A: AMD Strix Halo (Halogen Engine)
```bash
sudo podman run -d --name halogen --device=/dev/kfd --device=/dev/dri \
  -v /models:/models -p 8731:8731 halogen:latest \
  --model /models/qwen38-flash-next-w4b.hgn \
  --port 8731 --max-tokens-cap 65536
```

### Option B: Cross-Platform Llama.cpp with MTP Speculative Decoding
```bash
llama-server \
  -m UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf \
  --draft-model MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf \
  --draft-max 3 \
  -c 262144 \
  --port 8000
```

### Option C: Pure-Rust `hanzo-engine`
```bash
hanzo-engine serve \
  --model zenlm/zen6-coder \
  --format gguf \
  --mtp MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf \
  --context-window 262144 \
  --port 8000
```

---

## Citation

```bibtex
@techreport{zenlm2026zen6coder,
  title={Zen6 Coder: 180B-Class Hybrid Gated DeltaNet Sparse Attention MoE for Frontier Agentic Software Engineering},
  author={Hanzo AI and Zen LM Team},
  year={2026},
  publisher={Zen LM / Hanzo AI}
}
```
