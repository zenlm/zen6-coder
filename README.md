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
- halogen
- strix-halo
- mtp
base_model:
- Qwen/Qwen3.8-Flash-Next
- unsloth/Qwen3.8-Flash-Next-GGUF
---

<div align="center">

# Zen6 Coder

**Agentic coding on your own machine. 180B parameters, 6B active per token.**

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-zenlm%2Fzen6--coder-blue)](https://huggingface.co/zenlm/zen6-coder)
[![GitHub](https://img.shields.io/badge/GitHub-zenlm%2Fzen6--coder-black)](https://github.com/zenlm/zen6-coder)

</div>

---

## Overview

Zen6 Coder is Zen 6 for agentic coding, from Hanzo AI (Zen LM). It is built to
run locally: only 6B parameters are active per token, the 4-bit build is
93.7 GB and fits a 128 GB unified-memory machine such as AMD Strix Halo, it
reads 262,144 tokens natively, it calls tools, and its own MTP head drafts
tokens ahead for faster decoding.

It is also on the Hanzo API as `zen6-coder`.

## Specifications

| | |
| :--- | :--- |
| **Architecture** | GGUF `general.architecture`: `qwen4exp`. config.json `model_type`: `qwen4_exp`, `architectures`: `Qwen4ExpForConditionalGeneration`. hanzo-engine reads it as `zen6`. |
| **Parameters** | 180B total: 125B MoE language model, 51B n-gram embedding, 4B MTP head |
| **Active per token** | 6B (10 routed experts + 1 shared expert, of 512) |
| **Layers** | 48, as 12 × [3 × (Gated DeltaNet → MoE) → 1 × (sparse attention → MoE)] |
| **Context** | 262,144 tokens native; 1,048,576 with YaRN (`rope_theta: 10000000`, `factor: 4.0`) |

## Design

- **N-gram embedding.** A 51B table of 20,000,000 bigrams and trigrams,
  injected at layer 2. It is looked up, not computed, so it adds lexical
  memory without adding compute per token.
- **MTP head.** One dedicated layer (4B) trained for multi-step prediction.
  It drafts tokens the model verifies in one pass, for 1.3x–1.7x faster decoding
  at low concurrency.
- **Gated DeltaNet.** Linear attention with 48 value heads and 16 query/key
  heads (head dim 128); memory stays constant as the sequence grows.
- **Sparse attention.** 24 query heads, 2 KV heads (head dim 256, RoPE dim 64).
  An MQA indexer (4 query heads, 1 shared key) picks 512 blocks, 2,048 tokens,
  for each query.
- **Gated residuals.** 4 residual branches with data-dependent read and write
  gates, bottleneck rank 320.
- **Reasoning control.** The chat template takes `reasoning_effort` (`xhigh`
  default, `medium`, `low`) and `enable_thinking=false` to skip thinking.

## Benchmarks

Scores are for the full-precision weights; the GGUF files here are 4-bit.

| Benchmark | Zen6 Coder | Claude-Opus-4.6 (Max) | DeepSeek-V4-Flash-0731 | Zen 5.8 |
| :--- | :---: | :---: | :---: | :---: |
| SWE-bench Pro | 62.5% | 53.4% | 56.0% | 61.7% |
| DeepSWE 1.1 | 58.7% | — | 54.4% | 42.2% |
| SWE-bench Multilingual | 81.0% | 77.5% | — | 73.8% |
| LiveCodeBench v6 | 91.9% | 88.8% | 90.6% | 90.3% |
| NL2Repo-Bench | 48.1% | 47.6% | 54.2% | 42.3% |
| GPQA Diamond | 91.7% | 91.3% | 90.8% | 89.2% |
| Toolathlon Verified (Pass@1) | 73.5% | — | 70.3% | 67.1% |
| CoWorkBench | 73.9% | 68.2% | 45.1% | 70.7% |

## Speed on AMD Strix Halo

Radeon 8060S, 128 GB unified memory. Measured with the Halogen engine's
resumable prompt-state cache; speedup is a warm resume over a cold prefill of
the same context.

| Context | Cold prefill | Warm-resume speedup |
| :---: | :---: | :---: |
| 512 tokens | 454.4 tok/s | 7.89x |
| 2,048 tokens | 959.1 tok/s | 14.08x |
| 8,192 tokens | 1,298.4 tok/s | 36.31x |
| 16,384 tokens | 1,373.4 tok/s | 60.47x |
| 32,768 tokens | 1,451.8 tok/s | 79.45x |

## Files

| Path | Contents | Size |
| :--- | :--- | :---: |
| `UD-IQ4_XS/` | 4-bit IQ4_XS GGUF, 3 shards | 93.7 GB |
| `MTP/` | MTP draft head, Q8_0 GGUF | 4.14 GB |

```bash
hf download zenlm/zen6-coder --local-dir zen6-coder
```

## Run it

### Hanzo API

```bash
curl https://api.hanzo.ai/v1/chat/completions \
  -H "Authorization: Bearer $HANZO_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"zen6-coder","messages":[{"role":"user","content":"Hello!"}]}'
```

### llama.cpp with MTP

```bash
llama-server \
  -m zen6-coder/UD-IQ4_XS/*-00001-of-00003.gguf \
  -md zen6-coder/MTP/*.gguf \
  --spec-type draft-mtp --spec-draft-n-max 2 \
  -c 262144 --port 8000
```

MTP drafting for this architecture needs a llama.cpp build that includes
[ggml-org/llama.cpp#28243](https://github.com/ggml-org/llama.cpp/pull/28243).
On other builds, drop `-md` and `--spec-type`; the model runs without drafting.
Drafting helps a single stream; skip it for concurrent serving.

### hanzo-engine

```bash
hanzo-engine serve -p 8000 --format gguf -m zen6-coder \
  -f "$(cd zen6-coder && echo UD-IQ4_XS/*.gguf | tr ' ' ';')"
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

## License & attribution

The weights are a derivative of a model released under the Qwen Community
License 1.0. Copyright (c) 2026 Qwen. The full license and permission notice:
<https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE>.
