# Hy3 (hy_v3) for llama.cpp — macOS Metal support

Patches and binaries to run **Tencent Hy3 (hy_v3) 295B MoE** on **llama.cpp** with **Apple Silicon GPU (Metal)**.

---

## Why this project exists

llama.cpp supports dozens of architectures, but **Hy3 (hy_v3) wasn't one of them**. Tencent released Hy3 as open-source (Apache 2.0), AngelSlim quantized it to GGUF, but to run it on a Mac someone had to write the missing piece: hy_v3 architecture support in llama.cpp.

This repo contains **the patches that add hy_v3 to llama.cpp** — architecture detection, weight loading, MoE + shared expert forward pass, and MTP self-speculative decoding. All compiled with **Metal** for Apple Silicon GPU.

The goal is simple: **run a 295B model on a MacBook**. Not on the cloud, not on a cluster. On a laptop.

---

## Credits

This starts and ends with **AngelSlim** and their work on HuggingFace:

[**AngelSlim/Hy3-GGUF**](https://huggingface.co/AngelSlim/Hy3-GGUF) — GGUF-quantized model (IQ1_M and Q4_K_M), mixed-precision recipes, importance matrix, setup script, benchmarks, chat template. Without this work, this project wouldn't exist.

AngelSlim provided:
- The **base patches** for hy_v3 architecture in llama.cpp
- The **IQ1_M quantization** with mixed recipe (critical weights in Q8_0/Q6_K, experts in IQ1_M/IQ2_XXS)
- The **importance matrix** to allocate bits where they matter
- The **chat template** for tool calling and reasoning

This repo takes those patches, applies them to llama.cpp, and **builds them with Metal for macOS**.

**Thank you AngelSlim.** 🙌

---

## The model

| Detail | Value |
|--------|-------|
| **Architecture** | Hy3 (hy_v3) — Hunyuan V3 |
| **Developed by** | Tencent |
| **Parameters** | 295B |
| **Layers** | 81 (80 routed + 1 MTP) |
| **Experts** | 192 (8 active per token) |
| **Gating** | Sigmoid + correction bias + top-8 selection |
| **Quantization** | IQ1_M (AngelSlim mixed recipe) |
| **File size** | ~85 GB (with MTP) |

> Original HF model: [Tencent/Hy3](https://huggingface.co/tencent/Hy3)
> AngelSlim GGUF quant: [**AngelSlim/Hy3-GGUF**](https://huggingface.co/AngelSlim/Hy3-GGUF)
> Download: [Hy3-IQ1_M-mtp.gguf](https://huggingface.co/AngelSlim/Hy3-GGUF/resolve/main/Hy3-IQ1_M-mtp.gguf) (85 GB, IQ1_M with MTP)

**IQ1_M vs BF16 quality loss**: ~+0.3% PPL — **imperceptible**. Full benchmarks on [AngelSlim's HF page](https://huggingface.co/AngelSlim/Hy3-GGUF), file `assets/benchmark.png`.

---

## Why a MacBook?

| Mac | RAM | IQ1_M (85 GB) | MTP | Context | Notes |
|-----|-----|:---:|:---:|:--------:|-------|
| **M5 Max** | 128 GB | ✅ | ✅ | 64K | Everything on, comfortable |
| **M4 Max** | 128 GB | ✅ | ✅ | 64K | Everything on |
| **M3 Max** | 128 GB | ✅ | ✅ | 64K | Everything on |
| **MacBook Pro** | 128 GB | ✅ | ✅ | 64K | Runs on a laptop |
| **Mac Studio** | 96 GB | ✅ | ❌ | 64K | KV q8_0 only, no MTP |
| **MacBook Pro** | 96 GB | ✅ | ❌ | 64K | Same as above |

A **295B MoE running on a MacBook with 128 GB** is a concrete milestone for local AI:

- **No cloud**, no API keys, no subscriptions
- **Total privacy** — data never leaves your machine
- **No dedicated GPU needed** — Apple Silicon unified memory is enough
- **Portable** — no server rack, no cluster

With 96 GB it still works: the model is ~85 GB, leaving ~11 GB for the system. Just compress the KV cache (`-ctk q8_0 -ctv q8_0`) and skip MTP (which adds ~2 GB of weights plus a draft KV cache). With 128 GB everything runs — MTP included, with headroom.

---

## Download

### 1. The GGUF model

```bash
# IQ1_M with MTP (85 GB) — recommended for 128 GB
wget https://huggingface.co/AngelSlim/Hy3-GGUF/resolve/main/Hy3-IQ1_M-mtp.gguf

# IQ1_M without MTP (84 GB) — for 96 GB
wget https://huggingface.co/AngelSlim/Hy3-GGUF/resolve/main/Hy3-IQ1_M.gguf
```

### 2. The code

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
git checkout 19bba67c1
git apply /path/to/0001-add-hyv3-support.patch
cp /path/to/hyv3.cpp src/models/
```

Or clone the ready-made fork:

```bash
git clone https://github.com/<your-username>/llama.cpp-hyv3
cd llama.cpp-hyv3
```

### 3. Build

```bash
mkdir build && cd build
cmake .. -DLLAMA_METAL=ON
make -j$(sysctl -n hw.logicalcpu)
```

---

## Commands

### CLI (base inference)

```bash
./build/bin/llama-cli \
    -m ~/Downloads/Hy3-IQ1_M-mtp.gguf \
    -c 65536 \
    -ngl 99 \
    -fa on \
    -ctk q8_0 -ctv q8_0 \
    -p "Hello" \
    -n 100 \
    --temp 0.6
```

### CLI (with reasoning/thinking)

```bash
./build/bin/llama-cli \
    -m ~/Downloads/Hy3-IQ1_M-mtp.gguf \
    -c 65536 \
    -ngl 99 -fa on \
    -ctk q8_0 -ctv q8_0 \
    -p "Hello" \
    -n 200 \
    --temp 0.6 \
    --reasoning on \
    --reasoning-budget -1
```

### Server (OpenAI-compatible API)

```bash
./build/bin/llama-server \
    -m ~/Downloads/Hy3-IQ1_M-mtp.gguf \
    -c 65536 \
    -ngl 99 -fa on \
    -ctk q8_0 -ctv q8_0 \
    --temp 0.6 \
    --port 8080
```

Test API call:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Hy3-IQ1_M-mtp",
    "messages": [{"role": "user", "content": "Hello"}],
    "temperature": 0.6,
    "max_tokens": 200
  }'
```

### For Mac with 96 GB RAM

```bash
# No MTP, compressed KV cache
./build/bin/llama-cli \
    -m ~/Downloads/Hy3-IQ1_M.gguf \
    -c 65536 \
    -ngl 99 -fa on \
    -ctk q8_0 -ctv q8_0 \
    -p "Hello" -n 100 \
    --temp 0.6
```

---

## Flag reference

| Flag | What it does |
|------|--------------|
| `-m PATH` | Path to the GGUF model file |
| `-c N` | Context size in tokens. `65536` = 64K. Higher = more memory |
| `-ngl N` | Layers to offload to GPU. `99` = all layers on Metal |
| `-fa on` | Flash attention — reduces memory and speeds up attention |
| `-ctk q8_0 -ctv q8_0` | KV cache in q8_0. **Essential for 96 GB** (saves ~20 GB) |
| `--temp N` | Sampling temperature. `0.0` = deterministic/greedy |
| `--reasoning on` | Enable thinking/reasoning (tag) |
| `--reasoning-budget N` | Max tokens for thinking. `-1` = unlimited |
| `--spec-type draft-mtp` | MTP self-speculative decoding (*-mtp.gguf only) |
| `--spec-draft-n-max N` | Max draft tokens per MTP step |

---

## Project structure

```
├── 0001-add-hyv3-support.patch   # Patch for 9 llama.cpp files (383 lines)
├── src/models/hyv3.cpp           # hy_v3 model implementation + MTP (388 lines)
└── README.md                     # This file
```

## Modified files in llama.cpp

| File | Change |
|------|--------|
| `src/llama-arch.h` | New enum `LLM_ARCH_HYV3` |
| `src/llama-arch.cpp` | Architecture name `hy_v3` |
| `src/llama-model.cpp` | Model mapping + Neox rope type |
| `src/models/models.h` | `llama_model_hyv3` class declaration |
| `src/models/hyv3.cpp` | **New** — load, forward, MTP draft head |
| `gguf-py/gguf/constants.py` | Arch enum + tensor list (28 hy_v3 tensors) |
| `gguf-py/gguf/tensor_mapping.py` | MTP tensor name mapping |
| `conversion/__init__.py` | HF → GGUF model name mapping |
| `common/chat.cpp` | Chat template parser (tool calls + reasoning) |

## License

**Apache 2.0.** Same as the original [Tencent/Hy3](https://huggingface.co/tencent/Hy3) model and [AngelSlim](https://huggingface.co/AngelSlim)'s patches.

---

**Long live open local AI. A 295B model running on a MacBook. 🎉**
