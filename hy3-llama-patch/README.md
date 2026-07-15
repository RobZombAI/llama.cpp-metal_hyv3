# Hy3 (hy_v3) per llama.cpp — supporto macOS Metal

Patch e binari per eseguire **Tencent Hy3 (hy_v3) 295B MoE** su **llama.cpp** con **GPU Apple Silicon (Metal)**.

## Il modello

| Dettaglio | Valore |
|-----------|--------|
| **Architettura** | Hy3 (hy_v3) — Hunyuan V3 |
| **Parametri** | 295B |
| **Layer** | 81 (80 routed + 1 MTP) |
| **Esperti** | 192 (8 attivi per token) |
| **Gating** | Sigmoid + correction bias + top-8 |
| **Quantizzazione** | IQ1_M (ricetta mista AngelSlim) |

> Modello originale: [Tencent/Hy3](https://huggingface.co/tencent/Hy3)
> Quantizzazione GGUF: **[AngelSlim/Hy3-GGUF](https://huggingface.co/AngelSlim/Hy3-GGUF)** ← scarica il .gguf da qui

**Perdita di qualità IQ1_M vs BF16**: ~+0.3% PPL — non percepibile. I benchmark sono nella pagina HF di AngelSlim (`assets/benchmark.png`).

## Download

### 1. Il modello GGUF

```
# IQ1_M (85 GB) — ci sta su Mac 98-128 GB
wget https://huggingface.co/AngelSlim/Hy3-GGUF/resolve/main/Hy3-IQ1_M-mtp.gguf

# Alternativa: IQ1_M senza MTP (84 GB, leggermente più piccolo)
wget https://huggingface.co/AngelSlim/Hy3-GGUF/resolve/main/Hy3-IQ1_M.gguf
```

### 2. Il codice (repository GitHub)

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
git checkout 19bba67c1
git apply /path/to/0001-add-hyv3-support.patch
cp /path/to/hyv3.cpp src/models/
```

Oppure clona direttamente il fork già pronto:

```bash
git clone https://github.com/<tuo-utente>/llama.cpp-hyv3
cd llama.cpp-hyv3
```

### 3. Compilazione

```bash
mkdir build && cd build
cmake .. -DLLAMA_METAL=ON
make -j$(sysctl -n hw.logicalcpu)
```

## Comandi

### CLI (inferenza base)

```bash
./build/bin/llama-cli \
    -m ~/Downloads/Hy3-IQ1_M-mtp.gguf \
    -c 65536 \
    -ngl 99 \
    -fa on \
    -ctk q8_0 -ctv q8_0 \
    -p "Ciao" \
    -n 100 \
    --temp 0.6
```

### CLI (con ragionamento/pensiero)

```bash
./build/bin/llama-cli \
    -m ~/Downloads/Hy3-IQ1_M-mtp.gguf \
    -c 65536 \
    -ngl 99 -fa on \
    -ctk q8_0 -ctv q8_0 \
    -p "Ciao" \
    -n 200 \
    --temp 0.6 \
    --reasoning on \
    --reasoning-budget -1
```

### CLI (con MTP self-speculative)

```bash
./build/bin/llama-cli \
    -m ~/Downloads/Hy3-IQ1_M-mtp.gguf \
    -c 65536 \
    -ngl 99 -fa on \
    --spec-type draft-mtp \
    --spec-draft-n-max 3 \
    --spec-draft-n-min 1 \
    -ctk q8_0 -ctv q8_0 \
    -ctkd q8_0 -ctvd q8_0 \
    -p "Ciao" \
    -n 200 \
    --temp 0.6
```

### Server (API OpenAI-compatibile)

```bash
./build/bin/llama-server \
    -m ~/Downloads/Hy3-IQ1_M-mtp.gguf \
    -c 65536 \
    -ngl 99 -fa on \
    -ctk q8_0 -ctv q8_0 \
    --temp 0.6 \
    --port 8080
```

Chiamata API:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Hy3-IQ1_M-mtp",
    "messages": [{"role": "user", "content": "Ciao"}],
    "temperature": 0.6,
    "max_tokens": 200
  }'
```

### Server (con ragionamento + MTP)

```bash
./build/bin/llama-server \
    -m ~/Downloads/Hy3-IQ1_M-mtp.gguf \
    -c 65536 \
    -ngl 99 -fa on \
    --spec-type draft-mtp \
    --spec-draft-n-max 3 \
    --spec-draft-n-min 1 \
    -ctk q8_0 -ctv q8_0 \
    -ctkd q8_0 -ctvd q8_0 \
    --temp 0.6 \
    --reasoning on \
    --reasoning-budget -1 \
    --port 8080
```

## Spiegazione flag

| Flag | Cosa fa |
|------|---------|
| `-m` | Path del modello GGUF |
| `-c N` | Contesto (max token in memoria). `65536` = 64K |
| `-ngl N` | Layer su GPU (`99` = tutti su Metal) |
| `-fa on` | Flash attention — risparmia memoria GPU |
| `-ctk q8_0 -ctv q8_0` | KV cache compressa in q8_0 (risparmia ~20 GB) |
| `--temp N` | Temperatura di campionamento |
| `--reasoning on` | Abilita pensiero/ragionamento (tag) |
| `--reasoning-budget N` | Budget token per il pensiero (`-1` = illimitato) |
| `--spec-type draft-mtp` | MTP self-speculative decoding |
| `--spec-draft-n-max N` | Max draft token per passo MTP |

## Requisiti hardware

| Mac | RAM | Modello | MTP | Contesto |
|-----|-----|---------|-----|----------|
| M5 Max | 128 GB | IQ1_M | ✅ | 64K |
| M4 Max | 128 GB | IQ1_M | ✅ | 64K |
| M3 Max | 128 GB | IQ1_M | ✅ | 64K |
| Mac Studio | 96 GB | IQ1_M | ❌ | 64K (KV q8_0) |

> Con 96 GB: niente MTP, tieni `-c 65536 -ctk q8_0 -ctv q8_0` e togli `--spec-type draft-mtp`.

## Struttura del progetto

```
├── 0001-add-hyv3-support.patch   # Patch per 9 file di llama.cpp
├── src/models/hyv3.cpp           # Implementazione modello hy_v3 (388 righe)
└── README.md                     # Questo file
```

## File modificati in llama.cpp

| File | Modifica |
|------|----------|
| `src/llama-arch.h` | Nuovo enum `LLM_ARCH_HYV3` |
| `src/llama-arch.cpp` | Nome architettura `hy_v3` |
| `src/llama-model.cpp` | Model mapping + rope type Neox |
| `src/models/models.h` | Dichiarazione classe `llama_model_hyv3` |
| `src/models/hyv3.cpp` | **Nuovo** — load, forward, MTP draft head |
| `gguf-py/gguf/constants.py` | Arch enum + tensor list |
| `gguf-py/gguf/tensor_mapping.py` | MTP tensor name mapping |
| `gguf-py/gguf/__init__.py` | Version bump |
| `conversion/__init__.py` | HF → GGUF model name mapping |
| `common/chat.cpp` | Chat template (tool calls + reasoning) |

## Licenza

Apache 2.0 (come il modello originale Tencent/Hy3 e le patch AngelSlim).

**W l'open local AI.** 🎉
