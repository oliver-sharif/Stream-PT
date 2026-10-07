# Stream-PT

[![Go Version](https://img.shields.io/badge/Go-1.27.1%20(experimental.simd)-00ADD8?style=flat&logo=go)](https://go.dev/)
[![Model](https://img.shields.io/badge/Verified%20Model-GPT--OSS--120B%20Q4__0-8A2BE2)](https://huggingface.co/unsloth/gpt-oss-120b-GGUF/tree/main/Q4_0)
[![Inference](https://img.shields.io/badge/Architecture-Disk--Streaming%20Mmap%20%2B%20SIMD-orange)](#architecture--how-it-works)
[![Status](https://img.shields.io/badge/Status-Experimental%20%2F%20Active%20WIP-yellow)](#experimental-status--breaking-changes)

> **Empowering commodity hardware to run massive 120-billion parameter models directly from SSD.**

**Stream-PT** is an experimental CPU inference engine written in Go designed to run **GPT-OSS-120B** on virtually any machine equipped with a reasonably fast solid-state drive (NVMe / SATA SSD). By reading GGUF weights on-demand through memory-mapped chunks rather than loading the entire 60+ GB model into RAM or VRAM, Stream-PT bypasses high system memory requirements and leverages Go's upcoming native SIMD vectorization for execution.

---

### ⚠️ Experimental Status & Breaking Changes

> **Note**: Stream-PT is an ongoing proof-of-concept and research project. 
> - Development is active, rapid, and subject to frequent updates.
> - **Expect breaking changes, API refactoring, and code adjustments.**
> - We welcome feedback, benchmarks, and issue reports as we push CPU streaming to its limits.

---

## ⚡ Key Features

- **Disk-to-CPU Streaming Engine**: Streams model weights layer-by-layer directly from storage via `mmap` sliding windows. You do not need 70+ GB of system RAM or enterprise GPUs to run a 120B model.
- **Go Experimental SIMD Acceleration**: Built on Go 1.27.1 with `GOEXPERIMENT=simd` for high-throughput, vectorized CPU tensor arithmetic.
- **Modern Interactive Web UI**:
  - Live token streaming via Server-Sent Events (SSE).
  - Clean, dark-mode browser interface.
  - Interactive parameter drawer: adjust Max Tokens, Temperature, Repetition Penalty, System Prompts, Top-K, Top-P, and Min-P on the fly.
  - Live system diagnostics: displays active CPU model, RAM, layer count, worker threads, and chunk window size.
  - Client-side history and local storage settings persistence.
- **MoE Optimization & Hot-Expert Caching**:
  - Configurable locked memory cache for frequently activated Mixture-of-Experts (MoE) matrices.
  - Integrated expert lookahead and prefetching to mask disk I/O latency.
- **CLI & Web Server Modes**: Run in the terminal for batch/script generation or start the web server for conversational chat.

---

## 🎯 Verified & Supported Model

Currently, Stream-PT has been developed, tested, and validated exclusively against the **Q4_0 quantization of GPT-OSS-120B**:

* **Model Repository**: [unsloth/gpt-oss-120b-GGUF (Q4_0)](https://huggingface.co/unsloth/gpt-oss-120b-GGUF/tree/main/Q4_0)
* **Required Files (shards)**:
  - `model/gpt-oss-120b-Q4_0-00001-of-00002.gguf`
  - `model/gpt-oss-120b-Q4_0-00002-of-00002.gguf`

*Other models, architectures, or quantization types are not yet officially supported or guaranteed to function.*

---

## 🚀 Quick Start

### 1. Prerequisites

- **OS**: Linux (x86_64 or ARM64)
- **Go**: Go 1.27.1+ with experimental SIMD support
- **Hardware**: Any modern multi-core CPU and an SSD with at least ~70 GB of free space.

### 2. Download the Model

Create the `model/` directory and place both GGUF shard files inside:

```sh
mkdir -p model
cd model

# Download the two Q4_0 shards from Hugging Face
wget https://huggingface.co/unsloth/gpt-oss-120b-GGUF/resolve/main/Q4_0/gpt-oss-120b-Q4_0-00001-of-00002.gguf
wget https://huggingface.co/unsloth/gpt-oss-120b-GGUF/resolve/main/Q4_0/gpt-oss-120b-Q4_0-00002-of-00002.gguf

cd ..
```

### 3. Build & Run

#### Running the Web UI (Default)

Launch the integrated chat server:

```sh
GOEXPERIMENT=simd go run .
```

Open your browser and navigate to **[http://localhost:8080](http://localhost:8080)**.

#### Running CLI Inference

To generate responses directly in the terminal, provide the prompt using the `-p` / `-prompt` flag or as positional arguments:

```sh
GOEXPERIMENT=simd go run . -p "Explain how memory mapping works in modern operating systems." -n 256
```

#### Compiling a Binary

```sh
GOEXPERIMENT=simd go build -o bin/stream-pt .
./bin/stream-pt -p "Hello world!" -n 64
```

---

## ⚙️ CLI Flags & Configuration

| Flag | Default | Description |
| :--- | :--- | :--- |
| `-addr`, `-a` | `:8080` | HTTP listen address for the web server. |
| `-prompt`, `-p` | `""` | Input prompt (switches to CLI mode when provided). |
| `-max-tokens`, `-n` | `512` | Maximum number of new tokens to generate. |
| `-threads`, `-t` | `GOMAXPROCS` | Number of worker goroutines / compute threads. |
| `-window-mb`, `-w` | `64` | Streaming mmap chunk window size in MiB. |
| `-temp` | `0.8` | Sampling temperature (`0` selects deterministic / greedy). |
| `-top-k` | `40` | Top-K candidate limit (`0` disables). |
| `-top-p` | `0.95` | Nucleus probability threshold (`1` disables). |
| `-min-p` | `0.05` | Minimum relative probability threshold (`0` disables). |
| `-repeat-penalty` | `1.0` | Repetition penalty (`1` disables). |
| `-repeat-last-n` | `64` | Token context window length to penalize. |
| `-frequency-penalty` | `0.0` | Penalty subtracted per occurrence of repeated token. |
| `-presence-penalty` | `0.0` | Penalty applied for any previously seen token. |
| `-seed` | `""` (random) | Unsigned integer seed for deterministic sampling. |
| `-expert-cache-mb` | `0` | Budget in MiB to lock hot MoE expert weights in RAM (`0` disables). |
| `-expert-cache-min-uses`| `8` | Frequency threshold before caching an expert matrix. |
| `-expert-stats` | `false` | Log frequency statistics and cache hit ratios for expert matrices. |
| `-no-expert-lookahead` | `false` | Disable background prefetch of next selected expert weights. |

---

## 🐛 Known Issues & Limitations

- **Ellipsis Sentence Abbreviation Bug**: The model occasionally cuts off output or abbreviates thoughts mid-sentence with `(...)`. This is an identified issue within the forward/tokenizer handling that is actively being debugged and worked on.
- **Fixed KV-Cache Window**: The internal KV cache currently supports a maximum sequence length of 2,048 tokens.
- **Storage Dependency**: Inference throughput is directly correlated with storage read throughput and random I/O latency. Fast NVMe drives significantly outperform standard SATA SSDs.

---

## 🔬 Architecture: How it Works

```
                +------------------------------------+
                |  GPT-OSS-120B GGUF Shards on SSD   |
                +------------------------------------+
                                  |
                                  | mmap sliding windows (e.g. 64 MiB)
                                  v
                +------------------------------------+
                |   File-Backed Memory Mapping       |
                |   (ggufmap / OS Page Cache)        |
                +------------------------------------+
                                  |
                   +--------------+---------------+
                   |                              |
                   v                              v
      +-----------------------+      +-------------------------+
      |  Hot Expert RAM Cache |      |  Lookahead / Prefetch   |
      +-----------------------+      +-------------------------+
                   |                              |
                   +--------------+---------------+
                                  |
                                  v
                +------------------------------------+
                |  CPU Vectorized Compute            |
                |  (Go experimental SIMD routines)   |
                +------------------------------------+
                                  |
                                  v
                +------------------------------------+
                |  Token Stream & SSE Web Output     |
                +------------------------------------+
```

1. **Zero-Copy Memory-Mapped Weights**: The GGUF index registers byte-offset spans for every tensor. Weights are accessed through mapped address space without allocating large Go heap buffers.
2. **MoE Expert Streaming & Prefetch**: For Mixture-of-Experts layers, only the routed experts for each token are pulled into the working set. The lookahead engine prefetches upcoming matrices while earlier computations run.
3. **Go SIMD Acceleration**: Layer computations (RMSNorm, Q4_0 / Q8_0 dot products, Attention) utilize Go's experimental SIMD intrinsics to optimize hardware instruction throughput across available CPU cores.

---

## 🤝 Contributing & Community

Contributions, optimizations, and bug fixes are welcome! As this project is in an experimental phase, please feel free to open issues or pull requests discussing architecture enhancements, SIMD optimizations, or tokenizer fixes.
