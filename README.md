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
  - Live reasoning in a separate, expanded collapsible panel, with token progress and elapsed time even before the final answer appears. Only the final answer is added to chat history.
  - Clean, dark-mode browser interface.
  - Interactive parameter drawer: adjust Max Tokens, Temperature, Repetition Penalty, System Prompts, Top-K, Top-P, and Min-P on the fly.
  - Live system diagnostics: displays active CPU model, RAM, layer count, worker threads, and chunk window size.
  - Client-side history and local storage settings persistence.
- **MoE Optimization & Hot-Expert Caching**:
  - Shared, bounded weight cache enabled by default with a 2 GiB target; automatic defaults are reduced on memory-constrained systems.
  - Hot-expert eviction and best-effort memory locking, with reusable file-backed mappings when locking is unavailable.
  - Integrated expert lookahead and prefetching to mask disk I/O latency.
- **Conversation Prefix Reuse**: Exact token-prefix KV reuse for the most recent browser session, with bounded memory and invalidation after errors or direct forward calls.
- **Performance Diagnostics**: Separate prompt processing, decode compute throughput, first-answer latency, process storage reads, and major page faults.
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

- **OS**: Linux (x86_64 or ARM64) — *Currently only Linux is supported.*
- **Go**: Go 1.27.1+ with experimental SIMD support
- **Hardware**: Any modern multi-core CPU and an SSD with at least ~70 GB of free space.

> **Reference Development Hardware**:
> Stream-PT was specifically developed and tested on modest commodity hardware:
> ```text
> CPU: Intel(R) Core(TM) i7-4790T CPU @ 2.70GHz
> RAM: 15.5 GiB
> ```

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
| `-model-dir` | `model` | Directory containing both model shards. |
| `-prefill-batch` | `32` | Prompt positions processed per weight pass (`1` selects token-wise processing). |
| `-temp` | `0.8` | Sampling temperature (`0` selects deterministic / greedy). |
| `-top-k` | `40` | Top-K candidate limit (`0` disables). |
| `-top-p` | `0.95` | Nucleus probability threshold (`1` disables). |
| `-min-p` | `0.05` | Minimum relative probability threshold (`0` disables). |
| `-repeat-penalty` | `1.0` | Repetition penalty (`1` disables). |
| `-repeat-last-n` | `64` | Token context window length to penalize. |
| `-frequency-penalty` | `0.0` | Penalty subtracted per occurrence of repeated token. |
| `-presence-penalty` | `0.0` | Penalty applied for any previously seen token. |
| `-seed` | `""` (random) | Unsigned integer seed for deterministic sampling. |
| `-expert-cache-mb` | `2048` | Shared weight-cache budget in MiB (`0` disables). The automatic default is capped at a quarter of available RAM / cgroup v2 headroom; an explicit value overrides this cap. |
| `-resident-cache-mb` | `1024` | Maximum basis-weight reservation, capped at half the shared budget (`0` disables). Prioritizes output projection, norms, attention and routers; only tensors that fit are selected. |
| `-expert-cache-min-uses`| `2` | Frequency threshold before caching an expert matrix. |
| `-expert-stats` | `false` | Log frequency statistics and cache hit ratios for expert matrices. |
| `-no-expert-lookahead` | `false` | Disable background prefetch of next selected expert weights. |
| `-expert-lookahead-mb` | `32` | Total page-footprint limit for prefetched expert runs, excluding the current run. |
| `-expert-lookahead-runs` | `1` | Number of future expert runs prepared ahead (`1–8`), within the shared lookahead byte limit; no additional workers. |
| `-prefetch-mb` | `0` | Readahead coverage per newly mapped expert run in MiB (`0` covers the complete selected mapping). Small caps can lower SSD throughput under `MADV_RANDOM`. |
| `-no-prefetch` | `false` | Disable explicit `MADV_WILLNEED` hints for comparison. |

### Performance tuning and measurement

The shared cache keeps weights **quantized and file-backed**. Half of its budget
is available for basis weights by default; only selected basis tensors actually
reserve space, leaving the remainder for hot experts. Basis tensors are mapped
lazily and retained; experts use aged LFU with LRU tie-breaking and cannot be
evicted during an active callback. `mlock` is best effort: without sufficient
permissions, mappings still remain reusable but Linux can reclaim their pages.
The budget does not include transient mappings, activations, KV storage or the
OS page cache. Model files must remain unchanged while a reader is open, since
file sizes are validated once at opening rather than through per-range `stat`.

Normal startup enables the cache automatically:

```sh
GOEXPERIMENT=simd go run .
```

For comparable model measurements, build once and compare the same prompt,
thread count, token budget and greedy sampling. Retention disabled is a control,
**not** an OS page-cache flush:

```sh
GOEXPERIMENT=simd go build -o bin/stream-pt .
./bin/stream-pt -p "Explain memory mapping in detail." -n 128 -temp 0 -expert-cache-mb 2048 -expert-stats
./bin/stream-pt -p "Explain memory mapping in detail." -n 128 -temp 0 -expert-cache-mb 0 -expert-stats
```

Vary `-t`, `-prefill-batch`, `-prefetch-mb` and lookahead limits separately;
more parallelism/readahead is not always faster. Inspect **decode tokens/s**,
prefill duration, first-answer latency, process storage reads and major faults.
Decode throughput measures forward-and-sampling steps, excluding prompt work
and stream delivery; it includes analysis generation. One generated token has
already been sampled by prefill, so short runs with no decode step report zero
decode throughput rather than an artificial speed. Web completions expose these
measurements under SSE `performance`, and the UI retains the decode/prompt/KV
summary. `/api/info` and the UI show retained versus pinned cache bytes.

The browser sends a random request `session_id`. Only the last session's exact
token prefix is retained; switching sessions performs a cold prefill. Clients
without `session_id` stay stateless. The last prompt position is recomputed to
restore its output activation. Edited prefixes, cancelled/error requests and
direct `ForwardToken`/`Prefill` calls cannot silently reuse stale state. No
unbounded cache per user or permanent worker pool is introduced.

The Q8_0 output projection uses a numerically checked AVX2/FMA row-pair kernel
with shared activation loads and portable fallbacks. Kernel microbenchmarks are
not end-to-end inference measurements. **1 token/s is a target, not a guarantee**;
the actual 120B shards, SSD, cache hit rate and CPU determine achievable speed.

For a short, opt-in real-model cache comparison with one input vocabulary token:

```sh
GOEXPERIMENT=simd STREAM_PT_CACHE_PERF=1 go run ./tests/run . -run '^TestModelCacheDecodePerformance$' -count=1 -v -timeout=5m
```

This measures decode rather than chat answer quality, uses four workers and
checks identical greedy token output with retention disabled/enabled. The OS
page cache is not flushed. On the reference i7-4790T, a short run measured
`0.248` decode tokens/s without retention and `0.271` with 2 GiB retention.
A normal 73-token chat prompt measured `117.5 s` prefill and `0.240` decode
tokens/s. These are observations, not stable speedup guarantees. Full selected
expert prefetch is the default: a 1-MiB coverage cap produced many synchronous
page faults and only `0.043` decode tokens/s in the earlier measurement.

### Tests

All test sources and benchmarks live under [`tests/`](tests/README.md).
Run the complete suite from the project root:

```sh
GOEXPERIMENT=simd go run ./tests/run ./... -count=1
GOEXPERIMENT=simd go run ./tests/run ./... -race -count=1
```

The runner uses Go's `-overlay` support to compile internal tests in their
original packages without copying files back or exposing private implementation
details. Plain `go test ./...` only discovers the external tests directly in
`tests/`; use the runner for full coverage. All standard test flags and package
selectors are supported. See the test guide for targeted runs and benchmarks.

---

## 🐛 Known Issues & Limitations

- **Linux-Only Support**: Stream-PT currently runs exclusively on Linux (x86_64 / ARM64). Support for macOS and Windows is not yet available.
- **Harmony Completion Handling**: Message-end tokens no longer stop GPT-OSS generation before its final answer. Chat prompts use an open assistant header, and streaming preserves split UTF-8 characters while hiding Harmony headers. The web UI streams analysis separately through SSE `reasoning` events, alongside live `generated_tokens` progress. SSE completion events report `finish_reason` and, for model stops, `stop_token`; token/context limits are not reported as successful model completion. Tool handoffs are reported but tools are not executed.
- **Ellipsis Output Fix**: The premature Harmony-stop and UTF-8 streaming defects have regression coverage, and the original reporter confirmed that the unwanted `(...)` output disappeared. Literal ellipsis tokens are not suppressed.
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
