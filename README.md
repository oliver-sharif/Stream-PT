# Stream-PT

Stream-PT is an experimental, Linux-only Go inference project for a locally stored
GPT-OSS GGUF model. It keeps weights file-backed and processes transformer layers
in sequence instead of copying the entire model into the Go heap.

It provides both a command-line application and a Fyne desktop UI. The desktop
entry point is `cmd/stream-pt-ui`, its interface lives in `desktop/`, and the shared
production inference backend lives in `inference/`.

The command-line application:

- Indexes the two configured GGUF shards and reads model metadata and tensor ranges.
- Loads the vocabulary, encodes the prompt with Harmony chat framing, and reads
  token embeddings from Q4_0 weights.
- Runs RMSNorm, attention with a KV cache, RoPE/YaRN, and mixture-of-experts layers
  with GPT-OSS biases and SwiGLU.
- Uses a Q8_0 output projection and greedy argmax selection to generate tokens,
  stopping at a recognized end-of-sequence token or the requested token limit.
- Prints system/model information, generated token IDs and text, and inference time.

The kernels support Q4_0 and Q8_0 matrix multiplication, MXFP4 expert weights, and
F32/F16/BF16 normalization weights. This is not a general-purpose GGUF runtime:
other model architectures or quantization layouts may not work. The tokenizer
currently uses greedy longest-prefix vocabulary matching, not a full merge-ranked
BPE implementation. The engine allocates a KV cache for 2,048 positions, regardless
of the larger context length in model metadata.

## Requirements

- Linux: the reader uses Linux memory mapping, and mapping tests inspect `/proc`.
- Go 1.27 with the experimental SIMD API enabled via `GOEXPERIMENT=simd`.
- For inference and model-backed tests, both of these files under the repository root:

  ```text
  model/gpt-oss-120b-Q4_0-00001-of-00002.gguf
  model/gpt-oss-120b-Q4_0-00002-of-00002.gguf
  ```

The CLI currently hard-codes these paths in `main.go`; there is no model-path
flag or automatic model download. The desktop UI allows editing all shard paths.
Synthetic tests do not require model files.

## Run the command-line application

Run from the repository root so the relative model paths resolve correctly:

```sh
GOEXPERIMENT=simd go run . -prompt "Hello, reply with Yes!" -max-tokens 5 -threads 4 -window-mb 8
GOEXPERIMENT=simd go run . -h
```

To build and run the CLI instead:

```sh
mkdir -p bin
GOEXPERIMENT=simd go build -o bin/stream-pt .
./bin/stream-pt -prompt "Hello, reply with Yes!" -max-tokens 5 -threads 4 -window-mb 8
```

A CLI-only build does not require Fyne's native OpenGL/X11 development dependencies.

| Flag | Shorthand | Default | Meaning |
| --- | --- | --- | --- |
| `-prompt` | `-p` | Built-in example prompt | User text to wrap as a chat prompt; positional text is also accepted. |
| `-max-tokens` | `-n` | `5` | Maximum number of new tokens. |
| `-threads` | `-t` | `GOMAXPROCS` | Requested compute worker count; kernels may cap it. |
| `-window-mb` | `-w` | `8` | Q4_0/Q8_0 streaming window in MiB; use a positive value. |

The window is not a total RAM limit: MXFP4 computation currently maps one selected
expert matrix at a time, and activations, tokenizer data, KV cache, and the OS page
cache consume additional memory. Full 120B inference can be expensive even when
weights are file-backed. Some existing command-line status messages remain German.

## Desktop UI

The Linux desktop application uses Fyne pinned to **v2.8.1**. Building it requires
Go 1.27, `GOEXPERIMENT=simd`, `CGO_ENABLED=1`, a C compiler, and native OpenGL/X11
development dependencies. Running it requires a display server and an available
OpenGL runtime.

Install the native build dependencies on Debian/Ubuntu:

```sh
sudo apt-get install build-essential pkg-config libgl1-mesa-dev xorg-dev
```

Or on Fedora:

```sh
sudo dnf install gcc pkgconf-pkg-config mesa-libGL-devel libXcursor-devel libXrandr-devel libXinerama-devel libXi-devel libXxf86vm-devel libX11-devel
```

### Build and launch

From the repository root, with `CGO_ENABLED=1` enabled:

```sh
export CGO_ENABLED=1
mkdir -p bin
GOEXPERIMENT=simd go build -o bin/stream-pt-ui ./cmd/stream-pt-ui
./bin/stream-pt-ui
```

Run from the repository root so the default relative model paths resolve, or set
absolute model paths in the UI when launching from another working directory.
Models are not downloaded automatically.

### GoLand configuration

Create a **Go Build** run configuration with **Run kind: Package**, package
`./cmd/stream-pt-ui`, and the repository root as the working directory. Set the
environment variables to `GOEXPERIMENT=simd;CGO_ENABLED=1` and select the Go 1.27
SDK. Run this configuration in a desktop session with the native dependencies
installed; do not add the `ci` build tag to a production GUI configuration.

### Using the interface

- Choose a dark, light, or system theme and compact, standard, or large text.
- Enter **all** GGUF shard paths in the multiline model-path field. **Add GGUF
  shard** opens a file chooser and appends a path; clear the default paths first
  when switching to another model.
- Configure compute workers, the mmap window in MiB, and the maximum number of
  new tokens. The UI defaults to **128** new tokens, while the CLI defaults to **5**.
  The mmap window is not a total memory limit.
- Enter a prompt and press **Generate** or **Ctrl+Enter**. Each prompt is an
  independent request, not a conversation with prior prompts or responses.
- Watch progress/stage updates and the streamed response with auto-scroll. The
  UI reports generated token count, elapsed time, end-to-end tokens/second, and
  first-token latency. Timing includes initialization, so these are not pure
  decode-throughput measurements. Completed responses are rendered as Markdown.
  Generated images are shown as text placeholders, never fetched automatically.
- Use **Stop** to request cancellation, **Copy** to copy the response, **Export**
  to save it as a `.txt` file, and **Clear** to clear the displayed output. Use
  the reset inference settings action to restore inference defaults.

Fyne saves settings under application ID `io.streampt.studio`; prompts and
responses are not automatically persisted.

Each generation creates a fresh engine, tokenizer, and KV cache. The backend
validates the request against the 2,048-position cache before generation. Sampling
is greedy only: temperature and top-p controls are not supported.

Cancellation waits for the worker to stop before mmap resources are closed.
Closing the window cancels an active request and waits for it to finish. Shard
indexing and tokenizer initialization are not interruptible until they return,
so stopping or closing during initialization may take time.

### Troubleshooting

- Missing `gl.pc`, OpenGL headers, or X11 headers: install the native development
  packages above and ensure `pkg-config` can locate them.
- Missing `DISPLAY`, no display server, or OpenGL initialization errors: launch
  from a graphical session with a valid display connection and working GL runtime.
  The headless test driver is not a replacement for the production desktop runtime.
- SIMD source files excluded or an unknown `simd` experiment: check that the shell
  and GoLand both use the required Go 1.27 compiler with `GOEXPERIMENT=simd`.
- Missing or incompatible models: check every shard path, the working directory
  for relative paths, and the supported GPT-OSS architecture and quantization
  layouts. Clear old paths before selecting another model's complete shard set.

## Run the tests

All test files and benchmark fixtures live in `tests/`. They test the production
packages through their public APIs; no copied production kernels are used.
From the repository root:

```sh
GOEXPERIMENT=simd go test -count=1 ./tests
GOEXPERIMENT=simd go test -race ./tests
GOEXPERIMENT=simd go test ./...
```

For headless CI, use:

```sh
GOEXPERIMENT=simd go test -tags ci ./...
```

The `ci` tag selects Fyne's software driver for the desktop command package;
do not use it to build a production GUI binary. The existing `./tests` commands
need no display because UI tests use Fyne's test driver. All tests continue to
live in `tests/`; an untagged `./...` run also builds the native desktop entry
point and needs its OpenGL/X11 build dependencies.

They also run directly from the test directory:

```sh
cd tests
GOEXPERIMENT=simd go test -count=1 ./...
```

Use `-v` to see individual test results, model-inspection logs, and skip reasons.
Use `-run` to select a test, for example:

```sh
GOEXPERIMENT=simd go test ./tests -run '^TestMulQ40StreamingBatch$' -v
```

The suite covers:

- Q4_0/Q8_0/MXFP4 arithmetic and F16 decoding, including scalar reference comparisons.
- Streaming batches, worker counts, window boundaries, cancellation, invalid input,
  and output-buffer reuse.
- RMSNorm formats, in-place operation, and overlap validation.
- RoPE, attention sinks, expert biases/SwiGLU, chat framing, and EOS handling.
- Synthetic token generation that verifies generation stops before emitting EOS.
- Desktop validation, streamed UTF-8 text, cancellation, worker-safe shutdown,
  completion/error states, clipboard actions, settings persistence, rendering,
  and Markdown image privacy, using Fyne's headless test driver.
- Read-only mmap contents, alignment, range validation, and cleanup on errors.
- Local-model metadata, tokenizer round trips, embeddings, a selected Q4_0 row,
  and normalization weights.

Model-backed tests look in `../model` relative to `tests/` and skip when the required
model files are unavailable (the Q4_0 row test also skips if no suitable tensor is
found). They read metadata and selected weights, not a full-model inference pass.
The mmap-only tests can run without enabling SIMD:

```sh
go test ./tests
```

### Benchmarks

From the repository root:

```sh
GOEXPERIMENT=simd go test ./tests -run '^$' -bench '^BenchmarkMulQ40' -benchmem
```

This runs both `BenchmarkMulQ40` and `BenchmarkMulQ40BatchInto` using small synthetic
weights. `-run '^$'` selects no tests, so model-backed tests are not run as part of
this benchmark command. The results measure kernel overhead, not full-model
generation throughput.

Additional production-kernel benchmarks cover Q4_0/Q8_0/MXFP4 with 2,880 input
elements and attention with 512 cached positions:

```sh
GOEXPERIMENT=simd go test ./tests -run '^$' -bench '^(BenchmarkForward|BenchmarkMulQ40BatchInto)' -benchmem -benchtime=1s -count=3
GODEBUG=cpu.avx2=off GOEXPERIMENT=simd go test ./tests -run 'Test(QuantizedSIMDReference|AttentionSIMDReference|MulQ40StreamingBatch|LayerScratchReuse|Prefill)'
```

On an i7-4790T (AVX2), representative before/after median times from three runs
were as follows. The initial runs used 300 ms and the final runs 1 s per benchmark;
system load produced substantial outliers, so these are indicative, not guarantees.
Quantized benchmarks use one compute worker and warm synthetic weights.

| Benchmark | Before | After |
| --- | ---: | ---: |
| Q4_0, 2,880 inputs | 1.231 ms | 0.886 ms |
| Q8_0, 2,880 inputs | 1.065 ms | 0.901 ms |
| MXFP4, 2,880 inputs | 0.891 ms | 0.761 ms |
| Attention, window 128 | 1.791 ms | 1.300 ms |
| Q4_0 batch 4, 512 inputs | 0.673 ms | 0.469 ms |
| Q4_0 batch 8, 512 inputs | 1.403 ms | 0.906 ms |

Q4_0/Q8_0 use register-based byte widening on AVX2/FMA-capable amd64 CPUs,
with portable narrow-SIMD and scalar fallbacks selected by hardware capability.
MXFP4 uses a shared 16 KiB scaled-value table. Full SIMD loads avoid partial-load
overhead while tails remain supported. The engine reuses MoE and attention scratch across layers/tokens,
and the KV cache uses contiguous backing storage per layer rather than per-position
allocations. During prompt prefill, only the last token runs the output norm and
LM-head projection. Prefill processes up to 32 prompt positions layer by layer,
sharing projection weights across the batch; attention populates and reads the
KV cache in causal position order, including GQA, sinks, and sliding windows.
MoE routing groups positions by expert so each selected expert's projections
share their weight reads. Decode remains sequential.
These changes do not cache full weight tensors or establish full-model tokens/second.

`EngineOptions.PrefillBatchSize` controls activation memory versus weight reuse;
zero defaults to 32, and one selects sequential prefill. Larger values are capped
to the prompt length and KV capacity, not to SIMD width. Batch activation scratch
is reused across layers/chunks; it is additional memory beyond the mapping window
and KV cache. `Engine.Prefill` also accepts a starting position for appending a
prompt segment to an existing prefix. Engines and scratch are not concurrency-safe.
Standalone attention callers can pass `AttentionOptions.Scratch` to reuse both
scores and SIMD reduction storage; the engine does this automatically.
Q4_0/Q8_0 compute workers remain alive across all windows of a multiplication,
with a completion barrier before each window is unmapped. The one-worker path
does not launch a goroutine; workers do not survive the multiplication.

Tokenizer encoding retains greedy longest-prefix semantics (not merge-ranked
BPE), but uses a vocabulary trie rather than repeatedly creating candidate strings.
Vocabulary loading uses buffered reads and allocation-free rune counts. GGUF
fixed-width metadata arrays are skipped in one operation after validating their
element type, byte count, and file bounds; variable-width arrays still need parsing.

Additional synthetic benchmarks compare prefill batch sizes, worker/window
combinations, and startup costs:

```sh
GOEXPERIMENT=simd go test ./tests -run '^$' -bench '^(BenchmarkPrefillBatch|BenchmarkTokenizerOptimization|BenchmarkIndexOptimization|BenchmarkQuantOptimization)' -benchmem -benchtime=1s -count=3
```

On the i7-4790T, the synthetic 32-position prefill benchmark (three layers,
64 hidden elements, three experts/top-two, GQA, warm weights, one worker,
8 MiB window) measured these medians over three 1-second runs:

| Prefill batch | Time | Allocations per prompt |
| --- | ---: | ---: |
| 1 (sequential, final output projection only) | 69.54 ms | ~9,001 |
| 8 | 48.62 ms | 1,259 |
| 32 | 45.90 ms | 476 |

This is about 34% less time for this small synthetic prompt, not a measurement
of full-model tokens/second or cold-storage inference. Reused standalone
attention scratch performs no heap allocations after its initial sizing.

Keep the mapping window conservative on low-memory systems. Compare 8, 32, and
64 MiB with one, two, and four workers on the actual storage device; warm synthetic
weights cannot select an optimal HDD/SSD window or thread count for you.

## Project layout

| Path | Purpose |
| --- | --- |
| `main.go` | Command-line setup, model paths, prompt encoding, and generation. |
| `cmd/stream-pt-ui/` | Fyne desktop executable entry point. |
| `desktop/` | Desktop interface, settings, and generation controls. |
| `inference/` | Shared production inference backend used by the applications. |
| `ggufindex/` | GGUF shard indexing, tensor descriptors, and file-backed metadata. |
| `ggufmap/` | Linux read-only mmap reader and chunk/layer callbacks. |
| `forward/` | Quantized kernels, normalization, tokenizer, attention, MoE, and inference engine. |
| `tests/` | All automated tests, synthetic fixtures, and benchmarks. |
| `model/` | Local model shards used by inference and optional integration tests. |

## File-backed computation

`ggufindex.Open` retains tensor descriptors and metadata locations, not weights.
`ggufmmap.Reader.WithTensorChunks` supplies read-only, zero-copy mmap views. Each
view is unmapped before the next chunk. Callback slices must not be retained or
modified. Model files must remain immutable and must not be truncated while open;
do not call `Close` concurrently with computation.

Use the reusable-output APIs for repeated computation:

```go
x := make([]float32, int(tensor.Shape[0]))
y := make([]float32, int(tensor.Shape[1]))
options := forward.Q40Options{Workers: 1, WindowBytes: 8 << 20}
err := forward.MulQ40Into(ctx, reader, tensor, x, y, options)
// A normalization vector may be updated in place:
err = forward.RMSNormInto(ctx, reader, normWeight, x, x, epsilon)
```

Q4_0 windows are rounded down to complete rows, default to 8 MiB, and must fit
at least one encoded row. The active mapping includes at most one additional
alignment-page prefix (and page-rounded residency). No full floating-point
weight tensor is created; only one decoded 32-element block is needed at a time.
Workers process contiguous rows and join before a window is unmapped. Start with
one worker on low-memory machines; compare measured throughput before increasing it.
RMSNorm also uses bounded weight windows and SIMD-sized decoding scratch.

`MulQ40BatchInto` takes flattened, consecutive input and output vectors. It reuses
decoded blocks across SIMD-sized tiles of up to eight inputs, with bounded scratch
independent of the total batch. Larger batches require caller-owned activation
memory: approximately `4 * batch * (input + output)` bytes for these two buffers,
excluding other pipeline state. Inputs/outputs must not overlap for multiplication.
Outputs can be partially written on error or cancellation.

SIMD `Float32s.Len()` describes hardware vector lanes, **not** the maximum useful or
memory-safe token batch size. Batch prefill or independent sequences to amortize
weight reads; future tokens of one sequence depend on earlier results and cannot
be computed as an independent batch. Choose batch size from the available activation
and KV-cache budget, then benchmark it. `simd.Emulated()` and `simd.VectorBitSize()`
can distinguish emulation and vector width when reporting measurements.

## Memory and throughput limits

`mmap` avoids a Go-heap copy; it does **not** avoid physical RAM. Touched weight pages
live in the OS page cache. Windowing bounds this reader's active mapping, not the
system page cache or total process RAM. Sequential advice permits kernel readahead;
the reader deliberately does not evict globally shared file-cache pages.

When the working set exceeds RAM, repeated weight passes are storage-bandwidth
limited. A first-order ceiling is effective bytes/second divided by weight bytes
read per generated token (not necessarily the entire model for sparse MoE).
SIMD cannot remove that I/O cost. The small warm-cache kernel benchmarks are useful
for comparing compute overhead, **not** for claiming tokens/second on the 120B model.
