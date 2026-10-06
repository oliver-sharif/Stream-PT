# Stream-PT

## Description

Stream-PT runs GPT-OSS inference on a CPU using Go's experimental SIMD support.
GGUF weights stay file-backed and are read through memory-mapped windows rather
than copied into the Go heap. The CLI processes a chat prompt and prints generated
tokens using greedy decoding. Generation ends at an EOS token (including
`<|end|>`) or the token limit. The KV cache has 2,048 positions.

## Usage

Requires Linux and Go 1.27 with `GOEXPERIMENT=simd`. Place both model shards in
`model/`:

```text
model/gpt-oss-120b-Q4_0-00001-of-00002.gguf
model/gpt-oss-120b-Q4_0-00002-of-00002.gguf
```

Run from the project root:

```sh
GOEXPERIMENT=simd go run . -p "Write a small hello-world program in Python" -n 200
```

Alternatively, build and run:

```sh
GOEXPERIMENT=simd go build -o bin/stream-pt .
./bin/stream-pt -p "Hello" -n 32
```

| Option | Default | Purpose |
| --- | --- | --- |
| `-prompt`, `-p` | `Hallo, sag nur Ja!` | User prompt; positional text is also accepted. |
| `-max-tokens`, `-n` | `5` | Maximum number of generated tokens; nonpositive values select 64. |
| `-threads`, `-t` | `GOMAXPROCS` | Compute worker count. |
| `-window-mb`, `-w` | `64` | Streaming weight window in MiB; zero selects the engine default of 8 MiB. |
| `-expert-cache-mb` | `0` | Locked hot-expert cache budget in MiB; zero disables retention. |
| `-expert-cache-min-uses` | `8` | Uses required before an expert matrix enters the cache. |
| `-expert-stats` | `false` | Print expert-selection and cache statistics. |
| `-no-expert-lookahead` | `false` | Disable prefetching of the next selected expert run. |

Use `GOEXPERIMENT=simd go run . -h` for command-line help. Keep the prompt and
generation within the KV-cache capacity. Memory mapping still requires physical
RAM for accessed weight pages; performance depends on CPU, RAM and storage.
