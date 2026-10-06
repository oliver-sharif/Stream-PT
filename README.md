# Stream-PT

## Description

Stream-PT runs GPT-OSS inference on a CPU using Go's experimental SIMD support.
GGUF weights stay file-backed and are read through memory-mapped windows rather
than copied into the Go heap. The CLI processes a chat prompt and prints generated
tokens using configurable sampling. Generation ends at an EOS token (including
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
| `-temp` | `0.8` | Sampling temperature; zero selects deterministic decoding after penalties. |
| `-top-k` | `40` | Candidate limit; zero disables. |
| `-top-p` | `0.95` | Nucleus probability threshold; one disables. |
| `-min-p` | `0.05` | Minimum probability relative to the best candidate; zero disables. |
| `-repeat-penalty` | `1.1` | Sign-aware repetition penalty; one disables. |
| `-repeat-last-n` | `64` | Recent prompt and generated tokens to penalize; zero disables, -1 uses all history. |
| `-frequency-penalty` | `0` | Subtracted per occurrence of a repeated token. |
| `-presence-penalty` | `0` | Subtracted once for a previously seen token. |
| `-seed` | random | Optional unsigned integer seed for reproducible Stream-PT sampling. |
| `-expert-cache-mb` | `0` | Locked hot-expert cache budget in MiB; zero disables retention. |
| `-expert-cache-min-uses` | `8` | Uses required before an expert matrix enters the cache. |
| `-expert-stats` | `false` | Print expert-selection and cache statistics. |
| `-no-expert-lookahead` | `false` | Disable prefetching of the next selected expert run. |

Use `GOEXPERIMENT=simd go run . -h` for command-line help. Keep the prompt and
generation within the KV-cache capacity. Memory mapping still requires physical
RAM for accessed weight pages; performance depends on CPU, RAM and storage.

Sampling follows llama.cpp's active default chain: penalties, top-k, top-p,
min-p, temperature, then a random draw from the normalized probabilities. Unlike
current llama.cpp defaults (repeat penalty 1.0), Stream-PT enables a modest 1.1
repeat penalty to reduce loops. Use `-repeat-penalty 1` for neutral penalties or
`-temp 0 -repeat-penalty 1` for legacy greedy decoding. Seeds are reproducible
within Stream-PT, not token-for-token equivalent to llama.cpp's random generator.
Only the vocabulary logits/candidates are retained; model weights remain streamed.

Without a prompt, the web server runs on `:8080`. Its settings include Temperature
and repetition penalty, saved in browser local storage. CLI sampling flags set
server defaults, exposed as `default_sampling` by `/api/info`. `/api/chat` accepts
optional `temperature`, `top_k`, `top_p`, `min_p`, `repeat_penalty`, `repeat_last_n`,
`frequency_penalty`, `presence_penalty`, and `seed` fields. Omitted fields use server
defaults; explicit zero values are preserved where supported. Invalid parameters
return HTTP 400 before streaming. Each request has its own sampling history/RNG.
Low-level `ForwardToken` and `Prefill` remain greedy; `Generate` uses configured
defaults, and `GenerateWithSampling` accepts explicit per-generation parameters.
