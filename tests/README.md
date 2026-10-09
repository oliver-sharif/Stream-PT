# Tests

All tests and benchmarks are stored in this directory:

- `*_test.go`: external integration tests against the public packages.
- `testdata/forward/`: internal inference, tokenizer, sampling and SIMD tests.
- `testdata/ggufmap/`: internal cache and lookahead tests.
- `testdata/root/`: server, browser script and performance diagnostics tests.
- `testdata/run/`: test-runner tests.
- `run/`: the centralized test runner.

## Running tests

From the project root, with Go 1.27+ and experimental SIMD enabled:

```sh
GOEXPERIMENT=simd go run ./tests/run
GOEXPERIMENT=simd go run ./tests/run ./... -count=1
GOEXPERIMENT=simd go run ./tests/run ./... -race -count=1
GOEXPERIMENT=simd go run ./tests/run ./forward -run '^TestHarmony' -count=1
GOEXPERIMENT=simd go run ./tests/run ./forward -run '^$' -bench '^BenchmarkQuantRows' -benchmem
GOEXPERIMENT=simd go run ./tests/run ./... -cover
```

Arguments are passed to `go test`, with `./...` as the default. Package paths
refer to the production packages, not to the `testdata` directories. Existing
opt-in model/performance environment variables are inherited unchanged.

Go normally requires internal tests to reside beside their implementation.
The runner maps the centralized sources into their original packages using a
temporary Go `-overlay` file. This preserves access to private functions,
build constraints, package-local helpers and the original working directories
without changing production APIs or copying tests into source directories.
The runner refuses to shadow an existing source file and removes its own
temporary overlay directory when it exits. It propagates failed test exit codes.

`testdata` is intentionally excluded from ordinary Go package discovery.
**Plain `go test ./...` does not run the internal tests.** Use the runner above
for the complete suite, including coverage and race checks. Do not pass a second
`-overlay` flag, which would replace the runner's mappings.