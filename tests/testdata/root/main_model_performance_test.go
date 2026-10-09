//go:build goexperiment.simd

package main

import (
	"context"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"testing"

	"Stream-PT/forward"
	"Stream-PT/ggufindex"
	ggufmmap "Stream-PT/ggufmap"
)

// Opt in: isolates decode from chat-template prefill using one real vocabulary
// token. This is a hardware measurement, not an answer-quality benchmark.
func TestModelCacheDecodePerformance(t *testing.T) {
	if os.Getenv("STREAM_PT_CACHE_PERF") != "1" {
		t.Skip("set STREAM_PT_CACHE_PERF=1 for real-model cache/decode comparison")
	}
	old := runtime.GOMAXPROCS(4)
	defer runtime.GOMAXPROCS(old)
	model, err := ggufindex.Open([]string{
		filepath.Join("model", "gpt-oss-120b-Q4_0-00001-of-00002.gguf"),
		filepath.Join("model", "gpt-oss-120b-Q4_0-00002-of-00002.gguf"),
	})
	if err != nil {
		t.Fatal(err)
	}
	var reference []int
	for _, budget := range []uint64{0, 2 << 30} {
		func() {
			reader, err := ggufmmap.Open(model)
			if err != nil {
				t.Fatal(err)
			}
			defer func() {
				if err := reader.Close(); err != nil {
					t.Error(err)
				}
			}()
			if err := reader.ConfigureExpertCache(ggufmmap.ExpertCacheOptions{MaxBytes: budget, MinUses: 2, TrackStats: true}); err != nil {
				t.Fatal(err)
			}
			if err := reader.ConfigureResidentTensors(residentWeightCandidates(model), budget/2); err != nil {
				t.Fatal(err)
			}
			engine, err := forward.NewEngineWithOptions(model, reader, forward.EngineOptions{Workers: 4, WindowBytes: 64 << 20})
			if err != nil {
				t.Fatal(err)
			}
			prompt := engine.Tokenizer.Encode("Hello")
			if len(prompt) == 0 {
				t.Fatal("empty encoded prompt")
			}
			prompt = prompt[:1]
			options := forward.DefaultSamplingOptions()
			options.Temperature = 0
			before := readProcessIO()
			result, err := engine.GenerateWithSamplingResult(context.Background(), prompt, 4, options, nil)
			if err != nil {
				t.Fatal(err)
			}
			if result.Metrics.DecodeSteps == 0 {
				t.Fatal("no decode steps measured")
			}
			if reference == nil {
				reference = slices.Clone(result.Tokens)
			} else if !slices.Equal(reference, result.Tokens) {
				t.Fatalf("cache changed greedy tokens: %v vs %v", reference, result.Tokens)
			}
			stats := reader.ExpertCacheStats()
			p := inferencePerformance(result.Metrics, 0, before, readProcessIO())
			t.Logf("cache=%dMiB tokens=%v prefill=%.3fs decode=%.3fs steps=%d decode_tokens/s=%.3f total=%.3fs reads=%.2fMiB major_faults=%d retained=%.2fMiB pinned=%.2fMiB expert_hits=%d resident_hits=%d lock_failures=%d", budget>>20, result.Tokens, p.PrefillSeconds, p.DecodeSeconds, p.DecodeSteps, p.DecodeTokensPerSec, p.TotalSeconds, float64(p.ReadBytes)/(1<<20), p.MajorFaults, float64(stats.RetainedBytes)/(1<<20), float64(stats.PinnedBytes)/(1<<20), stats.CacheHits, stats.ResidentHits, stats.LockFailures)
		}()
	}
}
