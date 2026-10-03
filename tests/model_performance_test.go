//go:build goexperiment.simd

package tests

import (
	"context"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"syscall"
	"testing"
	"time"

	"Stream-PT/forward"
	"Stream-PT/ggufindex"
	ggufmmap "Stream-PT/ggufmap"
)

// Opt in explicitly: this reads real model weights and can take several minutes.
func TestModelPerformance(t *testing.T) {
	if os.Getenv("STREAM_PT_MODEL_PERF") != "1" {
		t.Skip("set STREAM_PT_MODEL_PERF=1 for full-model timing")
	}
	workers := 8
	if value := os.Getenv("STREAM_PT_WORKERS"); value != "" {
		var err error
		workers, err = strconv.Atoi(value)
		if err != nil || workers <= 0 {
			t.Fatal("invalid STREAM_PT_WORKERS")
		}
	}
	paths := []string{
		filepath.Join("..", "model", "gpt-oss-120b-Q4_0-00001-of-00002.gguf"),
		filepath.Join("..", "model", "gpt-oss-120b-Q4_0-00002-of-00002.gguf"),
	}
	model, err := ggufindex.Open(paths)
	if err != nil {
		t.Fatal(err)
	}
	reader, err := ggufmmap.Open(model)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := reader.Close(); err != nil {
			t.Error(err)
		}
	}()
	engine, err := forward.NewEngineWithOptions(model, reader, forward.EngineOptions{Workers: workers, WindowBytes: 64 << 20})
	if err != nil {
		t.Fatal(err)
	}
	tokens, err := engine.Tokenizer.EncodeChatPrompt("Hallo, sag nur Ja!")
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()
	t.Logf("workers=%d GOMAXPROCS=%d window=64MiB prompt=%d tokens h=%d d=%d experts=%d topK=%d",
		workers, runtime.GOMAXPROCS(0), len(tokens), engine.Config.HiddenDim, engine.Config.ExpertHiddenDim, engine.Config.NumExperts, engine.Config.NumExpertsUsed)
	var next int
	for step := 0; step < 3; step++ {
		before := reader.ExpertCacheStats()
		var startUsage, endUsage syscall.Rusage
		if err := syscall.Getrusage(syscall.RUSAGE_SELF, &startUsage); err != nil {
			t.Fatal(err)
		}
		start := time.Now()
		if step == 0 {
			next, err = engine.Prefill(ctx, tokens, 0)
		} else {
			next, err = engine.ForwardToken(ctx, next, len(tokens)+step-1)
		}
		elapsed := time.Since(start)
		if err != nil {
			t.Fatal(err)
		}
		if err := syscall.Getrusage(syscall.RUSAGE_SELF, &endUsage); err != nil {
			t.Fatal(err)
		}
		after := reader.ExpertCacheStats()
		cpu := time.Duration(endUsage.Utime.Nano() - startUsage.Utime.Nano() + endUsage.Stime.Nano() - startUsage.Stime.Nano())
		t.Logf("step=%d time=%s cpu=%s next=%d experts=%.2fMiB ranges=%d maps=%d major_faults=%d block_input=%d prefetch=%d failures=%d",
			step, elapsed.Round(time.Millisecond), cpu.Round(time.Millisecond), next,
			float64(after.SelectedBytes-before.SelectedBytes)/(1<<20), after.SelectedRanges-before.SelectedRanges,
			after.MappedRuns-before.MappedRuns, endUsage.Majflt-startUsage.Majflt, endUsage.Inblock-startUsage.Inblock,
			after.PrefetchCalls-before.PrefetchCalls, after.PrefetchFailures-before.PrefetchFailures)
	}
}
