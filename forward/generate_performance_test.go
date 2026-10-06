//go:build goexperiment.simd && linux

package forward

import (
	"context"
	"encoding/binary"
	"hash/fnv"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strconv"
	"syscall"
	"testing"
	"time"

	"Stream-PT/ggufindex"
	ggufmmap "Stream-PT/ggufmap"
)

// Real Generate selected-run lookahead A/B, opt-in because it reads many GiB of weights.
// Fresh engines/readers; OS page cache is deliberately not flushed.
func TestGenerateModelPerformance(t *testing.T) {
	if os.Getenv("STREAM_PT_GENERATE_PERF") != "1" {
		t.Skip("set STREAM_PT_GENERATE_PERF=1 for real Generate A/B")
	}
	integer := func(name string, fallback int) int {
		if value := os.Getenv(name); value != "" {
			n, err := strconv.Atoi(value)
			if err != nil || n <= 0 {
				t.Fatalf("invalid %s", name)
			}
			return n
		}
		return fallback
	}
	workers := integer("STREAM_PT_WORKERS", 4)
	repeats := integer("STREAM_PT_PERF_REPEATS", 5)
	limit := integer("STREAM_PT_PERF_TOKENS", 5)
	prompt := os.Getenv("STREAM_PT_PERF_PROMPT")
	if prompt == "" {
		prompt = "Hallo, sag nur Ja!"
	}
	old := runtime.GOMAXPROCS(workers)
	defer runtime.GOMAXPROCS(old)
	model, err := ggufindex.Open([]string{
		filepath.Join("..", "model", "gpt-oss-120b-Q4_0-00001-of-00002.gguf"),
		filepath.Join("..", "model", "gpt-oss-120b-Q4_0-00002-of-00002.gguf"),
	})
	if err != nil {
		t.Fatal(err)
	}
	// Match main's context: a timed context adds a mutex to every row's Err check.
	// The go test -timeout flag bounds the full workload instead.
	ctx := context.Background()
	var reference []int
	var referenceState []float32
	var referenceText string
	var referenceKV uint64
	var referenceIO ggufmmap.ExpertCacheStats
	var totals [2][]time.Duration
	variants := []string{"baseline", "optimized"}
	for round := range repeats + 1 {
		for turn := range len(variants) {
			variant := (turn + round) % len(variants)
			name := variants[variant]
			func() {
				r, err := ggufmmap.Open(model)
				if err != nil {
					t.Fatal(err)
				}
				defer func() {
					if err := r.Close(); err != nil {
						t.Error(err)
					}
				}()
				if err := r.ConfigureExpertLookahead(name == "optimized"); err != nil {
					t.Fatal(err)
				}
				startInit := time.Now()
				e, err := NewEngineWithOptions(model, r, EngineOptions{
					Workers: workers, WindowBytes: 64 << 20,
				})
				if err != nil {
					t.Fatal(err)
				}
				initTime := time.Since(startInit)
				tokens, err := e.Tokenizer.EncodeChatPrompt(prompt)
				if err != nil {
					t.Fatal(err)
				}
				var before, after syscall.Rusage
				if err := syscall.Getrusage(syscall.RUSAGE_SELF, &before); err != nil {
					t.Fatal(err)
				}
				start := time.Now()
				var first time.Duration
				var emitted []int
				var text string
				got, err := e.Generate(ctx, tokens, limit, func(id int, piece string) bool {
					if len(emitted) == 0 {
						first = time.Since(start)
					}
					emitted = append(emitted, id)
					text += piece
					return true
				})
				total := time.Since(start)
				if err != nil {
					t.Fatal(err)
				}
				if err := syscall.Getrusage(syscall.RUSAGE_SELF, &after); err != nil {
					t.Fatal(err)
				}
				stats := r.ExpertCacheStats()
				if stats.PrefetchFailures != 0 || stats.CachedBytes != 0 || stats.CacheHits != 0 {
					t.Fatalf("unexpected prefetch failure or retained weights: %+v", stats)
				}
				if reference == nil {
					referenceIO = stats
				} else if stats.SelectedRanges != referenceIO.SelectedRanges || stats.SelectedBytes != referenceIO.SelectedBytes || stats.MappedRuns != referenceIO.MappedRuns || stats.MappedBytes != referenceIO.MappedBytes || stats.PrefetchCalls != referenceIO.PrefetchCalls {
					t.Fatalf("different expert I/O schedule: got %+v want %+v", stats, referenceIO)
				}
				if !slices.Equal(got, emitted) {
					t.Fatal("callback tokens differ")
				}
				// Hash only populated positions, outside the timed interval.
				h := fnv.New64a()
				var bits [4]byte
				for layer := range e.KVCache.Keys {
					for pos := 0; pos < min(e.KVCache.MaxPos, len(tokens)+len(got)); pos++ {
						for _, row := range [][]float32{e.KVCache.Keys[layer][pos], e.KVCache.Values[layer][pos]} {
							for _, value := range row {
								binary.LittleEndian.PutUint32(bits[:], math.Float32bits(value))
								_, _ = h.Write(bits[:])
							}
						}
					}
				}
				if reference == nil {
					reference, referenceState, referenceText = slices.Clone(got), slices.Clone(e.X), text
					referenceKV = h.Sum64()
				} else if !slices.Equal(got, reference) || !slices.Equal(e.X, referenceState) || text != referenceText || h.Sum64() != referenceKV {
					t.Fatalf("%s: different tokens/text/state: got %v want %v", name, got, reference)
				}
				if round > 0 && variant < 2 {
					totals[variant] = append(totals[variant], total)
				}
				cpu := time.Duration(after.Utime.Nano() - before.Utime.Nano() + after.Stime.Nano() - before.Stime.Nano())
				t.Logf("round=%d variant=%s workers=%d prompt=%d limit=%d emitted=%d tokens=%v text=%q init=%s total=%s first=%s after_first=%s cpu=%s major_faults=%d block_input=%d",
					round, name, workers, len(tokens), limit, len(got), got, text, initTime.Round(time.Millisecond), total.Round(time.Millisecond), first.Round(time.Millisecond), (total - first).Round(time.Millisecond), cpu.Round(time.Millisecond), after.Majflt-before.Majflt, after.Inblock-before.Inblock)
			}()
		}
	}
	median := func(v []time.Duration) time.Duration {
		slices.Sort(v)
		if len(v)%2 == 0 {
			return (v[len(v)/2-1] + v[len(v)/2]) / 2
		}
		return v[len(v)/2]
	}
	a, b := median(totals[0]), median(totals[1])
	gain := 100 * (1 - float64(b)/float64(a))
	t.Logf("MEDIAN baseline=%s optimized=%s reduction=%.2f%% baseline_range=[%s,%s] optimized_range=[%s,%s] (warmup excluded, OS page cache uncontrolled)", a, b, gain, totals[0][0], totals[0][len(totals[0])-1], totals[1][0], totals[1][len(totals[1])-1])
	if gain < 10 {
		t.Log("10% target NOT reached; do not claim a 10% speedup")
		if os.Getenv("STREAM_PT_PERF_REQUIRE_GAIN") == "1" {
			t.Fatalf("required 10%% reduction, measured %.2f%%", gain)
		}
	}
}
