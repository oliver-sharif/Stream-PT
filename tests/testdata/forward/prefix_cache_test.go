//go:build goexperiment.simd

package forward

import (
	"context"
	"fmt"
	"slices"
	"testing"
)

func TestGenerationPrefixTransformerParity(t *testing.T) {
	for _, batch := range []int{1, 4} {
		t.Run(fmt.Sprint(batch), func(t *testing.T) {
			warm, cold := batchEngineFixture(t), batchEngineFixture(t)
			warm.Options.PrefillBatchSize, cold.Options.PrefillBatchSize = batch, batch
			options := DefaultSamplingOptions()
			options.Temperature = 0
			ctx := context.Background()
			_, err := warm.GenerateWithSessionResult(ctx, "a", []int{0, 1, 2, 3}, 3, options, nil)
			if err != nil {
				t.Fatal(err)
			}
			for _, prompt := range [][]int{{0, 1, 2, 3, 0}, {0, 1}, {0, 2, 1, 0}} {
				got, err := warm.GenerateWithSessionResult(ctx, "a", prompt, 3, options, nil)
				if err != nil {
					t.Fatal(err)
				}
				want, err := cold.GenerateWithSamplingResult(ctx, prompt, 3, options, nil)
				if err != nil {
					t.Fatal(err)
				}
				if !slices.Equal(got.Tokens, want.Tokens) || got.Metrics.ReusedPromptTokens == 0 {
					t.Fatalf("prompt %v warm %+v cold %+v", prompt, got, want)
				}
				compareVectors(t, warm.X, cold.X)
				for layer := range warm.Layers {
					for pos := 0; pos < len(prompt)+2; pos++ {
						compareVectors(t, warm.KVCache.Keys[layer][pos], cold.KVCache.Keys[layer][pos])
						compareVectors(t, warm.KVCache.Values[layer][pos], cold.KVCache.Values[layer][pos])
					}
				}
			}
		})
	}
}

func BenchmarkGenerationPrefix(b *testing.B) {
	for _, reuse := range []bool{false, true} {
		b.Run(fmt.Sprint(reuse), func(b *testing.B) {
			engine := batchEngineFixture(b)
			prompt := make([]int, 24)
			for i := range prompt {
				prompt[i] = i % 4
			}
			options := DefaultSamplingOptions()
			options.Temperature = 0
			session := ""
			if reuse {
				session = "benchmark"
			}
			if _, err := engine.GenerateWithSessionResult(context.Background(), session, prompt, 1, options, nil); err != nil {
				b.Fatal(err)
			}
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, err := engine.GenerateWithSessionResult(context.Background(), session, prompt, 1, options, nil); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func TestGenerationPrefixReuse(t *testing.T) {
	options := DefaultSamplingOptions()
	options.Temperature = 0
	engine := streamTestEngine(t, []string{"prompt", "one", "two", "three", "<|return|>"})
	engine.Options.PrefillBatchSize = 1
	run := func(session string, prompt []int) GenerationResult {
		t.Helper()
		got, err := engine.GenerateWithSessionResult(context.Background(), session, prompt, 2, options, nil)
		if err != nil {
			t.Fatal(err)
		}
		return got
	}
	first := run("a", []int{0})
	if !slices.Equal(first.Tokens, []int{1, 2}) {
		t.Fatalf("first %+v", first)
	}
	warm := run("a", []int{0, 1, 2})
	if warm.Metrics.ReusedPromptTokens != 2 || !slices.Equal(warm.Tokens, []int{3}) {
		t.Fatalf("warm %+v", warm)
	}
	if got := run("a", []int{0, 1, 2}); got.Metrics.ReusedPromptTokens != 2 || !slices.Equal(got.Tokens, warm.Tokens) {
		t.Fatalf("repeat %+v", got)
	}
	if got := run("b", []int{0, 1, 2}); got.Metrics.ReusedPromptTokens != 0 {
		t.Fatalf("cross-session reuse %+v", got)
	}
	if got := run("b", []int{0, 2}); got.Metrics.ReusedPromptTokens != 1 || !slices.Equal(got.Tokens, []int{3}) {
		t.Fatalf("edited prefix %+v", got)
	}
	if got := run("", []int{0, 2}); got.Metrics.ReusedPromptTokens != 0 {
		t.Fatalf("stateless reuse %+v", got)
	}
}

func TestGenerationPrefixInvalidation(t *testing.T) {
	options := DefaultSamplingOptions()
	options.Temperature = 0
	for _, action := range []string{"prefill", "forward", "cancel", "error"} {
		t.Run(action, func(t *testing.T) {
			engine := streamTestEngine(t, []string{"prompt", "one", "two", "<|return|>"})
			engine.Options.PrefillBatchSize = 1
			_, err := engine.GenerateWithSessionResult(context.Background(), "a", []int{0, 1}, 1, options, nil)
			if err != nil {
				t.Fatal(err)
			}
			switch action {
			case "prefill":
				_, err = engine.Prefill(context.Background(), []int{2}, 0)
			case "forward":
				_, err = engine.ForwardToken(context.Background(), 2, 0)
			case "cancel":
				ctx, cancel := context.WithCancel(context.Background())
				cancel()
				_, err = engine.GenerateWithSessionResult(ctx, "a", []int{0, 1}, 1, options, nil)
			case "error":
				_, err = engine.GenerateWithSessionResult(context.Background(), "a", []int{0, 99}, 1, options, nil)
			}
			if action == "cancel" || action == "error" {
				if err == nil {
					t.Fatal("expected error")
				}
			} else if err != nil {
				t.Fatal(err)
			}
			got, err := engine.GenerateWithSessionResult(context.Background(), "a", []int{0, 1}, 1, options, nil)
			if err != nil || got.Metrics.ReusedPromptTokens != 0 {
				t.Fatalf("after mutation %+v, %v", got, err)
			}
		})
	}
}
