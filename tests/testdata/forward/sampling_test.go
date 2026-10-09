//go:build goexperiment.simd

package forward

import (
	"context"
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"Stream-PT/ggufindex"
	ggufmmap "Stream-PT/ggufmap"
)

func samplingTestOptions() SamplingOptions {
	seed := uint64(42)
	return SamplingOptions{Temperature: 1, TopP: 1, RepeatPenalty: 1, RepeatLastN: 64, Seed: &seed}
}

func TestSamplingDefaultsPreserveRepeatedCodeTokens(t *testing.T) {
	o := DefaultSamplingOptions()
	if o.RepeatPenalty != 1 {
		t.Fatalf("default repeat penalty = %g, want llama.cpp's neutral 1", o.RepeatPenalty)
	}
	s, err := newTokenSampler(o, []int{0, 1, 0})
	if err != nil {
		t.Fatal(err)
	}
	for _, logit := range []float64{-10, 0, 10} {
		if got := s.penalize(0, logit); got != logit {
			t.Fatalf("default penalty changes repeated code token: %g -> %g", logit, got)
		}
	}
}

func TestSamplingPenaltiesAndHistory(t *testing.T) {
	o := samplingTestOptions()
	o.RepeatPenalty, o.FrequencyPenalty, o.PresencePenalty, o.RepeatLastN = 2, 0.5, 0.25, 3
	s, err := newTokenSampler(o, []int{9, 0, 0, 1})
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		id          int
		logit, want float64
	}{
		{0, 4, 0.75}, {0, -4, -9.25}, {1, 0, -0.75}, {9, 4, 4},
	} {
		if got := s.penalize(tc.id, tc.logit); got != tc.want {
			t.Fatalf("penalty = %g, want %g", got, tc.want)
		}
	}
	s.accept(2)
	s.accept(2)
	if _, ok := s.counts[0]; ok {
		t.Fatal("expired token still penalized")
	}
	if s.counts[2] != 2 {
		t.Fatal("generated history not counted")
	}
	for _, n := range []int{0, -1} {
		o.RepeatLastN = n
		s, err = newTokenSampler(o, []int{0, 0, 1})
		if err != nil {
			t.Fatal(err)
		}
		if n == 0 && len(s.counts) != 0 {
			t.Fatal("disabled penalties retain history")
		}
		if n == -1 && s.counts[0] != 2 {
			t.Fatal("full history lost")
		}
	}
}

func TestSamplingFilters(t *testing.T) {
	for _, tc := range []struct {
		name    string
		k       int
		p, minP float64
		allowed int
	}{
		{"top-k", 2, 1, 0, 2},
		{"top-p", 0, 0.8, 0, 2}, // softmax(3,2,1,0): .644, .237, .087, .032
		{"min-p", 0, 1, 0.3, 2},
		{"retain-one", 0, 0.01, 1, 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			o := samplingTestOptions()
			o.TopK, o.TopP, o.MinP = tc.k, tc.p, tc.minP
			s, err := newTokenSampler(o, nil)
			if err != nil {
				t.Fatal(err)
			}
			seen := make(map[int]bool)
			for range 1000 {
				id, err := s.sample([]float32{3, 2, 1, 0})
				if err != nil {
					t.Fatal(err)
				}
				if id >= tc.allowed {
					t.Fatalf("filtered token %d sampled", id)
				}
				seen[id] = true
			}
			if len(seen) != tc.allowed {
				t.Fatalf("only sampled %v", seen)
			}
		})
	}
}

func TestSamplingTemperatureDistributionAndSeed(t *testing.T) {
	for _, temperature := range []float64{0.5, 1, 2} {
		o := samplingTestOptions()
		o.Temperature = temperature
		s, _ := newTokenSampler(o, nil)
		identical, _ := newTokenSampler(o, nil)
		count := 0
		for range 20000 {
			id, err := s.sample([]float32{10000, 10001})
			if err != nil {
				t.Fatal(err)
			}
			other, _ := identical.sample([]float32{10000, 10001})
			if id != other {
				t.Fatal("seed is not reproducible")
			}
			if id == 1 {
				count++
			}
		}
		want := 1 / (1 + math.Exp(-1/temperature))
		if got := float64(count) / 20000; math.Abs(got-want) > 0.015 {
			t.Fatalf("temperature %g: p=%g, want %g", temperature, got, want)
		}
	}
}

func TestSamplingGreedyAndInvalidInputs(t *testing.T) {
	o := samplingTestOptions()
	o.Temperature, o.RepeatPenalty = 0, 2
	s, _ := newTokenSampler(o, []int{0})
	if id, err := s.sample([]float32{4, 3}); err != nil || id != 1 {
		t.Fatalf("greedy ignored penalty: %d %v", id, err)
	}
	if id, err := s.sample([]float32{0, 0}); err != nil || id != 0 {
		t.Fatalf("tie: %d %v", id, err)
	}
	for _, logits := range [][]float32{nil, {float32(math.Inf(-1))}, {float32(math.NaN())}, {float32(math.Inf(1))}} {
		if _, err := s.sample(logits); err == nil {
			t.Fatalf("accepted invalid logits %v", logits)
		}
	}
	for _, mutate := range []func(*SamplingOptions){
		func(o *SamplingOptions) { o.Temperature = -1 },
		func(o *SamplingOptions) { o.Temperature = math.NaN() },
		func(o *SamplingOptions) { o.TopK = -1 },
		func(o *SamplingOptions) { o.TopP = 0 },
		func(o *SamplingOptions) { o.TopP = 1.1 },
		func(o *SamplingOptions) { o.MinP = -0.1 },
		func(o *SamplingOptions) { o.MinP = 1.1 },
		func(o *SamplingOptions) { o.RepeatPenalty = 0 },
		func(o *SamplingOptions) { o.RepeatLastN = -2 },
		func(o *SamplingOptions) { o.FrequencyPenalty = math.Inf(1) },
		func(o *SamplingOptions) { o.PresencePenalty = math.NaN() },
	} {
		o := samplingTestOptions()
		mutate(&o)
		if _, err := newTokenSampler(o, nil); err == nil {
			t.Fatalf("accepted %+v", o)
		}
	}
}

func TestMulQ80Into(t *testing.T) {
	reader, tensor, data := quantOptimizationFixture(t, 8, 96, 17)
	x := make([]float32, 96)
	for i := range x {
		x[i] = float32(i%7-3) / 8
	}
	want := make([]float32, 17)
	for i := range want {
		want[i] = referenceQ80Optimization(data[i*102:(i+1)*102], x)
	}
	for _, workers := range []int{1, 4} {
		for _, rows := range []int{1, 4, 17} {
			got := make([]float32, 17)
			if err := MulQ80Into(context.Background(), reader, tensor, x, got, Q40Options{Workers: workers, WindowBytes: uint64(rows * 102)}); err != nil {
				t.Fatal(err)
			}
			compareVectors(t, got, want)
		}
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := MulQ80Into(ctx, reader, tensor, x, want, Q40Options{}); err != context.Canceled {
		t.Fatalf("cancellation: %v", err)
	}
	if err := MulQ80Into(context.Background(), reader, tensor, x, want, Q40Options{WindowBytes: 101}); err == nil {
		t.Fatal("accepted short window")
	}
	if err := MulQ80Into(context.Background(), reader, tensor, x, want[:1], Q40Options{}); err == nil {
		t.Fatal("accepted short output")
	}
}

func samplingEngineFixture(t *testing.T) *Engine {
	t.Helper()
	path := filepath.Join(t.TempDir(), "sampling.bin")
	var data []byte
	add := func(typ uint32, shape []uint64, values []byte) ggufindex.Tensor {
		start := len(data)
		data = append(data, values...)
		return ggufindex.Tensor{Type: typ, Shape: shape, Range: ggufindex.Range{File: path, Start: uint64(start), End: uint64(len(data))}}
	}
	embeddings := make([]byte, 3*18)
	for id := range 3 {
		binary.LittleEndian.PutUint16(embeddings[id*18:], 0x3c00)
		for i := 2; i < 18; i++ {
			embeddings[id*18+i] = 0x88
		}
		embeddings[id*18+2] = 0x89
	}
	embd := add(2, []uint64{32, 3}, embeddings)
	norms := make([]byte, 32*4)
	for i := range 32 {
		binary.LittleEndian.PutUint32(norms[i*4:], math.Float32bits(1))
	}
	norm := add(0, []uint64{32}, norms)
	weights := make([]byte, 3*34)
	for id := range 3 {
		binary.LittleEndian.PutUint16(weights[id*34:], 0x3c00)
		weights[id*34+2] = byte(10 - id/2)
	}
	output := add(8, []uint64{32, 3}, weights)
	if err := os.WriteFile(path, data, 0600); err != nil {
		t.Fatal(err)
	}
	reader, err := ggufmmap.Open(&ggufindex.Model{Paths: []string{path}})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = reader.Close() })
	return &Engine{Reader: reader, Config: &Config{EOS: -1, RMSNormEps: 1e-5}, TokenEmbd: embd, OutputNorm: norm, OutputWeight: output,
		KVCache: NewKVCache(0, 16, 1, 2), X: make([]float32, 32), Scratch: &LayerScratch{NormedX: make([]float32, 32)}}
}

func TestGenerateSamplingBreaksRepetitionAndResetsHistory(t *testing.T) {
	e := samplingEngineFixture(t)
	o := samplingTestOptions()
	o.Temperature, o.RepeatPenalty = 0, 2
	for range 2 {
		got, err := e.GenerateWithSampling(context.Background(), []int{0}, 2, o, nil)
		if err != nil || !slices.Equal(got, []int{1, 2}) {
			t.Fatalf("prompt/generated penalties or reset failed: %v %v", got, err)
		}
	}
	o.RepeatLastN = 0
	got, err := e.GenerateWithSampling(context.Background(), []int{0}, 2, o, nil)
	if err != nil || !slices.Equal(got, []int{0, 0}) {
		t.Fatalf("neutral decoding = %v %v", got, err)
	}
	// Neutral defaults must sample both equal maxima, not always pick argmax 0.
	defaults := DefaultSamplingOptions()
	e.Options.Sampling = &defaults
	seen := make(map[int]bool)
	for seed := uint64(0); seed < 32; seed++ {
		defaults.Seed = &seed
		got, err = e.Generate(context.Background(), []int{0}, 1, nil)
		if err != nil || len(got) != 1 || got[0] > 1 {
			t.Fatalf("default sampling = %v %v", got, err)
		}
		seen[got[0]] = true
	}
	if len(seen) != 2 {
		t.Fatalf("equal maxima not both sampled: %v", seen)
	}
}

func TestGenerateSamplingBatchAndSequentialAgree(t *testing.T) {
	a, b := prefillEngineFixture(t), prefillEngineFixture(t)
	a.Options.PrefillBatchSize, b.Options.PrefillBatchSize = 1, 3
	o := samplingTestOptions()
	o.RepeatPenalty, o.FrequencyPenalty = 1.1, 0.2
	prompt := []int{0, 1, 2}
	got, err := a.GenerateWithSampling(context.Background(), prompt, 4, o, nil)
	if err != nil {
		t.Fatal(err)
	}
	want, err := b.GenerateWithSampling(context.Background(), prompt, 4, o, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(got, want) {
		t.Fatalf("batch/sequential = %v/%v", got, want)
	}
}
