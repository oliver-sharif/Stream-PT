//go:build goexperiment.simd

package forward

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"simd"
	"testing"

	"Stream-PT/ggufindex"
	ggufmmap "Stream-PT/ggufmap"
)

func batchEngineFixture(t testing.TB) *Engine {
	t.Helper()
	const dim, vocab, heads, kvHeads, headDim, experts, layers = 64, 4, 4, 2, 16, 3, 3
	path := filepath.Join(t.TempDir(), "batch.bin")
	var data []byte
	add := func(name string, typ uint32, shape []uint64, values []byte) ggufindex.Tensor {
		start := len(data)
		data = append(data, values...)
		return ggufindex.Tensor{Name: name, Type: typ, Shape: shape,
			Range: ggufindex.Range{File: path, Start: uint64(start), End: uint64(len(data))}}
	}
	floats := func(name string, shape []uint64, seed int) ggufindex.Tensor {
		n := 1
		for _, d := range shape {
			n *= int(d)
		}
		values := make([]byte, n*4)
		for i := 0; i < n; i++ {
			v := float32((i*7+seed)%19-9) / 128
			if name == "norm" {
				v += 1
			}
			binary.LittleEndian.PutUint32(values[i*4:], math.Float32bits(v))
		}
		return add(name, 0, shape, values)
	}
	quant := func(name string, typ uint32, shape []uint64, seed int) ggufindex.Tensor {
		blocks := 1
		for _, d := range shape {
			blocks *= int(d)
		}
		blocks /= 32
		width := map[uint32]int{2: 18, 8: 34, 39: 17}[typ]
		values := make([]byte, blocks*width)
		for block := 0; block < blocks; block++ {
			encoded := values[block*width : (block+1)*width]
			start := 2
			if typ == 39 {
				encoded[0] = 120
				start = 1
			} else {
				binary.LittleEndian.PutUint16(encoded, 0x2400) // 1/64
			}
			for i := start; i < width; i++ {
				encoded[i] = byte(block*13 + i*7 + seed)
			}
		}
		return add(name, typ, shape, values)
	}
	cfg := &Config{HiddenDim: dim, NumLayers: layers, NumHeads: heads, NumKVHeads: kvHeads,
		HeadDim: headDim, NumExperts: experts, NumExpertsUsed: 2, ExpertHiddenDim: dim, VocabSize: vocab,
		RMSNormEps: 1e-5, RopeFreqBase: 10000, SlidingWindow: 3, EOS: -1,
		RopeScalingFactor: 2, RopeOriginalContext: 4}
	e := &Engine{Config: cfg, Options: EngineOptions{Workers: 1, WindowBytes: 3 * 36},
		X: make([]float32, dim), Scratch: NewLayerScratch(cfg), KVCache: NewKVCache(layers, 32, kvHeads, headDim)}
	e.TokenEmbd = quant("embedding", 2, []uint64{dim, vocab}, 1)
	e.OutputWeight = quant("output", 8, []uint64{dim, vocab}, 7)
	e.OutputNorm = floats("norm", []uint64{dim}, 5)
	for layer := range layers {
		lw := LayerWeights{
			AttnNorm: floats("norm", []uint64{dim}, layer), PostAttnNorm: floats("norm", []uint64{dim}, layer+1),
			AttnQ: quant("q", 2, []uint64{dim, dim}, layer), AttnK: quant("k", 2, []uint64{dim, kvHeads * headDim}, layer+2),
			AttnV: quant("v", 2, []uint64{dim, kvHeads * headDim}, layer+3), AttnOutput: quant("o", 2, []uint64{dim, dim}, layer+4),
			AttnQBias: floats("qb", []uint64{dim}, layer), AttnKBias: floats("kb", []uint64{kvHeads * headDim}, layer+1),
			AttnVBias: floats("vb", []uint64{kvHeads * headDim}, layer+2), AttnOutputBias: floats("ob", []uint64{dim}, layer+3),
			AttnSinks:  floats("sinks", []uint64{heads}, layer),
			FFNGateInp: floats("router", []uint64{dim, experts}, layer), FFNGateInpBias: floats("rb", []uint64{experts}, layer),
			FFNGateExps: quant("gate", 39, []uint64{dim, dim, experts}, layer), FFNUpExps: quant("up", 39, []uint64{dim, dim, experts}, layer+1),
			FFNDownExps:     quant("down", 39, []uint64{dim, dim, experts}, layer+2),
			FFNGateExpsBias: floats("gb", []uint64{dim, experts}, layer), FFNUpExpsBias: floats("ub", []uint64{dim, experts}, layer+1),
			FFNDownExpsBias: floats("db", []uint64{dim, experts}, layer+2),
		}
		e.Layers = append(e.Layers, lw)
	}
	if err := os.WriteFile(path, data, 0600); err != nil {
		t.Fatal(err)
	}
	reader, err := ggufmmap.Open(&ggufindex.Model{Paths: []string{path}})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := reader.Close(); err != nil {
			t.Error(err)
		}
	})
	e.Reader = reader
	return e
}

func compareEngineState(t *testing.T, got, want *Engine) {
	t.Helper()
	compareVectors(t, got.X, want.X)
	for layer := range got.KVCache.Keys {
		for pos := range got.KVCache.Keys[layer] {
			compareVectors(t, got.KVCache.Keys[layer][pos], want.KVCache.Keys[layer][pos])
			compareVectors(t, got.KVCache.Values[layer][pos], want.KVCache.Values[layer][pos])
		}
	}
}

func TestPrefillBatchMatchesSequential(t *testing.T) {
	prompt := []int{0, 1, 2, 3, 1, 0, 3, 2, 1, 2, 0}
	for _, size := range []int{0, 1, 2, 3, 8, 32} {
		for _, workers := range []int{1, 3} {
			t.Run(fmt.Sprintf("batch%d/workers%d", size, workers), func(t *testing.T) {
				got, want := batchEngineFixture(t), batchEngineFixture(t)
				got.Options.PrefillBatchSize, got.Options.Workers = size, workers
				var next int
				for pos, token := range prompt {
					var err error
					next, err = want.ForwardToken(context.Background(), token, pos)
					if err != nil {
						t.Fatal(err)
					}
				}
				actual, err := got.Prefill(context.Background(), prompt, 0)
				if err != nil || actual != next {
					t.Fatalf("next = %d, want %d; error = %v", actual, next, err)
				}
				compareEngineState(t, got, want)
				// Decode must consume precisely the same prefix state afterwards.
				for pos := len(prompt); pos < len(prompt)+3; pos++ {
					actual, err = got.ForwardToken(context.Background(), next, pos)
					if err != nil {
						t.Fatal(err)
					}
					next, err = want.ForwardToken(context.Background(), next, pos)
					if err != nil || actual != next {
						t.Fatalf("decode next = %d, want %d; error = %v", actual, next, err)
					}
					compareEngineState(t, got, want)
				}
			})
		}
	}
}

func TestPrefillBatchCausalAndAppend(t *testing.T) {
	prompt := []int{0, 1, 2, 3, 0, 2, 1}
	full, split := batchEngineFixture(t), batchEngineFixture(t)
	if _, err := full.Prefill(context.Background(), prompt, 0); err != nil {
		t.Fatal(err)
	}
	if _, err := split.Prefill(context.Background(), prompt[:3], 0); err != nil {
		t.Fatal(err)
	}
	for layer := range full.KVCache.Keys {
		for pos := range 3 {
			compareVectors(t, full.KVCache.Keys[layer][pos], split.KVCache.Keys[layer][pos])
			compareVectors(t, full.KVCache.Values[layer][pos], split.KVCache.Values[layer][pos])
		}
	}
	if _, err := split.Prefill(context.Background(), prompt[3:], 3); err != nil {
		t.Fatal(err)
	}
	compareEngineState(t, full, split)
}

func TestPrefillBatchValidation(t *testing.T) {
	for _, tc := range []struct {
		name         string
		tokens       []int
		start, batch int
	}{
		{"empty", nil, 0, 0}, {"negative position", []int{0, 1}, -1, 0},
		{"capacity", []int{0, 1}, 31, 0}, {"negative batch", []int{0, 1}, 0, -1},
		{"invalid token", []int{0, -1, 1}, 0, 2},
	} {
		t.Run(tc.name, func(t *testing.T) {
			e := batchEngineFixture(t)
			e.Options.PrefillBatchSize = tc.batch
			if _, err := e.Prefill(context.Background(), tc.tokens, tc.start); err == nil {
				t.Fatal("expected validation error")
			}
		})
	}
	e := batchEngineFixture(t)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := e.Prefill(ctx, []int{0, 1}, 0); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancellation = %v", err)
	}
	if _, err := e.Prefill(nil, []int{0, 1}, 0); err == nil {
		t.Fatal("nil context accepted")
	}
}

func TestPrefillBatchCancellationDuringLayer(t *testing.T) {
	for _, workers := range []int{1, 3} {
		e := batchEngineFixture(t)
		e.Options.Workers = workers
		base, cancel := context.WithCancel(context.Background())
		ctx := &cancelAfterChecks{Context: base, cancel: cancel, limit: 40}
		calls := 0
		got, err := e.Generate(ctx, []int{0, 1, 2, 3}, 2, func(int, string) bool { calls++; return true })
		cancel()
		if !errors.Is(err, context.Canceled) || len(got) != 0 || calls != 0 {
			t.Fatalf("workers %d: tokens=%v callbacks=%d error=%v", workers, got, calls, err)
		}
	}
}

func TestPrefillBatchNormalizationFormats(t *testing.T) {
	for _, typ := range []uint32{1, 28} {
		t.Run(fmt.Sprintf("type%d", typ), func(t *testing.T) {
			got, want := batchEngineFixture(t), batchEngineFixture(t)
			path := filepath.Join(t.TempDir(), "norm.bin")
			data := make([]byte, got.Config.HiddenDim*2)
			for i := 0; i < got.Config.HiddenDim; i++ {
				bits := uint16(0x3c00 + i%7)
				if typ == 28 {
					bits = 0x3f80 + uint16(i%7)
				}
				binary.LittleEndian.PutUint16(data[i*2:], bits)
			}
			if err := os.WriteFile(path, data, 0600); err != nil {
				t.Fatal(err)
			}
			weight := ggufindex.Tensor{Name: "norm", Type: typ, Shape: []uint64{uint64(got.Config.HiddenDim)},
				Range: ggufindex.Range{File: path, End: uint64(len(data))}}
			for _, e := range []*Engine{got, want} {
				reader, err := ggufmmap.Open(&ggufindex.Model{Paths: []string{e.TokenEmbd.Range.File, path}})
				if err != nil {
					t.Fatal(err)
				}
				t.Cleanup(func() {
					if err := reader.Close(); err != nil {
						t.Error(err)
					}
				})
				e.Reader, e.OutputNorm = reader, weight
				for layer := range e.Layers {
					e.Layers[layer].AttnNorm, e.Layers[layer].PostAttnNorm = weight, weight
				}
			}
			prompt := []int{0, 1, 2, 3, 0}
			for pos, token := range prompt {
				if _, err := want.ForwardToken(context.Background(), token, pos); err != nil {
					t.Fatal(err)
				}
			}
			if _, err := got.Prefill(context.Background(), prompt, 0); err != nil {
				t.Fatal(err)
			}
			compareEngineState(t, got, want)
		})
	}
}

func TestAttentionScratchNoAllocations(t *testing.T) {
	cache := NewKVCache(1, 16, 1, 7)
	q, k, v, out := make([]float32, 14), make([]float32, 7), make([]float32, 7), make([]float32, 14)
	for i := range q {
		q[i] = float32(i%5-2) / 8
	}
	for pos := range 16 {
		for i := range k {
			cache.Keys[0][pos][i] = float32((i+pos)%7-3) / 8
			cache.Values[0][pos][i] = float32((i+pos)%9-4) / 8
		}
	}
	options := AttentionOptions{Sinks: []float32{0, 1}, SlidingWindow: 5}
	var vector simd.Float32s
	var scratch AttentionScratch
	scratch.prepare(16, vector.Len())
	call := func() {
		forwardAttention(q, k, v, cache, 0, 15, 2, 1, 7, out, options, scratch.scores, scratch.partials)
	}
	call()
	want := append([]float32(nil), out...)
	if allocs := testing.AllocsPerRun(100, call); allocs != 0 {
		t.Fatalf("reused attention allocates %g times", allocs)
	}
	if !reflect.DeepEqual(out, want) {
		t.Fatal("scratch reuse changed output")
	}
	attentionForTest(q, k, v, cache, 0, 15, 2, 1, 7, out, AttentionOptions{Sinks: options.Sinks, SlidingWindow: 5})
	compareVectors(t, want, out)
}

func BenchmarkPrefillBatch(b *testing.B) {
	for _, size := range []int{1, 8, 32} {
		b.Run(fmt.Sprintf("batch%d", size), func(b *testing.B) {
			e := batchEngineFixture(b)
			e.Options.PrefillBatchSize = size
			e.Options.WindowBytes = DefaultQ40WindowBytes
			prompt := make([]int, 32)
			for i := range prompt {
				prompt[i] = i % 4
			}
			ctx := context.Background()
			if _, err := e.Prefill(ctx, prompt, 0); err != nil {
				b.Fatal(err)
			}
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, err := e.Prefill(ctx, prompt, 0); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
