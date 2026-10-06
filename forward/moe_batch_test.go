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
	"sync/atomic"
	"testing"

	"Stream-PT/ggufindex"
	ggufmmap "Stream-PT/ggufmap"
)

func moeBatchFixture(tb testing.TB, hidden, expertDim, experts, topK, biasType int, routerMode string) (*ggufmmap.Reader, *Config, *LayerWeights) {
	tb.Helper()
	path := filepath.Join(tb.TempDir(), "moe.bin")
	data := make([]byte, 37)
	appendTensor := func(name string, typ uint32, shape []uint64, encoded []byte) ggufindex.Tensor {
		start := len(data)
		data = append(data, encoded...)
		return ggufindex.Tensor{Name: name, Type: typ, Shape: shape,
			Range: ggufindex.Range{File: path, Start: uint64(start), End: uint64(len(data))}}
	}
	router := make([]byte, hidden*experts*4)
	for e := range experts {
		for i := range hidden {
			value := float32((e*7+i*3)%19-9) / 32
			switch routerMode {
			case "ties":
				value = 0
			case "nonfinite":
				value = float32(math.NaN())
			case "mixed-nan":
				if e != 1 {
					value = float32(math.NaN())
				} else {
					value = 0
				}
			case "infinite":
				value = float32(math.Inf(1))
			case "underflow":
				value = float32(e * 1000)
			}
			binary.LittleEndian.PutUint32(router[(e*hidden+i)*4:], math.Float32bits(value))
		}
	}
	lw := &LayerWeights{FFNGateInp: appendTensor("router", 0, []uint64{uint64(hidden), uint64(experts)}, router)}
	quantized := func(name string, input, output, salt int) ggufindex.Tensor {
		encoded := make([]byte, input/32*17*output*experts)
		for block := 0; block < len(encoded)/17; block++ {
			row := encoded[block*17 : (block+1)*17]
			row[0] = byte(120 + (block+salt)%4)
			for i := 1; i < 17; i++ {
				row[i] = byte(block*13 + i*7 + salt)
			}
		}
		return appendTensor(name, 39, []uint64{uint64(input), uint64(output), uint64(experts)}, encoded)
	}
	lw.FFNGateExps = quantized("gate", hidden, expertDim, 1)
	lw.FFNUpExps = quantized("up", hidden, expertDim, 7)
	lw.FFNDownExps = quantized("down", expertDim, hidden, 17)
	if biasType >= 0 {
		bias := func(name string, dim, count int, vector bool) ggufindex.Tensor {
			width := 4
			if biasType != 0 {
				width = 2
			}
			encoded := make([]byte, dim*count*width)
			for i := 0; i < dim*count; i++ {
				value := float32(i%5-2) / 4
				switch biasType {
				case 0:
					binary.LittleEndian.PutUint32(encoded[i*4:], math.Float32bits(value))
				case 30:
					binary.LittleEndian.PutUint16(encoded[i*2:], uint16(math.Float32bits(value)>>16))
				case 1:
					bits := [...]uint16{0xb800, 0xb400, 0, 0x3400, 0x3800}
					binary.LittleEndian.PutUint16(encoded[i*2:], bits[i%5])
				}
			}
			shape := []uint64{uint64(dim), uint64(count)}
			if vector {
				shape = shape[:1]
			}
			return appendTensor(name, uint32(biasType), shape, encoded)
		}
		lw.FFNGateInpBias = bias("router.bias", experts, 1, true)
		lw.FFNGateExpsBias = bias("gate.bias", expertDim, experts, false)
		lw.FFNUpExpsBias = bias("up.bias", expertDim, experts, false)
		lw.FFNDownExpsBias = bias("down.bias", hidden, experts, false)
	}
	if err := os.WriteFile(path, data, 0600); err != nil {
		tb.Fatal(err)
	}
	reader, err := ggufmmap.Open(&ggufindex.Model{Paths: []string{path}})
	if err != nil {
		tb.Fatal(err)
	}
	tb.Cleanup(func() {
		if err := reader.Close(); err != nil {
			tb.Error(err)
		}
	})
	return reader, &Config{HiddenDim: hidden, ExpertHiddenDim: expertDim, NumExperts: experts, NumExpertsUsed: topK}, lw
}

func moeBatchSequential(ctx context.Context, reader *ggufmmap.Reader, x []float32, cfg *Config, lw *LayerWeights, out []float32, batch int, options Q40Options) error {
	for token := range batch {
		if err := forwardMoE(ctx, reader, x[token*cfg.HiddenDim:(token+1)*cfg.HiddenDim], cfg,
			lw.FFNGateInp, lw.FFNGateInpBias, lw.FFNGateExps, lw.FFNGateExpsBias,
			lw.FFNUpExps, lw.FFNUpExpsBias, lw.FFNDownExps, lw.FFNDownExpsBias,
			out[token*cfg.HiddenDim:(token+1)*cfg.HiddenDim], options, nil); err != nil {
			return err
		}
	}
	return nil
}

func TestForwardMoEBatchMatchesSequential(t *testing.T) {
	for _, experts := range []int{1, 4, 7} {
		for _, topK := range []int{1, min(2, experts), experts} {
			for _, biasType := range []int{-1, 0, 1, 30} {
				for _, mode := range []string{"normal", "ties", "nonfinite", "mixed-nan", "infinite", "underflow", "clamp"} {
					t.Run(fmt.Sprintf("experts%d/k%d/bias%d/%s", experts, topK, biasType, mode), func(t *testing.T) {
						reader, cfg, lw := moeBatchFixture(t, 96, 64, experts, topK, biasType, mode)
						for _, batch := range []int{1, 5, 37} {
							x, want, got := make([]float32, batch*cfg.HiddenDim), make([]float32, batch*cfg.HiddenDim), make([]float32, batch*cfg.HiddenDim)
							for i := range x {
								x[i] = float32((i*17+i/cfg.HiddenDim)%31-15) / 16
								if mode == "clamp" {
									x[i] *= 1000
								}
							}
							original := append([]float32(nil), x...)
							if err := moeBatchSequential(context.Background(), reader, x, cfg, lw, want, batch, Q40Options{Workers: 1}); err != nil {
								t.Fatal(err)
							}
							for _, workers := range []int{0, 1, 3} {
								for i := range got {
									got[i] = 999
								}
								if err := forwardMoEBatch(context.Background(), reader, x, cfg, lw, got, batch, Q40Options{Workers: workers}, nil); err != nil {
									t.Fatal(err)
								}
								for i := range got {
									if math.Float32bits(got[i]) != math.Float32bits(want[i]) && !(math.IsNaN(float64(got[i])) && math.IsNaN(float64(want[i]))) {
										t.Fatalf("batch%d workers%d output[%d]=%g, want %g", batch, workers, i, got[i], want[i])
									}
								}
							}
							for i := range x {
								if x[i] != original[i] {
									t.Fatal("input modified")
								}
							}
						}
					})
				}
			}
		}
	}
}

func TestForwardMoEBatchValidation(t *testing.T) {
	reader, cfg, lw := moeBatchFixture(t, 32, 64, 4, 2, 0, "normal")
	ctx := context.Background()
	x, out := make([]float32, 64), make([]float32, 64)
	for _, tc := range []struct {
		name   string
		change func(*Config, *LayerWeights)
	}{
		{"zero hidden", func(c *Config, _ *LayerWeights) { c.HiddenDim = 0 }},
		{"negative expert dimension", func(c *Config, _ *LayerWeights) { c.ExpertHiddenDim = -1 }},
		{"unaligned expert dimension", func(c *Config, _ *LayerWeights) { c.ExpertHiddenDim = 63 }},
		{"invalid topk", func(c *Config, _ *LayerWeights) { c.NumExpertsUsed = 5 }},
		{"overflow", func(c *Config, _ *LayerWeights) { c.HiddenDim = int(^uint(0) >> 1) }},
		{"router type", func(_ *Config, w *LayerWeights) { w.FFNGateInp.Type = 1 }},
		{"router shape", func(_ *Config, w *LayerWeights) { w.FFNGateInp.Shape = nil }},
		{"router range", func(_ *Config, w *LayerWeights) { w.FFNGateInp.Range.End-- }},
		{"expert type", func(_ *Config, w *LayerWeights) { w.FFNUpExps.Type = 0 }},
		{"expert shape", func(_ *Config, w *LayerWeights) { w.FFNGateExps.Shape = []uint64{32, 64, 5} }},
		{"expert range", func(_ *Config, w *LayerWeights) {
			w.FFNDownExps.Range.End = w.FFNDownExps.Range.Start - 1
		}},
		{"expert range overflow", func(_ *Config, w *LayerWeights) { w.FFNGateExps.Range.Start = ^uint64(0) - 7 }},
		{"bias type", func(_ *Config, w *LayerWeights) { w.FFNUpExpsBias.Type = 39 }},
		{"bias shape", func(_ *Config, w *LayerWeights) { w.FFNGateExpsBias.Shape = nil }},
		{"bias range", func(_ *Config, w *LayerWeights) { w.FFNGateInpBias.Range.End-- }},
		{"file range", func(_ *Config, w *LayerWeights) {
			w.FFNGateInp.Range.Start += 100000
			w.FFNGateInp.Range.End += 100000
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c, w := *cfg, *lw
			tc.change(&c, &w)
			if err := forwardMoEBatch(ctx, reader, x, &c, &w, out, 2, Q40Options{}, nil); err == nil {
				t.Fatal("expected validation error")
			}
		})
	}
	for _, batch := range []int{-1, 0, 1, int(^uint(0) >> 1)} {
		if err := forwardMoEBatch(ctx, reader, x, cfg, lw, out, batch, Q40Options{}, nil); err == nil {
			t.Fatalf("accepted batch %d", batch)
		}
	}
	for _, call := range []func() error{
		func() error { return forwardMoEBatch(nil, reader, x, cfg, lw, out, 2, Q40Options{}, nil) },
		func() error { return forwardMoEBatch(ctx, nil, x, cfg, lw, out, 2, Q40Options{}, nil) },
		func() error { return forwardMoEBatch(ctx, reader, x, nil, lw, out, 2, Q40Options{}, nil) },
		func() error { return forwardMoEBatch(ctx, reader, x, cfg, nil, out, 2, Q40Options{}, nil) },
		func() error {
			return forwardMoEBatch(ctx, reader, x, cfg, lw, out[:63], 2, Q40Options{}, nil)
		},
		func() error { return forwardMoEBatch(ctx, reader, x, cfg, lw, x, 2, Q40Options{}, nil) },
	} {
		if err := call(); err == nil {
			t.Fatal("expected validation error")
		}
	}
	backing := make([]float32, 65)
	for _, reverse := range []bool{false, true} {
		x, y := backing[:64], backing[1:]
		if reverse {
			x, y = y, x
		}
		if err := forwardMoEBatch(ctx, reader, x, cfg, lw, y, 2, Q40Options{}, nil); err == nil {
			t.Fatal("accepted partial overlap")
		}
	}
}

type moeBatchCancelContext struct {
	context.Context
	calls atomic.Int64
	limit int64
}

func (c *moeBatchCancelContext) Err() error {
	if c.calls.Add(1) >= c.limit {
		return context.Canceled
	}
	return nil
}

func TestForwardMoEBatchCancellation(t *testing.T) {
	reader, cfg, lw := moeBatchFixture(t, 96, 64, 4, 4, 0, "ties")
	x, out := make([]float32, 37*cfg.HiddenDim), make([]float32, 37*cfg.HiddenDim)
	for _, limit := range []int64{1, 2, 60, 150} {
		ctx := &moeBatchCancelContext{Context: context.Background(), limit: limit}
		if err := forwardMoEBatch(ctx, reader, x, cfg, lw, out, 37, Q40Options{Workers: 3}, nil); !errors.Is(err, context.Canceled) {
			t.Fatalf("limit%d: got %v", limit, err)
		}
	}
}

func BenchmarkForwardMoEBatch(b *testing.B) {
	const batch = 32
	reader, cfg, lw := moeBatchFixture(b, 256, 512, 8, 2, 0, "normal")
	x, out := make([]float32, batch*cfg.HiddenDim), make([]float32, batch*cfg.HiddenDim)
	for i := range x {
		x[i] = float32(i%23-11) / 32
	}
	for _, batched := range []bool{false, true} {
		b.Run(fmt.Sprintf("batched%v", batched), func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				var err error
				if batched {
					err = forwardMoEBatch(context.Background(), reader, x, cfg, lw, out, batch, Q40Options{Workers: 1}, nil)
				} else {
					err = moeBatchSequential(context.Background(), reader, x, cfg, lw, out, batch, Q40Options{Workers: 1})
				}
				if err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
