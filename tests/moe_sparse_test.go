//go:build goexperiment.simd

package tests

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"sort"
	"testing"

	"Stream-PT/forward"
	"Stream-PT/ggufindex"
	ggufmmap "Stream-PT/ggufmap"
)

func scalarSparseMoE(t *testing.T, x []float32, cfg *forward.Config, lw *forward.LayerWeights) []float32 {
	t.Helper()
	data, err := os.ReadFile(lw.FFNGateInp.Range.File)
	if err != nil {
		t.Fatal(err)
	}
	logits := make([]float64, cfg.NumExperts)
	order := make([]int, cfg.NumExperts)
	for e := range order {
		order[e] = e
		start := int(lw.FFNGateInp.Range.Start) + e*cfg.HiddenDim*4
		for i, value := range x {
			w := math.Float32frombits(binary.LittleEndian.Uint32(data[start+i*4:]))
			logits[e] += float64(w) * float64(value)
		}
	}
	sort.SliceStable(order, func(i, j int) bool { return logits[order[i]] > logits[order[j]] })
	order = order[:cfg.NumExpertsUsed]
	weights := make([]float64, len(order))
	var total float64
	for i, e := range order {
		weights[i] = math.Exp(logits[e] - logits[order[0]])
		total += weights[i]
	}
	levels := [...]float64{0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6}
	project := func(tensor ggufindex.Tensor, expert int, input []float64) []float64 {
		rows, blocks := int(tensor.Shape[1]), int(tensor.Shape[0])/32
		out := make([]float64, rows)
		start := int(tensor.Range.Start) + expert*rows*blocks*17
		for row := range out {
			for block := range blocks {
				encoded := data[start+(row*blocks+block)*17:][:17]
				scale := math.Ldexp(1, int(encoded[0])-127)
				for i, packed := range encoded[1:] {
					out[row] += scale * levels[packed&15] * input[block*32+i]
					out[row] += scale * levels[packed>>4] * input[block*32+i+16]
				}
			}
		}
		return out
	}
	input := make([]float64, len(x))
	for i := range x {
		input[i] = float64(x[i])
	}
	result := make([]float32, cfg.HiddenDim)
	for rank, expert := range order {
		gate, up := project(lw.FFNGateExps, expert, input), project(lw.FFNUpExps, expert, input)
		for i := range gate {
			g, u := min(gate[i], 7), max(-7, min(up[i], 7))
			gate[i] = g / (1 + math.Exp(-1.702*g)) * (u + 1)
		}
		down := project(lw.FFNDownExps, expert, gate)
		for i := range result {
			result[i] += float32(down[i] * weights[rank] / total)
		}
	}
	return result
}

func TestSparseMoEIndependentReference(t *testing.T) {
	for _, mode := range []string{"ties", "normal", "underflow"} {
		for _, k := range []int{1, 2, 8} {
			t.Run(fmt.Sprintf("%s/k%d", mode, k), func(t *testing.T) {
				reader, cfg, lw := moeBatchFixture(t, 96, 64, 8, k, -1, mode)
				const batch = 19
				x, out := make([]float32, batch*cfg.HiddenDim), make([]float32, batch*cfg.HiddenDim)
				for i := range x {
					x[i] = float32(i%13+1) / 16
				}
				for _, workers := range []int{1, 4} {
					if err := forward.ForwardMoEBatch(context.Background(), reader, x, cfg, lw, out, batch, forward.Q40Options{Workers: workers}); err != nil {
						t.Fatal(err)
					}
					for token := range batch {
						want := scalarSparseMoE(t, x[token*cfg.HiddenDim:(token+1)*cfg.HiddenDim], cfg, lw)
						for i, value := range want {
							got := out[token*cfg.HiddenDim+i]
							if math.Abs(float64(got-value)) > 3e-5*max(1, math.Abs(float64(value))) {
								t.Fatalf("workers%d token%d output%d=%g want %g", workers, token, i, got, value)
							}
						}
					}
				}
			})
		}
	}
}

func BenchmarkSparseMoE(b *testing.B) {
	for _, batch := range []int{1, 32} {
		for _, workers := range []int{1, 2, 4} {
			b.Run(fmt.Sprintf("batch%d/workers%d", batch, workers), func(b *testing.B) {
				reader, cfg, lw := moeBatchFixture(b, 256, 512, 32, 2, -1, "ties")
				x, out := make([]float32, batch*cfg.HiddenDim), make([]float32, batch*cfg.HiddenDim)
				for i := range x {
					x[i] = .25
				}
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					if err := forward.ForwardMoEBatch(context.Background(), reader, x, cfg, lw, out, batch, forward.Q40Options{Workers: workers}); err != nil {
						b.Fatal(err)
					}
				}
			})
		}
	}
}

func TestSparseMoESelectedWeightBytes(t *testing.T) {
	for _, batch := range []int{1, 23} {
		for _, mode := range []string{"ties", "underflow", "nonfinite"} {
			t.Run(fmt.Sprintf("batch%d/%s", batch, mode), func(t *testing.T) {
				reader, cfg, lw := moeBatchFixture(t, 64, 64, 8, 2, -1, mode)
				if err := reader.ConfigureExpertCache(ggufmmap.ExpertCacheOptions{TrackStats: true}); err != nil {
					t.Fatal(err)
				}
				x, out := make([]float32, batch*cfg.HiddenDim), make([]float32, batch*cfg.HiddenDim)
				for i := range x {
					x[i] = .5
				}
				if batch == 1 {
					if err := moeBatchSequential(context.Background(), reader, x, cfg, lw, out, batch, forward.Q40Options{Workers: 4}); err != nil {
						t.Fatal(err)
					}
				} else if err := forward.ForwardMoEBatch(context.Background(), reader, x, cfg, lw, out, batch, forward.Q40Options{Workers: 4}); err != nil {
					t.Fatal(err)
				}
				stats := reader.ExpertCacheStats()
				active := 2
				if mode == "underflow" {
					active = 1
				}
				matrixBytes := uint64(64 / 32 * 17 * 64)
				wantBytes := uint64(active) * 3 * matrixBytes
				if stats.SelectedRanges != uint64(active*3) || stats.SelectedBytes != wantBytes || stats.MappedBytes != wantBytes {
					t.Fatalf("stats=%+v, want %d ranges/%d bytes", stats, active*3, wantBytes)
				}
				wantRuns := uint64(3)
				if active == 0 {
					wantRuns = 0
				}
				if stats.MappedRuns != wantRuns || len(stats.RangeUses) != active*3 {
					t.Fatalf("runs=%d frequencies=%v", stats.MappedRuns, stats.RangeUses)
				}
				for span, uses := range stats.RangeUses {
					found := false
					for _, tensor := range []ggufindex.Tensor{lw.FFNGateExps, lw.FFNUpExps, lw.FFNDownExps} {
						if span.File != tensor.Range.File || span.Start < tensor.Range.Start || span.End > tensor.Range.End {
							continue
						}
						expert := (span.Start - tensor.Range.Start) / matrixBytes
						found = span.End-span.Start == matrixBytes && uses == 1 &&
							(((mode == "ties" || mode == "nonfinite") && expert < 2) || (mode == "underflow" && expert == 7))
					}
					if !found {
						t.Fatalf("unexpected expert access %+v (%d uses)", span, uses)
					}
				}
			})
		}
	}
}

func TestSparseMoEReversedProjectionLayout(t *testing.T) {
	reader, cfg, lw := moeBatchFixture(t, 64, 64, 8, 2, -1, "ties")
	lw.FFNGateExps, lw.FFNUpExps = lw.FFNUpExps, lw.FFNGateExps
	x, out := make([]float32, 64), make([]float32, 64)
	for i := range x {
		x[i] = float32(i%7-3) / 16
	}
	want := scalarSparseMoE(t, x, cfg, lw)
	if err := moeBatchSequential(context.Background(), reader, x, cfg, lw, out, 1, forward.Q40Options{Workers: 4}); err != nil {
		t.Fatal(err)
	}
	for i, value := range want {
		if math.Abs(float64(out[i]-value)) > 3e-5*max(1, math.Abs(float64(value))) {
			t.Fatalf("output%d=%g want %g", i, out[i], value)
		}
	}
}
