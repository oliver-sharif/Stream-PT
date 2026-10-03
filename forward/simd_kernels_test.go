//go:build goexperiment.simd

package forward

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"testing"
)

func simdKernelRow(input int) ([]byte, []float32) {
	row, x := make([]byte, input/32*17), make([]float32, input)
	for block := 0; block < input/32; block++ {
		row[block*17] = byte(123 + block%8)
		for i := 0; i < 16; i++ {
			row[block*17+i+1] = byte(block*37 + i*17)
		}
	}
	for i := range x {
		x[i] = float32(i%31-15) / 32
	}
	return row, x
}

func TestSIMDKernelGather(t *testing.T) {
	levels := [...]float64{0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6}
	for _, input := range []int{32, 96, 2880} {
		for _, count := range []int{1, 2, 3, 4, 8, 16} {
			t.Run(fmt.Sprintf("input%d/count%d", input, count), func(t *testing.T) {
				row, vector := simdKernelRow(input)
				const output = 3
				data := make([]byte, output*len(row))
				for i := 0; i < output; i++ {
					copy(data[i*len(row):], row)
				}
				x := make([]float32, count*(input+1))
				y := make([]float32, count*(output+1))
				starts, ends := make([]int, count), make([]int, count)
				bias := []float32{.125, -.25, .5}
				for token := 0; token < count; token++ {
					starts[token], ends[token] = (count-1-token)*(input+1)+1, token*(output+1)+1
					for i, value := range vector {
						x[starts[token]+i] = value * float32(token+1)
					}
				}
				computeMoEBatchExpertRows(context.Background(), data, input, x, starts, bias, y, ends, 0, output)
				for token, start := range starts {
					var want float64
					for block := 0; block < input/32; block++ {
						scale := math.Ldexp(1, int(row[block*17])-127)
						for i := 0; i < 16; i++ {
							packed := row[block*17+i+1]
							want += scale * levels[packed&15] * float64(x[start+block*32+i])
							want += scale * levels[packed>>4] * float64(x[start+block*32+i+16])
						}
					}
					for i, b := range bias {
						got, expected := float64(y[ends[token]+i]), want+float64(b)
						if math.IsNaN(got) || math.Abs(got-expected) > 3e-5*max(1, math.Abs(expected)) {
							t.Fatalf("token%d row%d=%g want %g", token, i, got, expected)
						}
					}
					if y[token*(output+1)] != 0 {
						t.Fatal("scatter overwrote sentinel")
					}
				}
			})
		}
	}
}

func BenchmarkSIMDKernelMXFP4(b *testing.B) {
	row, vector := simdKernelRow(2880)
	for _, count := range []int{1, 2, 4, 8, 16} {
		b.Run(fmt.Sprintf("positions%d", count), func(b *testing.B) {
			const output = 64
			data := make([]byte, output*len(row))
			for i := 0; i < output; i++ {
				copy(data[i*len(row):], row)
			}
			x, y, bias := make([]float32, count*2880), make([]float32, count*output), make([]float32, output)
			starts, ends := make([]int, count), make([]int, count)
			for i := 0; i < count; i++ {
				copy(x[i*2880:], vector)
				starts[i], ends[i] = i*2880, i*output
			}
			ctx := context.Background()
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				computeMoEBatchExpertRows(ctx, data, 2880, x, starts, bias, y, ends, 0, output)
			}
		})
	}
}

func TestSIMDKernelRouter(t *testing.T) {
	for _, hidden := range []int{3, 31, 32, 96, 2880} {
		for _, batch := range []int{1, 2, 17} {
			t.Run(fmt.Sprintf("hidden%d/batch%d", hidden, batch), func(t *testing.T) {
				const experts = 5
				s := &moeBatchScratch{batch: batch, hidden: hidden, experts: experts,
					logits: make([]float32, batch*experts), routerBias: []float32{.125, -.25, .5, -1, 2}}
				storage := make([]byte, experts*hidden*4+1)
				data := storage[1:] // Deliberately unaligned mmap-equivalent view.
				x := make([]float32, batch*hidden)
				for i := range x {
					x[i] = float32(i%31-15) / 32
				}
				for i := 0; i < len(data)/4; i++ {
					binary.LittleEndian.PutUint32(data[i*4:], math.Float32bits(float32(i%23-11)/64))
				}
				if err := routeMoEBatch(context.Background(), data, x, s); err != nil {
					t.Fatal(err)
				}
				for token := 0; token < batch; token++ {
					for expert := 0; expert < experts; expert++ {
						want := float64(s.routerBias[expert])
						for i := 0; i < hidden; i++ {
							want += float64(math.Float32frombits(binary.LittleEndian.Uint32(data[(expert*hidden+i)*4:]))) * float64(x[token*hidden+i])
						}
						got := float64(s.logits[token*experts+expert])
						if math.IsNaN(got) || math.Abs(got-want) > 3e-5*max(1, math.Abs(want)) {
							t.Fatalf("token%d expert%d=%g want %g", token, expert, got, want)
						}
					}
				}
				ctx, cancel := context.WithCancel(context.Background())
				cancel()
				if err := routeMoEBatch(ctx, data, x, s); err != context.Canceled {
					t.Fatalf("cancellation: %v", err)
				}
				for _, value := range []float32{float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1))} {
					binary.LittleEndian.PutUint32(data, math.Float32bits(value))
					x[0] = 1
					if err := routeMoEBatch(context.Background(), data, x, s); err != nil {
						t.Fatal(err)
					}
					got := s.logits[0]
					if math.IsNaN(float64(value)) {
						if !math.IsNaN(float64(got)) {
							t.Fatal("NaN router weight lost")
						}
					} else if got != value {
						t.Fatalf("nonfinite router: %g want %g", got, value)
					}
				}
			})
		}
	}
}

func TestSIMDKernelRoPETails(t *testing.T) {
	for _, dim := range []int{2, 14, 16, 18, 64, 66} {
		for _, pos := range []int{0, 17, 2047} {
			t.Run(fmt.Sprintf("dim%d/pos%d", dim, pos), func(t *testing.T) {
				q, k := make([]float32, 3*dim), make([]float32, 2*dim)
				for i := range q {
					q[i] = float32(i%31-15) / 32
				}
				for i := range k {
					k[i] = float32(i%23-11) / 64
				}
				wantQ, wantK := append([]float32(nil), q...), append([]float32(nil), k...)
				for _, dst := range [][]float32{wantQ, wantK} {
					for start := 0; start < len(dst); start += dim {
						for i := 0; i < dim/2; i++ {
							angle := float64(pos) / math.Pow(10000, float64(2*i)/float64(dim))
							c, s := float32(math.Cos(angle)), float32(math.Sin(angle))
							a, b := dst[start+i], dst[start+i+dim/2]
							dst[start+i], dst[start+i+dim/2] = a*c-b*s, a*s+b*c
						}
					}
				}
				ApplyRoPE(q, k, pos, 3, 2, dim, 10000)
				for n, pair := range [][2][]float32{{q, wantQ}, {k, wantK}} {
					for i, want := range pair[1] {
						if math.Abs(float64(pair[0][i]-want)) > 1e-7 {
							t.Fatalf("vector%d element%d=%g want %g", n, i, pair[0][i], want)
						}
					}
				}
			})
		}
	}
}

func BenchmarkSIMDKernelRouter(b *testing.B) {
	const hidden, experts, batch = 2880, 128, 15
	s := &moeBatchScratch{batch: batch, hidden: hidden, experts: experts,
		logits: make([]float32, batch*experts), routerBias: make([]float32, experts)}
	data, x := make([]byte, hidden*experts*4), make([]float32, hidden*batch)
	for i := range x {
		x[i] = float32(i%31-15) / 32
	}
	for i := 0; i < len(data)/4; i++ {
		binary.LittleEndian.PutUint32(data[i*4:], math.Float32bits(float32(i%23-11)/64))
	}
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := routeMoEBatch(ctx, data, x, s); err != nil {
			b.Fatal(err)
		}
	}
}

func TestSIMDKernelMXFP4DecodeAllEncodings(t *testing.T) {
	const blocks = 256 * 256
	row, decoded := make([]byte, blocks*17), make([]float32, blocks*32)
	for scale := 0; scale < 256; scale++ {
		for packed := 0; packed < 256; packed++ {
			start := (scale*256 + packed) * 17
			row[start] = byte(scale)
			for i := 1; i < 17; i++ {
				row[start+i] = byte(packed)
			}
		}
	}
	if !decodeMXFP4Fast(row, decoded) {
		t.Skip("AVX2/FMA decode unavailable; fallback covered by batch reference tests")
	}
	levels := [16]float32{0, .5, 1, 1.5, 2, 3, 4, 6, float32(math.Copysign(0, -1)), -.5, -1, -1.5, -2, -3, -4, -6}
	for scale := 0; scale < 256; scale++ {
		factor := float32(math.Ldexp(1, scale-127))
		for packed := 0; packed < 256; packed++ {
			for i := 0; i < 32; i++ {
				nibble := packed & 15
				if i >= 16 {
					nibble = packed >> 4
				}
				want, got := factor*levels[nibble], decoded[(scale*256+packed)*32+i]
				if math.IsNaN(float64(want)) {
					if !math.IsNaN(float64(got)) {
						t.Fatalf("scale%d packed%d element%d: expected NaN, got %g", scale, packed, i, got)
					}
				} else if math.Float32bits(got) != math.Float32bits(want) {
					t.Fatalf("scale%d packed%d element%d=%g want %g (bits %08x/%08x)", scale, packed, i, got, want, math.Float32bits(got), math.Float32bits(want))
				}
			}
		}
	}
}
