//go:build goexperiment.simd

package tests

import (
	"context"
	"fmt"
	"math"
	"testing"

	"Stream-PT/forward"
)

func TestMXFP4NibbleReference(t *testing.T) {
	levels := [...]float64{0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6}
	for _, input := range []int{32, 96, 2880} {
		t.Run(fmt.Sprintf("input%d", input), func(t *testing.T) {
			const rows = 256
			blocks := input / 32
			data, x := make([]byte, rows*blocks*17), make([]float32, input)
			for i := range x {
				x[i] = float32(i%23-11) / 32
			}
			want := make([]float64, rows)
			for row := range rows {
				for block := range blocks {
					encoded := data[(row*blocks+block)*17:][:17]
					encoded[0] = byte(120 + block%12)
					scale := math.Ldexp(1, int(encoded[0])-127)
					for i := range 16 {
						packed := byte(row + block*7 + i*11)
						encoded[i+1] = packed
						want[row] += scale * levels[packed&15] * float64(x[block*32+i])
						want[row] += scale * levels[packed>>4] * float64(x[block*32+i+16])
					}
				}
			}
			reader, tensor := tensorFixture(t, 39, []uint64{uint64(input), rows, 1}, data)
			out := make([]float32, rows)
			for _, workers := range []int{1, 8} {
				if err := forward.MulMXFP4Expert(context.Background(), reader, tensor, 0, x, nil, out, forward.Q40Options{Workers: workers}); err != nil {
					t.Fatal(err)
				}
				for row, expected := range want {
					if math.IsNaN(float64(out[row])) || math.Abs(float64(out[row])-expected) > 3e-5*max(1, math.Abs(expected)) {
						t.Fatalf("workers%d row%d=%g want %g", workers, row, out[row], expected)
					}
				}
			}
		})
	}
}

func TestMXFP4AllScales(t *testing.T) {
	data := make([]byte, 257*17)
	for scale := range 256 {
		data[scale*17] = byte(scale)
		for i := 1; i < 17; i++ {
			data[scale*17+i] = 0x22
		}
	}
	data[256*17] = 255 // Inf scale times zero must remain NaN, not become zero.
	x, out := make([]float32, 32), make([]float32, 257)
	for i := range x {
		x[i] = 1.0 / 32
	}
	reader, tensor := tensorFixture(t, 39, []uint64{32, 257, 1}, data)
	if err := forward.MulMXFP4Expert(context.Background(), reader, tensor, 0, x, nil, out, forward.Q40Options{Workers: 8}); err != nil {
		t.Fatal(err)
	}
	for scale := range 256 {
		want := float32(math.Ldexp(1, scale-127))
		if out[scale] != want {
			t.Fatalf("scale%d=%g want %g", scale, out[scale], want)
		}
	}
	if !math.IsNaN(float64(out[256])) {
		t.Fatalf("Inf scale times zero: %g", out[256])
	}
}
