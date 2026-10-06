//go:build goexperiment.simd

package forward

import (
	"math"
	"simd"
)

// Precomputed E8M0 scale table (2^(b - 127)).
var e8m0ScaleTable [256]float32

// FP4 E2M1 lookup table.
var fp4E2M1Table = [16]float32{
	0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
	-0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
}

var scaledFP4Table [256][16]float32

func init() {
	for i := 0; i < 256; i++ {
		e8m0ScaleTable[i] = float32(math.Ldexp(1.0, i-127))
		for j, value := range fp4E2M1Table {
			scaledFP4Table[i][j] = e8m0ScaleTable[i] * value
		}
	}
}

const (
	mxfp4Elements   = 32
	mxfp4BlockBytes = 17
)

func dotMXFP4(row []byte, x []float32) float32 {
	if value, ok := dotMXFP4Fast(row, x); ok {
		return value
	}
	var accumulator simd.Float32s
	lanes := accumulator.Len()
	var local [64]float32
	partials := local[:min(lanes, len(local))]
	if lanes > len(local) {
		partials = make([]float32, lanes)
	}

	numBlocks := len(row) / mxfp4BlockBytes
	for block := 0; block < numBlocks; block++ {
		begin := block * mxfp4BlockBytes
		encoded := row[begin : begin+mxfp4BlockBytes]

		table := &scaledFP4Table[encoded[0]]
		quantized := encoded[1:]

		var weights [mxfp4Elements]float32
		for i, packed := range quantized {
			weights[i] = table[packed&0x0f]
			weights[i+16] = table[packed>>4]
		}

		input := x[block*mxfp4Elements : (block+1)*mxfp4Elements]
		for offset := 0; offset < mxfp4Elements; offset += lanes {
			end := offset + lanes
			if end > mxfp4Elements {
				end = mxfp4Elements
			}

			var w, v simd.Float32s
			if end-offset == lanes {
				w = simd.LoadFloat32s(weights[offset:end])
				v = simd.LoadFloat32s(input[offset:end])
			} else {
				w, _ = simd.LoadFloat32sPart(weights[offset:end])
				v, _ = simd.LoadFloat32sPart(input[offset:end])
			}
			accumulator = w.MulAdd(v, accumulator)
		}
	}

	accumulator.Store(partials)
	var result float32
	for _, val := range partials {
		result += val
	}
	return result
}
