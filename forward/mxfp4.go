//go:build goexperiment.simd

package forward

import (
	"context"
	"fmt"
	"math"
	"runtime"
	"simd"
	"sync"

	"Stream-PT/ggufindex"
	ggufmmap "Stream-PT/ggufmap"
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

// MulMXFP4Expert computes y = W_expert * x + bias for a specific expert index.
func MulMXFP4Expert(
	ctx context.Context,
	reader *ggufmmap.Reader,
	tensor ggufindex.Tensor,
	expertIdx int,
	x, bias, y []float32,
	options Q40Options,
) error {
	if ctx == nil {
		return fmt.Errorf("nil context")
	}
	if reader == nil {
		return fmt.Errorf("nil reader")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if len(tensor.Shape) != 3 {
		return fmt.Errorf("%q: expected 3D expert tensor, got shape %v", tensor.Name, tensor.Shape)
	}
	inDim, outDim, numExperts := int(tensor.Shape[0]), int(tensor.Shape[1]), int(tensor.Shape[2])
	if expertIdx < 0 || expertIdx >= numExperts {
		return fmt.Errorf("expert index %d out of bounds [0, %d)", expertIdx, numExperts)
	}
	if len(x) != inDim || len(y) != outDim {
		return fmt.Errorf("%q: dimension mismatch (x=%d in=%d, y=%d out=%d)",
			tensor.Name, len(x), inDim, len(y), outDim)
	}
	if err := validateMoEBatchExpert(tensor, inDim, outDim, numExperts); err != nil {
		return err
	}

	blocksPerRow := inDim / mxfp4Elements
	rowBytes := uint64(blocksPerRow * mxfp4BlockBytes)
	expertBytes := uint64(outDim) * rowBytes
	expertOffset := uint64(expertIdx) * expertBytes

	span := ggufindex.Range{
		File:  tensor.Range.File,
		Start: tensor.Range.Start + expertOffset,
		End:   tensor.Range.Start + expertOffset + expertBytes,
	}

	return reader.WithExpertRanges([]ggufindex.Range{span}, func(_ int, data []byte) error {
		if uint64(len(data)) < expertBytes {
			return fmt.Errorf("short read for expert %d: %d < %d", expertIdx, len(data), expertBytes)
		}
		workers := min(max(options.Workers, 1), runtime.GOMAXPROCS(0), outDim)
		compute := func(begin, end int) {
			for r := begin; r < end; r++ {
				if ctx.Err() != nil {
					return
				}
				rowData := data[uint64(r)*rowBytes : uint64(r+1)*rowBytes]
				val := dotMXFP4(rowData, x)
				if bias != nil && len(bias) > r {
					val += bias[r]
				}
				y[r] = val
			}
		}
		if workers <= 1 {
			compute(0, outDim)
		} else {
			var wg sync.WaitGroup
			for w := 0; w < workers; w++ {
				begin, end := w*outDim/workers, (w+1)*outDim/workers
				wg.Add(1)
				go func() {
					defer wg.Done()
					compute(begin, end)
				}()
			}
			wg.Wait()
		}
		return ctx.Err()
	})
}

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
