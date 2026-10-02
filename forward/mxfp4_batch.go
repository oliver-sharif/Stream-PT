//go:build goexperiment.simd

package forward

import (
	"context"
	"fmt"
	"runtime"
	"simd"
	"sync"

	"Stream-PT/ggufindex"
	ggufmmap "Stream-PT/ggufmap"
)

const moeBatchTile = 16

func validateMoEBatchExpert(t ggufindex.Tensor, input, output, experts int) error {
	if t.Type != 39 || len(t.Shape) != 3 || t.Shape[0] != uint64(input) ||
		t.Shape[1] != uint64(output) || t.Shape[2] != uint64(experts) || input%mxfp4Elements != 0 {
		return fmt.Errorf("%q: expected MXFP4 expert tensor [%d %d %d]", t.Name, input, output, experts)
	}
	n, err := moeBatchProduct(input/mxfp4Elements, mxfp4BlockBytes, output, experts)
	if err != nil || t.Range.End < t.Range.Start || t.Range.End-t.Range.Start != uint64(n) {
		return fmt.Errorf("%q: invalid expert range or overflow", t.Name)
	}
	return nil
}

func withMoEBatchExpert(reader *ggufmmap.Reader, t ggufindex.Tensor, expert int, fn func([]byte) error) error {
	// Dimensions and the complete range were checked before allocating scratch.
	size := t.Shape[0] / mxfp4Elements * mxfp4BlockBytes * t.Shape[1]
	t.Range.Start += uint64(expert) * size
	t.Range.End = t.Range.Start + size
	t.Shape = t.Shape[:2]
	return reader.WithTensor(t, fn)
}

// mulMoEBatchExpert shares each decoded block between all tokens in a tile.
// Offsets allow gathering routed tokens and scattering into top-k rank slots.
func mulMoEBatchExpert(ctx context.Context, data []byte, input, output int,
	x []float32, xOffsets []int, bias, y []float32, yOffsets []int, options Q40Options,
) error {
	workers := min(max(options.Workers, 1), runtime.GOMAXPROCS(0), output)
	compute := func(begin, end int) {
		computeMoEBatchExpertRows(ctx, data, input, x, xOffsets, bias, y, yOffsets, begin, end)
	}
	if workers == 1 {
		compute(0, output)
	} else {
		var wg sync.WaitGroup
		for worker := 0; worker < workers; worker++ {
			begin, end := worker*output/workers, (worker+1)*output/workers
			wg.Add(1)
			go func() {
				defer wg.Done()
				compute(begin, end)
			}()
		}
		wg.Wait()
	}
	return ctx.Err()
}

func computeMoEBatchExpertRows(ctx context.Context, data []byte, input int,
	x []float32, xOffsets []int, bias, y []float32, yOffsets []int, begin, end int,
) {
	rowBytes := input / mxfp4Elements * mxfp4BlockBytes
	weights := make([]float32, input)
	for row := begin; row < end; row++ {
		if ctx.Err() != nil {
			return
		}
		rowData := data[row*rowBytes : (row+1)*rowBytes]
		for block := 0; block < input/mxfp4Elements; block++ {
			if block%64 == 0 && ctx.Err() != nil {
				return
			}
			encoded := rowData[block*mxfp4BlockBytes : (block+1)*mxfp4BlockBytes]
			table := &scaledFP4Table[encoded[0]]
			base := block * mxfp4Elements
			for i, packed := range encoded[1:] {
				weights[base+i], weights[base+i+16] = table[packed&15], table[packed>>4]
			}
		}
		for token, start := range xOffsets {
			y[yOffsets[token]+row] = dotMoEBatchDecoded(weights, x[start:start+input]) + bias[row]
		}
	}
}

func dotMoEBatchDecoded(weights, x []float32) float32 {
	var acc simd.Float32s
	lanes := acc.Len()
	for block := 0; block < len(weights)/mxfp4Elements; block++ {
		base := block * mxfp4Elements
		for offset := 0; offset < mxfp4Elements; offset += lanes {
			end := min(offset+lanes, mxfp4Elements)
			var w, v simd.Float32s
			if end-offset == lanes {
				w = simd.LoadFloat32s(weights[base+offset : base+end])
				v = simd.LoadFloat32s(x[base+offset : base+end])
			} else {
				w, _ = simd.LoadFloat32sPart(weights[base+offset : base+end])
				v, _ = simd.LoadFloat32sPart(x[base+offset : base+end])
			}
			acc = w.MulAdd(v, acc)
		}
	}
	partials := make([]float32, lanes)
	acc.Store(partials)
	var sum float32
	for _, p := range partials {
		sum += p
	}
	return sum
}
