//go:build goexperiment.simd

package forward

import (
	"context"
	"fmt"
	"simd"

	"Stream-PT/ggufindex"
	ggufmmap "Stream-PT/ggufmap"
)

func mulQ40ForTest(ctx context.Context, reader *ggufmmap.Reader, tensor ggufindex.Tensor, x []float32, workers int) ([]float32, error) {
	y := make([]float32, int(tensor.Shape[1]))
	err := MulQ40Into(ctx, reader, tensor, x, y, Q40Options{Workers: workers})
	return y, err
}

func rmsNormForTest(ctx context.Context, reader *ggufmmap.Reader, tensor ggufindex.Tensor, x []float32, epsilon float32) ([]float32, error) {
	y := make([]float32, len(x))
	err := RMSNormInto(ctx, reader, tensor, x, y, epsilon)
	return y, err
}

func attentionForTest(q, k, v []float32, cache *KVCache, layer, pos, heads, kvHeads, dim int, out []float32, options AttentionOptions) {
	var vector simd.Float32s
	var scratch AttentionScratch
	scratch.prepare(pos+1, vector.Len())
	forwardAttention(q, k, v, cache, layer, pos, heads, kvHeads, dim, out, options, scratch.scores, scratch.partials)
}

func moeForTest(ctx context.Context, reader *ggufmmap.Reader, x []float32, cfg *Config,
	router, routerBias, gate, gateBias, up, upBias, down, downBias ggufindex.Tensor,
	out []float32, options Q40Options,
) error {
	return forwardMoE(ctx, reader, x, cfg, router, routerBias, gate, gateBias, up, upBias, down, downBias, out, options, nil)
}

func mulMXFP4ForTest(ctx context.Context, reader *ggufmmap.Reader, tensor ggufindex.Tensor, expert int, x, bias, y []float32, options Q40Options) error {
	input, output, experts := int(tensor.Shape[0]), int(tensor.Shape[1]), int(tensor.Shape[2])
	if err := validateMoEBatchExpert(tensor, input, output, experts); err != nil {
		return err
	}
	if expert < 0 || expert >= experts || len(x) != input || len(y) != output {
		return fmt.Errorf("invalid test expert dimensions")
	}
	if bias == nil {
		bias = make([]float32, output)
	}
	size := uint64(input / mxfp4Elements * mxfp4BlockBytes * output)
	span := ggufindex.Range{File: tensor.Range.File, Start: tensor.Range.Start + uint64(expert)*size}
	span.End = span.Start + size
	return reader.WithExpertRanges([]ggufindex.Range{span}, func(_ int, data []byte) error {
		return mulMoEBatchExpert(ctx, data, input, output, x, []int{0}, bias, y, []int{0}, options)
	})
}

// q80KernelRowsForTest checks every output of the kernel used by streaming argmax.
// Worker and window behavior is exercised separately through MulQ80Argmax.
func q80KernelRowsForTest(reader *ggufmmap.Reader, tensor ggufindex.Tensor, x, y []float32) error {
	return reader.WithTensor(tensor, func(data []byte) error {
		rowBytes := len(x) / q80Elements * q80BlockBytes
		for row := range y {
			y[row] = dotQ80(data[row*rowBytes:(row+1)*rowBytes], x)
		}
		return nil
	})
}
