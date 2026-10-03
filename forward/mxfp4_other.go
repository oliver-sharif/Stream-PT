//go:build goexperiment.simd && !amd64

package forward

import "context"

func dotMXFP4Fast(row []byte, x []float32) (float32, bool) {
	return 0, false
}

func dotMoEBatchDecodedFast(weights, x []float32) (float32, bool) {
	return 0, false
}

func decodeMXFP4Fast(row []byte, weights []float32) bool {
	return false
}

func computeMoEBatchPairFast(ctx context.Context, data []byte, input int,
	x []float32, xOffsets []int, bias, y []float32, yOffsets []int, begin, end int,
) bool {
	return false
}
