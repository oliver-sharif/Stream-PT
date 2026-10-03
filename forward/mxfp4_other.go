//go:build goexperiment.simd && !amd64

package forward

func dotMXFP4Fast(row []byte, x []float32) (float32, bool) {
	return 0, false
}

func dotMoEBatchDecodedFast(weights, x []float32) (float32, bool) {
	return 0, false
}

func decodeMXFP4Fast(row []byte, weights []float32) bool {
	return false
}
