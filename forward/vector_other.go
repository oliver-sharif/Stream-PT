//go:build goexperiment.simd && !amd64

package forward

import "context"

func routeMoEBatchFast(ctx context.Context, data []byte, x []float32, s *moeBatchScratch) (bool, error) {
	return false, nil
}

func rotateRoPEFast(head, cosTable, sinTable []float32) int {
	return 0
}
