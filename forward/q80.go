//go:build goexperiment.simd

package forward

import (
	"context"
	"fmt"
	"math"
	"runtime"

	"Stream-PT/ggufindex"
	ggufmmap "Stream-PT/ggufmap"
)

const (
	q80Elements   = 32
	q80BlockBytes = 34
)

// MulQ80Argmax computes argmax(W*x) while streaming W, tracking the top token
// without storing the complete logits vector.
func MulQ80Argmax(
	ctx context.Context,
	reader *ggufmmap.Reader,
	tensor ggufindex.Tensor,
	x []float32,
	options Q40Options,
) (bestToken int, bestLogit float32, err error) {
	if ctx == nil {
		return 0, 0, fmt.Errorf("nil context")
	}
	if reader == nil {
		return 0, 0, fmt.Errorf("nil reader")
	}
	if err := ctx.Err(); err != nil {
		return 0, 0, err
	}
	input, output, rowBytes, err := q80Shape(tensor)
	if err != nil {
		return 0, 0, err
	}
	if len(x) != input {
		return 0, 0, fmt.Errorf("%q: expected %d input elements, got %d", tensor.Name, input, len(x))
	}

	windowBytes := options.WindowBytes
	if windowBytes == 0 {
		windowBytes = DefaultQ40WindowBytes
	}
	if windowBytes < uint64(rowBytes) {
		return 0, 0, fmt.Errorf("window of %d bytes cannot accommodate a %d-byte row", windowBytes, rowBytes)
	}
	rowsPerWindow := int(min(windowBytes/uint64(rowBytes), uint64(output)))
	workers := min(max(options.Workers, 1), runtime.GOMAXPROCS(0), rowsPerWindow)

	bestToken = 0
	bestLogit = float32(math.Inf(-1))
	type workerResult struct {
		token int
		logit float32
	}
	results := make([]workerResult, workers)
	pool := newQuantWorkers(workers, func(worker int, job quantRowJob) {
		local := workerResult{token: job.firstRow + job.begin, logit: float32(math.Inf(-1))}
		for row := job.begin; row < job.end; row++ {
			if ctx.Err() != nil {
				break
			}
			encoded := job.data[row*rowBytes : (row+1)*rowBytes]
			val := dotQ80(encoded, x)
			if val > local.logit {
				local = workerResult{token: job.firstRow + row, logit: val}
			}
		}
		results[worker] = local
	})
	defer pool.close()

	err = reader.WithTensorChunks(tensor, uint64(rowsPerWindow)*uint64(rowBytes),
		func(offset uint64, data []byte) error {
			firstRow := int(offset / uint64(rowBytes))
			rows := len(data) / rowBytes
			pool.run(data, firstRow, rows)
			active := min(workers, rows)
			for i := 0; i < active; i++ {
				if results[i].logit > bestLogit ||
					(results[i].logit == bestLogit && results[i].token < bestToken) {
					bestLogit = results[i].logit
					bestToken = results[i].token
				}
			}
			return ctx.Err()
		})

	return bestToken, bestLogit, err
}

func q80Shape(tensor ggufindex.Tensor) (input, output, rowBytes int, err error) {
	if tensor.Type != 8 {
		return 0, 0, 0, fmt.Errorf("%q: GGML type %d instead of Q8_0 (8)", tensor.Name, tensor.Type)
	}
	if len(tensor.Shape) != 2 {
		return 0, 0, 0, fmt.Errorf("%q: expected 2D matrix", tensor.Name)
	}
	in, out := tensor.Shape[0], tensor.Shape[1]
	if in == 0 || out == 0 || in%q80Elements != 0 {
		return 0, 0, 0, fmt.Errorf("%q: invalid Q8_0 shape %v", tensor.Name, tensor.Shape)
	}
	maxInt := uint64(^uint(0) >> 1)
	if in > maxInt || out > maxInt || in/q80Elements > maxInt/q80BlockBytes {
		return 0, 0, 0, fmt.Errorf("%q: matrix dimensions exceed platform limits", tensor.Name)
	}
	bytes := in / q80Elements * q80BlockBytes
	if out > ^uint64(0)/bytes || tensor.Range.End <= tensor.Range.Start || tensor.Range.End-tensor.Range.Start != out*bytes {
		return 0, 0, 0, fmt.Errorf("%q: invalid byte range", tensor.Name)
	}
	return int(in), int(out), int(bytes), nil
}

func dotQ80(row []byte, x []float32) float32 {
	return quantDotQ80(row, x)
}
