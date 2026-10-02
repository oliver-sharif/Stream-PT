//go:build goexperiment.simd

package forward

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"runtime"
	"simd"
	"sync"

	"Stream-PT/ggufindex"
	ggufmmap "Stream-PT/ggufmap"
)

const (
	q80Elements   = 32
	q80BlockBytes = 34
)

// MulQ80Into computes y = W*x where W is a Q8_0 quantized matrix.
func MulQ80Into(
	ctx context.Context,
	reader *ggufmmap.Reader,
	tensor ggufindex.Tensor,
	x, y []float32,
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
	input, output, rowBytes, err := q80Shape(tensor)
	if err != nil {
		return err
	}
	if len(x) != input || len(y) != output {
		return fmt.Errorf("%q: expected %d input and %d output elements, got %d and %d",
			tensor.Name, input, output, len(x), len(y))
	}
	if float32Overlap(x, y) {
		return fmt.Errorf("input and output must not overlap")
	}

	windowBytes := options.WindowBytes
	if windowBytes == 0 {
		windowBytes = DefaultQ40WindowBytes
	}
	if windowBytes < uint64(rowBytes) {
		return fmt.Errorf("window of %d bytes cannot accommodate a %d-byte row", windowBytes, rowBytes)
	}

	rowsPerWindow := int(min(windowBytes/uint64(rowBytes), uint64(output)))
	workers := min(max(options.Workers, 1), runtime.GOMAXPROCS(0), rowsPerWindow)

	return reader.WithTensorChunks(tensor, uint64(rowsPerWindow)*uint64(rowBytes),
		func(offset uint64, data []byte) error {
			firstRow := int(offset / uint64(rowBytes))
			rows := len(data) / rowBytes
			compute := func(begin, end int) {
				for row := begin; row < end; row++ {
					if ctx.Err() != nil {
						return
					}
					encoded := data[row*rowBytes : (row+1)*rowBytes]
					y[firstRow+row] = dotQ80(encoded, x)
				}
			}
			active := min(workers, rows)
			if active == 1 {
				compute(0, rows)
			} else {
				var wg sync.WaitGroup
				for worker := 0; worker < active; worker++ {
					begin, end := worker*rows/active, (worker+1)*rows/active
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
	rowsPerWindow := int(min(windowBytes/uint64(rowBytes), uint64(output)))
	workers := min(max(options.Workers, 1), runtime.GOMAXPROCS(0), rowsPerWindow)

	bestToken = 0
	bestLogit = float32(math.Inf(-1))
	var mu sync.Mutex

	err = reader.WithTensorChunks(tensor, uint64(rowsPerWindow)*uint64(rowBytes),
		func(offset uint64, data []byte) error {
			firstRow := int(offset / uint64(rowBytes))
			rows := len(data) / rowBytes

			type workerResult struct {
				token int
				logit float32
			}
			results := make([]workerResult, workers)

			compute := func(workerID, begin, end int) {
				localBestToken := firstRow + begin
				localBestLogit := float32(math.Inf(-1))
				for row := begin; row < end; row++ {
					if ctx.Err() != nil {
						return
					}
					encoded := data[row*rowBytes : (row+1)*rowBytes]
					val := dotQ80(encoded, x)
					if val > localBestLogit {
						localBestLogit = val
						localBestToken = firstRow + row
					}
				}
				results[workerID] = workerResult{token: localBestToken, logit: localBestLogit}
			}

			active := min(workers, rows)
			if active == 1 {
				compute(0, 0, rows)
			} else {
				var wg sync.WaitGroup
				for worker := 0; worker < active; worker++ {
					begin, end := worker*rows/active, (worker+1)*rows/active
					wID := worker
					wg.Add(1)
					go func() {
						defer wg.Done()
						compute(wID, begin, end)
					}()
				}
				wg.Wait()
			}

			mu.Lock()
			for i := 0; i < active; i++ {
				if results[i].logit > bestLogit {
					bestLogit = results[i].logit
					bestToken = results[i].token
				}
			}
			mu.Unlock()

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
	bytes := in / q80Elements * q80BlockBytes
	if tensor.Range.End <= tensor.Range.Start || tensor.Range.End-tensor.Range.Start != out*bytes {
		return 0, 0, 0, fmt.Errorf("%q: invalid byte range", tensor.Name)
	}
	return int(in), int(out), int(bytes), nil
}

func dotQ80(row []byte, x []float32) float32 {
	var accumulator simd.Float32s
	lanes := accumulator.Len()
	partials := make([]float32, lanes)

	numBlocks := len(row) / q80BlockBytes
	for block := 0; block < numBlocks; block++ {
		begin := block * q80BlockBytes
		encoded := row[begin : begin+q80BlockBytes]

		scale := float16(binary.LittleEndian.Uint16(encoded[:2]))
		qs := encoded[2:]

		var weights [q80Elements]int32
		for i := 0; i < q80Elements; i++ {
			weights[i] = int32(int8(qs[i]))
		}
		vectorScale := simd.BroadcastFloat32s(scale)

		input := x[block*q80Elements : (block+1)*q80Elements]
		for offset := 0; offset < q80Elements; offset += lanes {
			end := offset + lanes
			if end > q80Elements {
				end = q80Elements
			}

			var quant simd.Int32s
			var v simd.Float32s
			if end-offset == lanes {
				quant = simd.LoadInt32s(weights[offset:end])
				v = simd.LoadFloat32s(input[offset:end])
			} else {
				quant, _ = simd.LoadInt32sPart(weights[offset:end])
				v, _ = simd.LoadFloat32sPart(input[offset:end])
			}
			w := quant.ConvertToFloat32().Mul(vectorScale)
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
