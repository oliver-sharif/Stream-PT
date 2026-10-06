//go:build goexperiment.simd

package forward

import (
	"context"
	"fmt"
	"math"
	"runtime"
	"unsafe"

	"Stream-PT/ggufindex"
	ggufmmap "Stream-PT/ggufmap"
)

const (
	q40Elements   = 32
	q40BlockBytes = 18
)

// DefaultQ40WindowBytes bounds the active weight window, excluding page alignment.
const DefaultQ40WindowBytes = 8 << 20

// Q40Options controls streaming, not the number of SIMD lanes or input vectors.
type Q40Options struct {
	// Workers defaults to one and is capped at GOMAXPROCS and window rows.
	Workers int
	// WindowBytes defaults to 8 MiB and must accommodate at least one encoded row.
	// Windows are rounded down to whole rows, never up beyond this limit.
	WindowBytes uint64
}

// MulQ40Into writes W*x into y. x and y must not overlap.
// On error y may be partially written. Do not close reader during computation.
func MulQ40Into(ctx context.Context, reader *ggufmmap.Reader, tensor ggufindex.Tensor,
	x, y []float32, options Q40Options) error {
	return MulQ40BatchInto(ctx, reader, tensor, x, y, 1, options)
}

// MulQ40BatchInto streams W once per batch, reusing decoded blocks across inputs.
// x and y contain batch consecutive vectors of Shape[0] and Shape[1] elements,
// respectively, and must not overlap. Input/output storage belongs to the caller.
// Small SIMD-sized tiles bound scratch space independently of the batch size.
// This is useful for prefill or independent sequences, not dependent decode steps.
// On error y may be partially written. Do not close reader during computation.
func MulQ40BatchInto(ctx context.Context, reader *ggufmmap.Reader, tensor ggufindex.Tensor,
	x, y []float32, batch int, options Q40Options) error {
	if ctx == nil {
		return fmt.Errorf("nil context")
	}
	if reader == nil {
		return fmt.Errorf("nil GGUF reader")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	input, output, rowBytes, err := q40Shape(tensor)
	if err != nil {
		return err
	}
	maxInt := int(^uint(0) >> 1)
	if batch < 1 || batch > maxInt/input || batch > maxInt/output {
		return fmt.Errorf("invalid or oversized batch: %d", batch)
	}
	if len(x) != batch*input || len(y) != batch*output {
		return fmt.Errorf("%q: expected %d input and %d output elements, got %d and %d",
			tensor.Name, batch*input, batch*output, len(x), len(y))
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

	pool := newQuantWorkers(workers, func(_ int, job quantRowJob) {
		for row := job.begin; row < job.end; row++ {
			if ctx.Err() != nil {
				return
			}
			encoded := job.data[row*rowBytes : (row+1)*rowBytes]
			if batch == 1 {
				y[job.firstRow+row] = dotQ40(encoded, x)
			} else {
				dotQ40Batch(encoded, x, y, input, output, job.firstRow+row, batch)
			}
		}
	})
	defer pool.close()
	return reader.WithTensorChunks(tensor, uint64(rowsPerWindow)*uint64(rowBytes),
		func(offset uint64, data []byte) error {
			firstRow := int(offset / uint64(rowBytes))
			rows := len(data) / rowBytes
			pool.run(data, firstRow, rows)
			return ctx.Err()
		})
}

func q40Shape(tensor ggufindex.Tensor) (input, output, rowBytes int, err error) {
	if tensor.Type != 2 {
		return 0, 0, 0, fmt.Errorf("%q: GGML type %d instead of Q4_0 (2)",
			tensor.Name, tensor.Type)
	}
	if len(tensor.Shape) != 2 {
		return 0, 0, 0, fmt.Errorf("%q: expected a two-dimensional matrix",
			tensor.Name)
	}
	maxInt := uint64(^uint(0) >> 1)
	in, out := tensor.Shape[0], tensor.Shape[1]
	if in == 0 || out == 0 || in%q40Elements != 0 {
		return 0, 0, 0, fmt.Errorf("%q: invalid Q4_0 matrix shape %v",
			tensor.Name, tensor.Shape)
	}
	if in > maxInt || out > maxInt || in/q40Elements > maxInt/q40BlockBytes {
		return 0, 0, 0, fmt.Errorf("%q: matrix dimensions exceed platform limits",
			tensor.Name)
	}
	bytes := in / q40Elements * q40BlockBytes
	if out > ^uint64(0)/bytes || tensor.Range.End <= tensor.Range.Start ||
		tensor.Range.End-tensor.Range.Start != out*bytes {
		return 0, 0, 0, fmt.Errorf("%q: invalid tensor byte range for shape %v", tensor.Name, tensor.Shape)
	}
	return int(in), int(out), int(bytes), nil
}

func float32Overlap(x, y []float32) bool {
	if len(x) == 0 || len(y) == 0 {
		return false
	}
	a, b := uintptr(unsafe.Pointer(unsafe.SliceData(x))), uintptr(unsafe.Pointer(unsafe.SliceData(y)))
	return a < b+uintptr(len(y))*4 && b < a+uintptr(len(x))*4
}

// dotQ40 decodes packed weights in SIMD registers where supported.
func dotQ40(row []byte, x []float32) float32 {
	return quantDotQ40(row, x)
}

func dotQ40Batch(row []byte, x, y []float32, input, output, rowIndex, batch int) {
	quantDotQ40Batch(row, x, y, input, output, rowIndex, batch)
}

func float16(bits uint16) float32 {
	sign := uint32(bits&0x8000) << 16
	exponent := uint32(bits>>10) & 0x1f
	mantissa := uint32(bits & 0x03ff)

	switch exponent {
	case 0:
		if mantissa == 0 {
			return math.Float32frombits(sign)
		}
		// Normalize a subnormal F16 value to F32.
		e := uint32(113)
		for mantissa&0x0400 == 0 {
			mantissa <<= 1
			e--
		}
		return math.Float32frombits(
			sign | e<<23 | (mantissa&0x03ff)<<13,
		)
	case 31:
		return math.Float32frombits(
			sign | 0x7f800000 | mantissa<<13,
		)
	default:
		return math.Float32frombits(
			sign | (exponent+112)<<23 | mantissa<<13,
		)
	}
}
