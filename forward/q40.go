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

// MulQ40 computes y = W*x without loading or fully dequantizing W.
// Shape[0] is the input dimension; Shape[1] is the output dimension.
// Use MulQ40Into to reuse the output allocation across inference steps.
func MulQ40(
	ctx context.Context,
	reader *ggufmmap.Reader,
	tensor ggufindex.Tensor,
	x []float32,
	workers int,
) ([]float32, error) {
	_, output, _, err := q40Shape(tensor)
	if err != nil {
		return nil, err
	}
	if ctx == nil {
		return nil, fmt.Errorf("nil context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	y := make([]float32, output)
	if err := MulQ40Into(ctx, reader, tensor, x, y, Q40Options{Workers: workers}); err != nil {
		return nil, err
	}
	return y, nil
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
					if batch == 1 {
						y[firstRow+row] = dotQ40(encoded, x)
					} else {
						dotQ40Batch(encoded, x, y, input, output, firstRow+row, batch)
					}
				}
			}
			// Contiguous row ranges avoid a channel operation per output element.
			// All workers join before the current mmap window is unmapped.
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

// dotQ40 decodes only one 18-byte block into a small local buffer at a time.
func dotQ40(row []byte, x []float32) float32 {
	var accumulator simd.Float32s
	lanes := accumulator.Len()
	partials := make([]float32, lanes)

	for block := 0; block < len(row)/q40BlockBytes; block++ {
		begin := block * q40BlockBytes
		encoded := row[begin : begin+q40BlockBytes]

		scale := float16(binary.LittleEndian.Uint16(encoded[:2]))
		quantized := encoded[2:]

		var weights [q40Elements]int32
		for i, packed := range quantized {
			// GGML Q4_0 stores the low 16 nibbles before the high 16 nibbles.
			weights[i] = int32(packed&0x0f) - 8
			weights[i+16] = int32(packed>>4) - 8
		}
		vectorScale := simd.BroadcastFloat32s(scale)

		input := x[block*q40Elements : (block+1)*q40Elements]
		for offset := 0; offset < q40Elements; offset += lanes {
			end := offset + lanes
			if end > q40Elements {
				end = q40Elements
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
	for _, value := range partials {
		result += value
	}
	return result
}

func dotQ40Batch(row []byte, x, y []float32, input, output, rowIndex, batch int) {
	var vector simd.Float32s
	lanes := vector.Len()
	// SIMD width determines the tile, not a memory-safe total input batch size.
	const maxTile = 8
	tileSize := min(lanes, maxTile)
	partials := make([]float32, lanes)
	for first := 0; first < batch; first += tileSize {
		count := min(tileSize, batch-first)
		var accumulators [maxTile]simd.Float32s
		for block := 0; block < len(row)/q40BlockBytes; block++ {
			encoded := row[block*q40BlockBytes : (block+1)*q40BlockBytes]
			scale := float16(binary.LittleEndian.Uint16(encoded[:2]))
			var weights [q40Elements]int32
			for i, packed := range encoded[2:] {
				weights[i] = int32(packed&15) - 8
				weights[i+16] = int32(packed>>4) - 8
			}
			vectorScale := simd.BroadcastFloat32s(scale)
			for offset := 0; offset < q40Elements; offset += lanes {
				end := min(offset+lanes, q40Elements)
				var quant simd.Int32s
				if end-offset == lanes {
					quant = simd.LoadInt32s(weights[offset:end])
				} else {
					quant, _ = simd.LoadInt32sPart(weights[offset:end])
				}
				w := quant.ConvertToFloat32().Mul(vectorScale)
				for item := 0; item < count; item++ {
					start := (first+item)*input + block*q40Elements + offset
					var v simd.Float32s
					if end-offset == lanes {
						v = simd.LoadFloat32s(x[start : start+lanes])
					} else {
						v, _ = simd.LoadFloat32sPart(x[start : start+end-offset])
					}
					accumulators[item] = w.MulAdd(v, accumulators[item])
				}
			}
		}
		for item := 0; item < count; item++ {
			accumulators[item].Store(partials)
			var sum float32
			for _, partial := range partials {
				sum += partial
			}
			y[(first+item)*output+rowIndex] = sum
		}
	}
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
