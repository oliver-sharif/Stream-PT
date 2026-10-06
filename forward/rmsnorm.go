//go:build goexperiment.simd

package forward

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"simd"

	"Stream-PT/ggufindex"
	ggufmmap "Stream-PT/ggufmap"
)

// RMSNormInto reuses y and supports exact in-place normalization (x == y).
// Partial overlap is rejected. On error y may be partially written.
// Do not close reader during computation.
func RMSNormInto(ctx context.Context, reader *ggufmmap.Reader, weight ggufindex.Tensor,
	x, y []float32, epsilon float32) error {
	if ctx == nil {
		return fmt.Errorf("nil context")
	}
	if reader == nil {
		return fmt.Errorf("nil GGUF reader")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if len(x) == 0 || len(y) != len(x) || len(weight.Shape) != 1 ||
		weight.Shape[0] != uint64(len(x)) {
		return fmt.Errorf("%q: normalization weight, input, and output dimensions differ",
			weight.Name)
	}
	if &x[0] != &y[0] && float32Overlap(x, y) {
		return fmt.Errorf("input and output partially overlap")
	}
	if epsilon < 0 || math.IsNaN(float64(epsilon)) || math.IsInf(float64(epsilon), 0) {
		return fmt.Errorf("invalid epsilon: %g", epsilon)
	}

	var bytesPerElement uint64
	switch weight.Type {
	case 0: // F32
		bytesPerElement = 4
	case 1, 28: // F16, BF16
		bytesPerElement = 2
	default:
		return fmt.Errorf("%q: unsupported normalization weight type %d",
			weight.Name, weight.Type)
	}
	if uint64(len(x)) > uint64(^uint(0)>>1)/bytesPerElement {
		return fmt.Errorf("%q: normalization weight exceeds platform limits", weight.Name)
	}
	if weight.Range.End <= weight.Range.Start ||
		weight.Range.End-weight.Range.Start != uint64(len(x))*bytesPerElement {
		return fmt.Errorf("%q: invalid weight data length", weight.Name)
	}

	// SIMD sum of squares; zero-pad the final partial vector.
	var squares simd.Float32s
	lanes := squares.Len()
	for offset := 0; offset < len(x); offset += lanes {
		if offset%4096 == 0 {
			if err := ctx.Err(); err != nil {
				return err
			}
		}
		end := offset + lanes
		if end > len(x) {
			end = len(x)
		}

		var values simd.Float32s
		if end-offset == lanes {
			values = simd.LoadFloat32s(x[offset:end])
		} else {
			values, _ = simd.LoadFloat32sPart(x[offset:end])
		}
		squares = values.MulAdd(values, squares)
	}
	partials := make([]float32, lanes)
	squares.Store(partials)

	var sum float32
	for _, value := range partials {
		sum += value
	}
	factor := float32(1) / float32(math.Sqrt(
		float64(sum/float32(len(x))+epsilon),
	))

	// Decode only one SIMD vector of weights at a time.
	weights := make([]float32, lanes)
	vectorFactor := simd.BroadcastFloat32s(factor)
	chunkBytes := uint64(DefaultQ40WindowBytes)
	chunkBytes -= chunkBytes % (uint64(lanes) * bytesPerElement)
	return reader.WithTensorChunks(weight, chunkBytes, func(byteOffset uint64, data []byte) error {
		first := int(byteOffset / bytesPerElement)
		count := len(data) / int(bytesPerElement)
		for local := 0; local < count; local += lanes {
			if err := ctx.Err(); err != nil {
				return err
			}
			offset := first + local
			end := first + min(local+lanes, count)

			clear(weights)
			for i := offset; i < end; i++ {
				index := i - first
				switch weight.Type {
				case 0:
					bits := binary.LittleEndian.Uint32(data[index*4:])
					weights[i-offset] = math.Float32frombits(bits)
				case 1:
					bits := binary.LittleEndian.Uint16(data[index*2:])
					weights[i-offset] = float16(bits)
				case 28:
					bits := binary.LittleEndian.Uint16(data[index*2:])
					weights[i-offset] = math.Float32frombits(
						uint32(bits) << 16,
					)
				}
			}

			var values, scales simd.Float32s
			if end-offset == lanes {
				values = simd.LoadFloat32s(x[offset:end])
				scales = simd.LoadFloat32s(weights)
			} else {
				values, _ = simd.LoadFloat32sPart(x[offset:end])
				scales, _ = simd.LoadFloat32sPart(weights[:end-offset])
			}

			result := values.Mul(scales).Mul(vectorFactor)
			if end-offset == lanes {
				result.Store(y[offset:end])
			} else {
				result.StorePart(y[offset:end])
			}
		}
		return nil
	})
}
