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

	// SIMD sum of squares with 4 unrolled accumulators.
	var squares simd.Float32s
	lanes := squares.Len()
	var sq0, sq1, sq2, sq3 simd.Float32s
	offset := 0
	for ; offset+4*lanes <= len(x); offset += 4 * lanes {
		if offset%4096 == 0 {
			if err := ctx.Err(); err != nil {
				return err
			}
		}
		v0 := simd.LoadFloat32s(x[offset : offset+lanes])
		v1 := simd.LoadFloat32s(x[offset+lanes : offset+2*lanes])
		v2 := simd.LoadFloat32s(x[offset+2*lanes : offset+3*lanes])
		v3 := simd.LoadFloat32s(x[offset+3*lanes : offset+4*lanes])
		sq0 = v0.MulAdd(v0, sq0)
		sq1 = v1.MulAdd(v1, sq1)
		sq2 = v2.MulAdd(v2, sq2)
		sq3 = v3.MulAdd(v3, sq3)
	}
	squares = sq0.Add(sq1).Add(sq2).Add(sq3)
	for ; offset < len(x); offset += lanes {
		end := min(offset+lanes, len(x))
		var values simd.Float32s
		if end-offset == lanes {
			values = simd.LoadFloat32s(x[offset:end])
		} else {
			values, _ = simd.LoadFloat32sPart(x[offset:end])
		}
		squares = values.MulAdd(values, squares)
	}
	var partials [32]float32
	squares.Store(partials[:lanes])

	var sum float32
	for _, value := range partials[:lanes] {
		sum += value
	}
	factor := float32(1) / float32(math.Sqrt(
		float64(sum/float32(len(x))+epsilon),
	))

	// Decode weights in reusable stack buffer without heap allocations.
	var weights [32]float32
	vectorFactor := simd.BroadcastFloat32s(factor)
	chunkBytes := uint64(DefaultQ40WindowBytes)
	chunkBytes -= chunkBytes % (uint64(lanes) * bytesPerElement)
	return reader.WithTensorChunks(weight, chunkBytes, func(byteOffset uint64, data []byte) error {
		first := int(byteOffset / bytesPerElement)
		count := len(data) / int(bytesPerElement)
		for local := 0; local < count; local += lanes {
			if local%4096 == 0 {
				if err := ctx.Err(); err != nil {
					return err
				}
			}
			offset := first + local
			end := first + min(local+lanes, count)

			var scales simd.Float32s
			if weight.Type == 0 && end-offset == lanes {
				raw := data[local*4 : (local+lanes)*4]
				for i := 0; i < lanes; i++ {
					weights[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
				}
				scales = simd.LoadFloat32s(weights[:lanes])
			} else {
				clear(weights[:lanes])
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
				if end-offset == lanes {
					scales = simd.LoadFloat32s(weights[:lanes])
				} else {
					scales, _ = simd.LoadFloat32sPart(weights[:end-offset])
				}
			}

			var values simd.Float32s
			if end-offset == lanes {
				values = simd.LoadFloat32s(x[offset:end])
			} else {
				values, _ = simd.LoadFloat32sPart(x[offset:end])
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
