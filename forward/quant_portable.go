//go:build goexperiment.simd

package forward

import (
	"encoding/binary"
	"math/bits"
	"simd"
)

// Keep scale conversion in integer registers, avoiding scalar SSE calls in an
// AVX loop. Subnormals, signed zero, infinities and NaN payloads are preserved.
func quantFloat16Bits(h uint16) uint32 {
	sign := uint32(h&0x8000) << 16
	exponent := uint32(h>>10) & 31
	fraction := uint32(h & 1023)
	if exponent == 31 {
		return sign | 0x7f800000 | fraction<<13
	}
	if exponent != 0 {
		return sign | (exponent+112)<<23 | fraction<<13
	}
	if fraction == 0 {
		return sign
	}
	shift := uint32(bits.LeadingZeros32(fraction) - 21)
	return sign | (113-shift)<<23 | (fraction<<shift&1023)<<13
}

func quantDotPortable(row []byte, x []float32, q40 bool) float32 {
	if simd.Emulated() {
		return quantDotScalar(row, x, q40)
	}
	var accumulator simd.Float32s
	lanes := accumulator.Len()
	var partials [32]float32
	blockBytes := q80BlockBytes
	if q40 {
		blockBytes = q40BlockBytes
	}
	for block := 0; block < len(row)/blockBytes; block++ {
		encoded := row[block*blockBytes : (block+1)*blockBytes]
		var weights [32]float32
		quantDecodePortable(encoded, &weights, q40)
		for offset := 0; offset < 32; offset += lanes {
			w := simd.LoadFloat32s(weights[offset:])
			v := simd.LoadFloat32s(x[block*32+offset:])
			accumulator = w.MulAdd(v, accumulator)
		}
	}
	accumulator.Store(partials[:lanes])
	var result float32
	for _, value := range partials[:lanes] {
		result += value
	}
	return result
}

func quantDecodePortable(encoded []byte, weights *[32]float32, q40 bool) {
	scale := float16(binary.LittleEndian.Uint16(encoded))
	if q40 {
		for i, packed := range encoded[2:] {
			weights[i] = float32(int(packed&15)-8) * scale
			weights[i+16] = float32(int(packed>>4)-8) * scale
		}
	} else {
		for i, q := range encoded[2:] {
			weights[i] = float32(int8(q)) * scale
		}
	}
}

func quantDotScalar(row []byte, x []float32, q40 bool) float32 {
	blockBytes := q80BlockBytes
	if q40 {
		blockBytes = q40BlockBytes
	}
	var sum float32
	for block := 0; block < len(row)/blockBytes; block++ {
		encoded := row[block*blockBytes : (block+1)*blockBytes]
		scale := float16(binary.LittleEndian.Uint16(encoded))
		if q40 {
			for i, packed := range encoded[2:] {
				sum += float32(int(packed&15)-8) * scale * x[block*32+i]
				sum += float32(int(packed>>4)-8) * scale * x[block*32+i+16]
			}
		} else {
			for i, q := range encoded[2:] {
				sum += float32(int8(q)) * scale * x[block*32+i]
			}
		}
	}
	return sum
}

func quantQ40BatchPortable(row []byte, x, y []float32, input, output, rowIndex, batch int) {
	if simd.Emulated() {
		for item := 0; item < batch; item++ {
			y[item*output+rowIndex] = quantDotScalar(row, x[item*input:(item+1)*input], true)
		}
		return
	}
	var vector simd.Float32s
	lanes := vector.Len()
	const maxTile = 8
	var partials [32]float32
	for first := 0; first < batch; first += maxTile {
		count := min(maxTile, batch-first)
		var accumulators [maxTile]simd.Float32s
		for block := 0; block < len(row)/q40BlockBytes; block++ {
			var weights [32]float32
			quantDecodePortable(row[block*q40BlockBytes:(block+1)*q40BlockBytes], &weights, true)
			for offset := 0; offset < 32; offset += lanes {
				w := simd.LoadFloat32s(weights[offset:])
				for item := 0; item < count; item++ {
					start := (first+item)*input + block*32 + offset
					v := simd.LoadFloat32s(x[start:])
					accumulators[item] = w.MulAdd(v, accumulators[item])
				}
			}
		}
		for item := 0; item < count; item++ {
			accumulators[item].Store(partials[:lanes])
			var sum float32
			for _, partial := range partials[:lanes] {
				sum += partial
			}
			y[(first+item)*output+rowIndex] = sum
		}
	}
}
