//go:build goexperiment.simd && amd64

package forward

import (
	"encoding/binary"
	"simd"
	"simd/archsimd"
)

func quantDotQ40(row []byte, x []float32) float32 {
	if archsimd.X86.AVX2() && archsimd.X86.FMA() && !simd.Emulated() {
		return quantQ40AVX2(row, x)
	}
	return quantDotPortable(row, x, true)
}

func quantDotQ80(row []byte, x []float32) float32 {
	if archsimd.X86.AVX2() && archsimd.X86.FMA() && !simd.Emulated() {
		return quantQ80AVX2(row, x)
	}
	return quantDotPortable(row, x, false)
}

// Widen packed bytes in registers; neither kernel materializes int32 weights.
// AVX2 also serves as the narrow path on AVX512 machines.
func quantQ40WeightsAVX2(q []byte, scale archsimd.Float32x8) (archsimd.Float32x8, archsimd.Float32x8, archsimd.Float32x8, archsimd.Float32x8) {
	packed := archsimd.LoadUint8x16(q)
	a := packed.ExtendLo8ToUint32()
	b := packed.ConcatShiftBytesRight(packed, 8).ExtendLo8ToUint32()
	mask := archsimd.BroadcastUint32x8(15)
	bias := archsimd.BroadcastInt32x8(8)
	return a.And(mask).AsInt32x8().Sub(bias).ConvertToFloat32().Mul(scale),
		b.And(mask).AsInt32x8().Sub(bias).ConvertToFloat32().Mul(scale),
		a.ShiftAllRight(4).AsInt32x8().Sub(bias).ConvertToFloat32().Mul(scale),
		b.ShiftAllRight(4).AsInt32x8().Sub(bias).ConvertToFloat32().Mul(scale)
}

func quantQ40AVX2(row []byte, x []float32) float32 {
	var accumulator archsimd.Float32x8
	for block := 0; block < len(row)/q40BlockBytes; block++ {
		encoded := row[block*q40BlockBytes : (block+1)*q40BlockBytes]
		scale := archsimd.BroadcastUint32x8(quantFloat16Bits(binary.LittleEndian.Uint16(encoded))).AsFloat32x8()
		w0, w1, w2, w3 := quantQ40WeightsAVX2(encoded[2:], scale)
		input := x[block*32 : (block+1)*32]
		accumulator = w0.MulAdd(archsimd.LoadFloat32x8(input[:8]), accumulator)
		accumulator = w1.MulAdd(archsimd.LoadFloat32x8(input[8:16]), accumulator)
		accumulator = w2.MulAdd(archsimd.LoadFloat32x8(input[16:24]), accumulator)
		accumulator = w3.MulAdd(archsimd.LoadFloat32x8(input[24:]), accumulator)
	}
	return quantSumAVX2(accumulator)
}

func quantQ80AVX2(row []byte, x []float32) float32 {
	var accumulator archsimd.Float32x8
	for block := 0; block < len(row)/q80BlockBytes; block++ {
		encoded := row[block*q80BlockBytes : (block+1)*q80BlockBytes]
		scale := archsimd.BroadcastUint32x8(quantFloat16Bits(binary.LittleEndian.Uint16(encoded))).AsFloat32x8()
		packed0 := archsimd.LoadUint8x16(encoded[2:18])
		packed1 := archsimd.LoadUint8x16(encoded[18:])
		w0 := packed0.AsInt8x16().ExtendLo8ToInt32().ConvertToFloat32().Mul(scale)
		w1 := packed0.ConcatShiftBytesRight(packed0, 8).AsInt8x16().ExtendLo8ToInt32().ConvertToFloat32().Mul(scale)
		w2 := packed1.AsInt8x16().ExtendLo8ToInt32().ConvertToFloat32().Mul(scale)
		w3 := packed1.ConcatShiftBytesRight(packed1, 8).AsInt8x16().ExtendLo8ToInt32().ConvertToFloat32().Mul(scale)
		input := x[block*32 : (block+1)*32]
		accumulator = w0.MulAdd(archsimd.LoadFloat32x8(input[:8]), accumulator)
		accumulator = w1.MulAdd(archsimd.LoadFloat32x8(input[8:16]), accumulator)
		accumulator = w2.MulAdd(archsimd.LoadFloat32x8(input[16:24]), accumulator)
		accumulator = w3.MulAdd(archsimd.LoadFloat32x8(input[24:]), accumulator)
	}
	return quantSumAVX2(accumulator)
}

func quantSumAVX2(accumulator archsimd.Float32x8) float32 {
	var partials [8]float32
	accumulator.StoreArray(&partials)
	var sum float32
	for _, partial := range partials {
		sum += partial
	}
	return sum
}

func quantDotQ40Batch(row []byte, x, y []float32, input, output, rowIndex, batch int) {
	if !archsimd.X86.AVX2() || !archsimd.X86.FMA() || simd.Emulated() {
		quantQ40BatchPortable(row, x, y, input, output, rowIndex, batch)
		return
	}
	const tile = 8
	for first := 0; first < batch; first += tile {
		count := min(tile, batch-first)
		var accumulators [tile]archsimd.Float32x8
		for block := 0; block < len(row)/q40BlockBytes; block++ {
			encoded := row[block*q40BlockBytes : (block+1)*q40BlockBytes]
			scale := archsimd.BroadcastUint32x8(quantFloat16Bits(binary.LittleEndian.Uint16(encoded))).AsFloat32x8()
			w0, w1, w2, w3 := quantQ40WeightsAVX2(encoded[2:], scale)
			for item := 0; item < count; item++ {
				start := (first+item)*input + block*32
				v := x[start : start+32]
				acc := accumulators[item]
				acc = w0.MulAdd(archsimd.LoadFloat32x8(v[:8]), acc)
				acc = w1.MulAdd(archsimd.LoadFloat32x8(v[8:16]), acc)
				acc = w2.MulAdd(archsimd.LoadFloat32x8(v[16:24]), acc)
				acc = w3.MulAdd(archsimd.LoadFloat32x8(v[24:]), acc)
				accumulators[item] = acc
			}
		}
		for item := 0; item < count; item++ {
			y[(first+item)*output+rowIndex] = quantSumAVX2(accumulators[item])
		}
	}
}
