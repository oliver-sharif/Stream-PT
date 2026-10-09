//go:build goexperiment.simd && amd64

package forward

import (
	"encoding/binary"
	"simd"
	"simd/archsimd"
)

func init() {
	if archsimd.X86.AVX2() && archsimd.X86.FMA() && !simd.Emulated() {
		quantDotQ80Pair = quantQ80PairAVX2
	}
}

// Keep each row's four accumulation chains and final reduction unchanged.
// Each activation vector feeds both rows without rounding the activations.
func quantQ80PairAVX2(rows []byte, x []float32) (float32, float32) {
	rowBytes := len(x) / q80Elements * q80BlockBytes
	r0, r1 := rows[:rowBytes], rows[rowBytes:2*rowBytes]
	var a0, a1, a2, a3, b0, b1, b2, b3 archsimd.Float32x8
	for block := 0; block < rowBytes/q80BlockBytes; block++ {
		e0 := (*[q80BlockBytes]byte)(r0[block*q80BlockBytes:])
		e1 := (*[q80BlockBytes]byte)(r1[block*q80BlockBytes:])
		s0 := archsimd.BroadcastUint32x8(quantFloat16Bits(binary.LittleEndian.Uint16(e0[:]))).AsFloat32x8()
		s1 := archsimd.BroadcastUint32x8(quantFloat16Bits(binary.LittleEndian.Uint16(e1[:]))).AsFloat32x8()
		input := (*[32]float32)(x[block*32:])
		p0 := archsimd.LoadUint8x16(e0[2:18])
		p1 := archsimd.LoadUint8x16(e1[2:18])
		v := archsimd.LoadFloat32x8(input[:8])
		a0 = p0.AsInt8x16().ExtendLo8ToInt32().ConvertToFloat32().Mul(s0).MulAdd(v, a0)
		b0 = p1.AsInt8x16().ExtendLo8ToInt32().ConvertToFloat32().Mul(s1).MulAdd(v, b0)
		v = archsimd.LoadFloat32x8(input[8:16])
		a1 = p0.ConcatShiftBytesRight(p0, 8).AsInt8x16().ExtendLo8ToInt32().ConvertToFloat32().Mul(s0).MulAdd(v, a1)
		b1 = p1.ConcatShiftBytesRight(p1, 8).AsInt8x16().ExtendLo8ToInt32().ConvertToFloat32().Mul(s1).MulAdd(v, b1)
		p0 = archsimd.LoadUint8x16(e0[18:])
		p1 = archsimd.LoadUint8x16(e1[18:])
		v = archsimd.LoadFloat32x8(input[16:24])
		a2 = p0.AsInt8x16().ExtendLo8ToInt32().ConvertToFloat32().Mul(s0).MulAdd(v, a2)
		b2 = p1.AsInt8x16().ExtendLo8ToInt32().ConvertToFloat32().Mul(s1).MulAdd(v, b2)
		v = archsimd.LoadFloat32x8(input[24:])
		a3 = p0.ConcatShiftBytesRight(p0, 8).AsInt8x16().ExtendLo8ToInt32().ConvertToFloat32().Mul(s0).MulAdd(v, a3)
		b3 = p1.ConcatShiftBytesRight(p1, 8).AsInt8x16().ExtendLo8ToInt32().ConvertToFloat32().Mul(s1).MulAdd(v, b3)
	}
	return quantSumAVX2(a0.Add(a1).Add(a2).Add(a3)), quantSumAVX2(b0.Add(b1).Add(b2).Add(b3))
}

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
	packed := archsimd.LoadUint8x16((*[16]byte)(q)[:])
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
	var a0, a1, a2, a3 archsimd.Float32x8
	for block := 0; block < len(row)/q40BlockBytes; block++ {
		encoded := (*[q40BlockBytes]byte)(row[block*q40BlockBytes:])
		scale := archsimd.BroadcastUint32x8(quantFloat16Bits(binary.LittleEndian.Uint16(encoded[:]))).AsFloat32x8()
		w0, w1, w2, w3 := quantQ40WeightsAVX2(encoded[2:], scale)
		input := (*[32]float32)(x[block*32:])
		a0 = w0.MulAdd(archsimd.LoadFloat32x8(input[:8]), a0)
		a1 = w1.MulAdd(archsimd.LoadFloat32x8(input[8:16]), a1)
		a2 = w2.MulAdd(archsimd.LoadFloat32x8(input[16:24]), a2)
		a3 = w3.MulAdd(archsimd.LoadFloat32x8(input[24:]), a3)
	}
	return quantSumAVX2(a0.Add(a1).Add(a2).Add(a3))
}

func quantQ80AVX2(row []byte, x []float32) float32 {
	var a0, a1, a2, a3 archsimd.Float32x8
	for block := 0; block < len(row)/q80BlockBytes; block++ {
		encoded := (*[q80BlockBytes]byte)(row[block*q80BlockBytes:])
		scale := archsimd.BroadcastUint32x8(quantFloat16Bits(binary.LittleEndian.Uint16(encoded[:]))).AsFloat32x8()
		packed0 := archsimd.LoadUint8x16(encoded[2:18])
		packed1 := archsimd.LoadUint8x16(encoded[18:])
		w0 := packed0.AsInt8x16().ExtendLo8ToInt32().ConvertToFloat32().Mul(scale)
		w1 := packed0.ConcatShiftBytesRight(packed0, 8).AsInt8x16().ExtendLo8ToInt32().ConvertToFloat32().Mul(scale)
		w2 := packed1.AsInt8x16().ExtendLo8ToInt32().ConvertToFloat32().Mul(scale)
		w3 := packed1.ConcatShiftBytesRight(packed1, 8).AsInt8x16().ExtendLo8ToInt32().ConvertToFloat32().Mul(scale)
		input := (*[32]float32)(x[block*32:])
		a0 = w0.MulAdd(archsimd.LoadFloat32x8(input[:8]), a0)
		a1 = w1.MulAdd(archsimd.LoadFloat32x8(input[8:16]), a1)
		a2 = w2.MulAdd(archsimd.LoadFloat32x8(input[16:24]), a2)
		a3 = w3.MulAdd(archsimd.LoadFloat32x8(input[24:]), a3)
	}
	return quantSumAVX2(a0.Add(a1).Add(a2).Add(a3))
}

func quantSumAVX2(accumulator archsimd.Float32x8) float32 {
	pairs := accumulator.GetLo().Add(accumulator.GetHi())
	pairs = pairs.ConcatAddPairs(pairs)
	return pairs.ConcatAddPairs(pairs).GetElem(0)
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
			encoded := (*[q40BlockBytes]byte)(row[block*q40BlockBytes:])
			scale := archsimd.BroadcastUint32x8(quantFloat16Bits(binary.LittleEndian.Uint16(encoded[:]))).AsFloat32x8()
			w0, w1, w2, w3 := quantQ40WeightsAVX2(encoded[2:], scale)
			for item := 0; item < count; item++ {
				start := (first+item)*input + block*32
				v := (*[32]float32)(x[start:])
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
