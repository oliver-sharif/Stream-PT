//go:build goexperiment.simd && amd64

package forward

import (
	"simd"
	"simd/archsimd"
)

func dotMXFP4Fast(row []byte, x []float32) (float32, bool) {
	if !archsimd.X86.AVX2() || !archsimd.X86.FMA() || simd.Emulated() {
		return 0, false
	}
	var a0, a1, a2, a3 archsimd.Float32x8
	for block := 0; block < len(row)/mxfp4BlockBytes; block++ {
		encoded := row[block*mxfp4BlockBytes : (block+1)*mxfp4BlockBytes]
		w0, w1, w2, w3 := mxfp4BlockAVX2(encoded)
		input := x[block*mxfp4Elements : (block+1)*mxfp4Elements]
		a0 = w0.MulAdd(archsimd.LoadFloat32x8(input[:8]), a0)
		a1 = w1.MulAdd(archsimd.LoadFloat32x8(input[8:16]), a1)
		a2 = w2.MulAdd(archsimd.LoadFloat32x8(input[16:24]), a2)
		a3 = w3.MulAdd(archsimd.LoadFloat32x8(input[24:]), a3)
	}
	return quantSumAVX2(a0.Add(a1).Add(a2).Add(a3)), true
}

func dotMoEBatchDecodedFast(weights, x []float32) (float32, bool) {
	if !archsimd.X86.AVX2() || !archsimd.X86.FMA() || simd.Emulated() {
		return 0, false
	}
	var a0, a1, a2, a3 archsimd.Float32x8
	for offset := 0; offset < len(weights); offset += mxfp4Elements {
		w, v := weights[offset:offset+32], x[offset:offset+32]
		a0 = archsimd.LoadFloat32x8(w[:8]).MulAdd(archsimd.LoadFloat32x8(v[:8]), a0)
		a1 = archsimd.LoadFloat32x8(w[8:16]).MulAdd(archsimd.LoadFloat32x8(v[8:16]), a1)
		a2 = archsimd.LoadFloat32x8(w[16:24]).MulAdd(archsimd.LoadFloat32x8(v[16:24]), a2)
		a3 = archsimd.LoadFloat32x8(w[24:]).MulAdd(archsimd.LoadFloat32x8(v[24:]), a3)
	}
	return quantSumAVX2(a0.Add(a1).Add(a2).Add(a3)), true
}

func mxfp4BlockAVX2(encoded []byte) (archsimd.Float32x8, archsimd.Float32x8, archsimd.Float32x8, archsimd.Float32x8) {
	levels := archsimd.LoadFloat32x8(fp4E2M1Table[:8])
	mask, sign := archsimd.BroadcastUint32x8(7), archsimd.BroadcastUint32x8(8)
	packed := archsimd.LoadUint8x16(encoded[1:])
	a := packed.ExtendLo8ToUint32()
	b := packed.ConcatShiftBytesRight(packed, 8).ExtendLo8ToUint32()
	hiA, hiB := a.ShiftAllRight(4), b.ShiftAllRight(4)
	scale := archsimd.BroadcastFloat32x8(e8m0ScaleTable[encoded[0]])
	// Permute the positive E2M1 levels and apply each nibble's sign in registers.
	return levels.Permute(a.And(mask)).ToBits().Xor(a.And(sign).ShiftAllLeft(28)).AsFloat32x8().Mul(scale),
		levels.Permute(b.And(mask)).ToBits().Xor(b.And(sign).ShiftAllLeft(28)).AsFloat32x8().Mul(scale),
		levels.Permute(hiA.And(mask)).ToBits().Xor(hiA.And(sign).ShiftAllLeft(28)).AsFloat32x8().Mul(scale),
		levels.Permute(hiB.And(mask)).ToBits().Xor(hiB.And(sign).ShiftAllLeft(28)).AsFloat32x8().Mul(scale)
}

func decodeMXFP4Fast(row []byte, weights []float32) bool {
	if !archsimd.X86.AVX2() || !archsimd.X86.FMA() || simd.Emulated() {
		return false
	}
	for block := 0; block < len(row)/mxfp4BlockBytes; block++ {
		encoded := row[block*mxfp4BlockBytes : (block+1)*mxfp4BlockBytes]
		w0, w1, w2, w3 := mxfp4BlockAVX2(encoded)
		out := weights[block*mxfp4Elements : (block+1)*mxfp4Elements]
		w0.Store(out[:8])
		w1.Store(out[8:16])
		w2.Store(out[16:24])
		w3.Store(out[24:])
	}
	return true
}
