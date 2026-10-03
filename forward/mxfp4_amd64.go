//go:build goexperiment.simd && amd64

package forward

import (
	"context"
	"simd"
	"simd/archsimd"
)

func dotMXFP4Fast(row []byte, x []float32) (float32, bool) {
	if !archsimd.X86.AVX2() || !archsimd.X86.FMA() || simd.Emulated() {
		return 0, false
	}
	var a0, a1, a2, a3 archsimd.Float32x8
	for block := 0; block < len(row)/mxfp4BlockBytes; block++ {
		encoded := (*[mxfp4BlockBytes]byte)(row[block*mxfp4BlockBytes:])
		w0, w1, w2, w3 := mxfp4BlockAVX2(encoded[:])
		input := (*[mxfp4Elements]float32)(x[block*mxfp4Elements:])
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
		w, v := (*[32]float32)(weights[offset:]), (*[32]float32)(x[offset:])
		a0 = archsimd.LoadFloat32x8(w[:8]).MulAdd(archsimd.LoadFloat32x8(v[:8]), a0)
		a1 = archsimd.LoadFloat32x8(w[8:16]).MulAdd(archsimd.LoadFloat32x8(v[8:16]), a1)
		a2 = archsimd.LoadFloat32x8(w[16:24]).MulAdd(archsimd.LoadFloat32x8(v[16:24]), a2)
		a3 = archsimd.LoadFloat32x8(w[24:]).MulAdd(archsimd.LoadFloat32x8(v[24:]), a3)
	}
	return quantSumAVX2(a0.Add(a1).Add(a2).Add(a3)), true
}

func mxfp4BlockAVX2(encoded []byte) (archsimd.Float32x8, archsimd.Float32x8, archsimd.Float32x8, archsimd.Float32x8) {
	block := (*[mxfp4BlockBytes]byte)(encoded)
	levels := archsimd.LoadFloat32x8(scaledFP4Table[block[0]][:8])
	sign := archsimd.BroadcastUint32x8(8)
	packed := archsimd.LoadUint8x16(block[1:])
	a := packed.ExtendLo8ToUint32()
	b := packed.ConcatShiftBytesRight(packed, 8).ExtendLo8ToUint32()
	hiA, hiB := a.ShiftAllRight(4), b.ShiftAllRight(4)
	// The shared 16 KiB table folds in E8M0 scaling before register permutation.
	return levels.Permute(a).ToBits().Xor(a.And(sign).ShiftAllLeft(28)).AsFloat32x8(),
		levels.Permute(b).ToBits().Xor(b.And(sign).ShiftAllLeft(28)).AsFloat32x8(),
		levels.Permute(hiA).ToBits().Xor(hiA.And(sign).ShiftAllLeft(28)).AsFloat32x8(),
		levels.Permute(hiB).ToBits().Xor(hiB.And(sign).ShiftAllLeft(28)).AsFloat32x8()
}

func decodeMXFP4Fast(row []byte, weights []float32) bool {
	if !archsimd.X86.AVX2() || !archsimd.X86.FMA() || simd.Emulated() {
		return false
	}
	for block := 0; block < len(row)/mxfp4BlockBytes; block++ {
		encoded := (*[mxfp4BlockBytes]byte)(row[block*mxfp4BlockBytes:])
		w0, w1, w2, w3 := mxfp4BlockAVX2(encoded[:])
		out := (*[mxfp4Elements]float32)(weights[block*mxfp4Elements:])
		w0.Store(out[:8])
		w1.Store(out[8:16])
		w2.Store(out[16:24])
		w3.Store(out[24:])
	}
	return true
}

func computeMoEBatchPairFast(ctx context.Context, data []byte, input int,
	x []float32, xOffsets []int, bias, y []float32, yOffsets []int, begin, end int,
) bool {
	if len(xOffsets) < 2 || len(xOffsets) > 4 || !archsimd.X86.AVX2() || !archsimd.X86.FMA() || simd.Emulated() {
		return false
	}
	rowBytes := input / mxfp4Elements * mxfp4BlockBytes
	for row := begin; row < end; row++ {
		if ctx.Err() != nil {
			return true
		}
		encoded := data[row*rowBytes : (row+1)*rowBytes]
		token := 0
		for ; token+1 < len(xOffsets); token += 2 {
			x0, x1 := x[xOffsets[token]:xOffsets[token]+input], x[xOffsets[token+1]:xOffsets[token+1]+input]
			a, b := dotMXFP4PairAVX2(encoded, x0, x1)
			y[yOffsets[token]+row], y[yOffsets[token+1]+row] = a+bias[row], b+bias[row]
		}
		if token < len(xOffsets) {
			value, _ := dotMXFP4Fast(encoded, x[xOffsets[token]:xOffsets[token]+input])
			y[yOffsets[token]+row] = value + bias[row]
		}
	}
	return true
}

// Two routed positions share decoded register weights, without a float row buffer.
func dotMXFP4PairAVX2(row []byte, x, y []float32) (float32, float32) {
	var a0, a1, a2, a3, b0, b1, b2, b3 archsimd.Float32x8
	for block := 0; block < len(row)/mxfp4BlockBytes; block++ {
		encoded := (*[mxfp4BlockBytes]byte)(row[block*mxfp4BlockBytes:])
		w0, w1, w2, w3 := mxfp4BlockAVX2(encoded[:])
		xv, yv := (*[32]float32)(x[block*32:]), (*[32]float32)(y[block*32:])
		a0 = w0.MulAdd(archsimd.LoadFloat32x8(xv[:8]), a0)
		b0 = w0.MulAdd(archsimd.LoadFloat32x8(yv[:8]), b0)
		a1 = w1.MulAdd(archsimd.LoadFloat32x8(xv[8:16]), a1)
		b1 = w1.MulAdd(archsimd.LoadFloat32x8(yv[8:16]), b1)
		a2 = w2.MulAdd(archsimd.LoadFloat32x8(xv[16:24]), a2)
		b2 = w2.MulAdd(archsimd.LoadFloat32x8(yv[16:24]), b2)
		a3 = w3.MulAdd(archsimd.LoadFloat32x8(xv[24:]), a3)
		b3 = w3.MulAdd(archsimd.LoadFloat32x8(yv[24:]), b3)
	}
	return quantSumAVX2(a0.Add(a1).Add(a2).Add(a3)), quantSumAVX2(b0.Add(b1).Add(b2).Add(b3))
}
