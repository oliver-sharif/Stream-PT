//go:build goexperiment.simd && amd64

package forward

import (
	"context"
	"simd"
	"simd/archsimd"
)

func routeMoEBatchFast(ctx context.Context, data []byte, x []float32, s *moeBatchScratch) (bool, error) {
	if s.hidden%32 != 0 || !archsimd.X86.AVX2() || !archsimd.X86.FMA() || simd.Emulated() {
		return false, nil
	}
	for expert := 0; expert < s.experts; expert++ {
		row := data[expert*s.hidden*4 : (expert+1)*s.hidden*4]
		for token := 0; token < s.batch; token++ {
			if err := ctx.Err(); err != nil {
				return true, err
			}
			s.logits[token*s.experts+expert] = dotF32AVX2(row, x[token*s.hidden:(token+1)*s.hidden]) + s.routerBias[expert]
		}
	}
	return true, ctx.Err()
}

// Byte loads permit unaligned, read-only F32 mmap rows without scalar decoding.
func dotF32AVX2(row []byte, x []float32) float32 {
	var a0, a1, a2, a3, a4, a5, a6, a7 archsimd.Float32x8
	offset := 0
	for ; offset+64 <= len(x); offset += 64 {
		w0 := (*[128]byte)(row[offset*4:])
		w1 := (*[128]byte)(row[(offset+32)*4:])
		v0 := (*[32]float32)(x[offset:])
		v1 := (*[32]float32)(x[offset+32:])
		a0 = archsimd.LoadUint8x32(w0[:32]).AsFloat32x8().MulAdd(archsimd.LoadFloat32x8(v0[:8]), a0)
		a1 = archsimd.LoadUint8x32(w0[32:64]).AsFloat32x8().MulAdd(archsimd.LoadFloat32x8(v0[8:16]), a1)
		a2 = archsimd.LoadUint8x32(w0[64:96]).AsFloat32x8().MulAdd(archsimd.LoadFloat32x8(v0[16:24]), a2)
		a3 = archsimd.LoadUint8x32(w0[96:]).AsFloat32x8().MulAdd(archsimd.LoadFloat32x8(v0[24:]), a3)
		a4 = archsimd.LoadUint8x32(w1[:32]).AsFloat32x8().MulAdd(archsimd.LoadFloat32x8(v1[:8]), a4)
		a5 = archsimd.LoadUint8x32(w1[32:64]).AsFloat32x8().MulAdd(archsimd.LoadFloat32x8(v1[8:16]), a5)
		a6 = archsimd.LoadUint8x32(w1[64:96]).AsFloat32x8().MulAdd(archsimd.LoadFloat32x8(v1[16:24]), a6)
		a7 = archsimd.LoadUint8x32(w1[96:]).AsFloat32x8().MulAdd(archsimd.LoadFloat32x8(v1[24:]), a7)
	}
	for ; offset < len(x); offset += 32 {
		w := (*[128]byte)(row[offset*4:])
		v := (*[32]float32)(x[offset:])
		a0 = archsimd.LoadUint8x32(w[:32]).AsFloat32x8().MulAdd(archsimd.LoadFloat32x8(v[:8]), a0)
		a1 = archsimd.LoadUint8x32(w[32:64]).AsFloat32x8().MulAdd(archsimd.LoadFloat32x8(v[8:16]), a1)
		a2 = archsimd.LoadUint8x32(w[64:96]).AsFloat32x8().MulAdd(archsimd.LoadFloat32x8(v[16:24]), a2)
		a3 = archsimd.LoadUint8x32(w[96:]).AsFloat32x8().MulAdd(archsimd.LoadFloat32x8(v[24:]), a3)
	}
	sum0 := a0.Add(a1).Add(a2).Add(a3)
	sum1 := a4.Add(a5).Add(a6).Add(a7)
	return quantSumAVX2(sum0.Add(sum1))
}

func rotateRoPEFast(head, cosTable, sinTable []float32) int {
	if !archsimd.X86.AVX2() || simd.Emulated() {
		return 0
	}
	half := len(cosTable)
	i := 0
	for ; i+16 <= half; i += 16 {
		a0, b0 := archsimd.LoadFloat32x8(head[i:i+8]), archsimd.LoadFloat32x8(head[half+i:half+i+8])
		c0, s0 := archsimd.LoadFloat32x8(cosTable[i:i+8]), archsimd.LoadFloat32x8(sinTable[i:i+8])
		a1, b1 := archsimd.LoadFloat32x8(head[i+8:i+16]), archsimd.LoadFloat32x8(head[half+i+8:half+i+16])
		c1, s1 := archsimd.LoadFloat32x8(cosTable[i+8:i+16]), archsimd.LoadFloat32x8(sinTable[i+8:i+16])

		a0.Mul(c0).Sub(b0.Mul(s0)).Store(head[i : i+8])
		a0.MulAdd(s0, b0.Mul(c0)).Store(head[half+i : half+i+8])
		a1.Mul(c1).Sub(b1.Mul(s1)).Store(head[i+8 : i+16])
		a1.MulAdd(s1, b1.Mul(c1)).Store(head[half+i+8 : half+i+16])
	}
	for ; i+8 <= half; i += 8 {
		a, b := archsimd.LoadFloat32x8(head[i:i+8]), archsimd.LoadFloat32x8(head[half+i:half+i+8])
		c, s := archsimd.LoadFloat32x8(cosTable[i:i+8]), archsimd.LoadFloat32x8(sinTable[i:i+8])
		a.Mul(c).Sub(b.Mul(s)).Store(head[i : i+8])
		a.MulAdd(s, b.Mul(c)).Store(head[half+i : half+i+8])
	}
	return i
}
