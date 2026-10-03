//go:build goexperiment.simd

package forward

import (
	"math"
)

type RoPEOptions struct {
	ScalingFactor   float32
	OriginalContext int
}

// ApplyRoPE applies Rotary Position Embeddings to query and key vectors in-place.
func ApplyRoPE(
	q, k []float32,
	pos int,
	numHeads, numKVHeads, headDim int,
	freqBase float32,
	options ...RoPEOptions,
) {
	if headDim%2 != 0 {
		return
	}
	halfDim := headDim / 2

	// Precompute cos and sin for this position.
	cosTable := make([]float32, halfDim)
	sinTable := make([]float32, halfDim)
	scaling, concentration := 1.0, 1.0
	var low, high float64
	if len(options) > 0 && options[0].ScalingFactor > 1 {
		scaling = float64(options[0].ScalingFactor)
		concentration = 1 + 0.1*math.Log(scaling)
		originalContext := float64(options[0].OriginalContext)
		low = float64(halfDim) * math.Log(originalContext/(32*2*math.Pi)) / math.Log(float64(freqBase))
		high = float64(halfDim) * math.Log(originalContext/(2*math.Pi)) / math.Log(float64(freqBase))
	}
	for i := 0; i < halfDim; i++ {
		theta := 1 / math.Pow(float64(freqBase), float64(2*i)/float64(headDim))
		if scaling > 1 {
			ramp := max(0.0, min(1.0, (float64(i)-low)/(high-low)))
			theta *= (1 - ramp) + ramp/scaling
		}
		angle := float64(pos) * theta
		cosTable[i] = float32(math.Cos(angle) * concentration)
		sinTable[i] = float32(math.Sin(angle) * concentration)
	}

	// Apply to Q heads.
	for h := 0; h < numHeads; h++ {
		head := q[h*headDim : (h+1)*headDim]
		for i := rotateRoPEFast(head, cosTable, sinTable); i < halfDim; i++ {
			x0 := head[i]
			x1 := head[i+halfDim]
			c := cosTable[i]
			s := sinTable[i]
			head[i] = x0*c - x1*s
			head[i+halfDim] = x0*s + x1*c
		}
	}

	// Apply to KV heads.
	for h := 0; h < numKVHeads; h++ {
		head := k[h*headDim : (h+1)*headDim]
		for i := rotateRoPEFast(head, cosTable, sinTable); i < halfDim; i++ {
			x0 := head[i]
			x1 := head[i+halfDim]
			c := cosTable[i]
			s := sinTable[i]
			head[i] = x0*c - x1*s
			head[i+halfDim] = x0*s + x1*c
		}
	}
}
