//go:build goexperiment.simd

package forward

import (
	"math"
	"simd"
)

// KVCache holds the key and value states across token generation steps.
type KVCache struct {
	Keys   [][][]float32 // [layer][pos][kvDim]
	Values [][][]float32 // [layer][pos][kvDim]
	MaxPos int
}

// NewKVCache allocates KV cache storage for maxTokens.
func NewKVCache(numLayers, maxTokens, numKVHeads, headDim int) *KVCache {
	kvDim := numKVHeads * headDim
	keys := make([][][]float32, numLayers)
	vals := make([][][]float32, numLayers)
	for l := 0; l < numLayers; l++ {
		keys[l] = make([][]float32, maxTokens)
		vals[l] = make([][]float32, maxTokens)
		keyData := make([]float32, maxTokens*kvDim)
		valueData := make([]float32, maxTokens*kvDim)
		for p := 0; p < maxTokens; p++ {
			begin, end := p*kvDim, (p+1)*kvDim
			keys[l][p] = keyData[begin:end:end]
			vals[l][p] = valueData[begin:end:end]
		}
	}
	return &KVCache{
		Keys:   keys,
		Values: vals,
		MaxPos: maxTokens,
	}
}

type AttentionOptions struct {
	Sinks         []float32
	SlidingWindow int
}

// AttentionScratch reuses softmax scores and SIMD reduction storage.
type AttentionScratch struct {
	scores, partials []float32
}

func (s *AttentionScratch) prepare(positions, lanes int) {
	if len(s.scores) < positions {
		s.scores = make([]float32, positions)
	}
	if len(s.partials) != lanes {
		s.partials = make([]float32, lanes)
	}
}

func forwardAttention(
	q, k, v []float32,
	cache *KVCache,
	layer, pos int,
	numHeads, numKVHeads, headDim int,
	out []float32,
	options AttentionOptions,
	scores []float32,
	partials []float32,
) {
	kvDim := numKVHeads * headDim
	if pos < cache.MaxPos {
		copy(cache.Keys[layer][pos], k[:kvDim])
		copy(cache.Values[layer][pos], v[:kvDim])
	}

	headRatio := numHeads / numKVHeads
	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	sinks := options.Sinks
	startPos := 0
	if options.SlidingWindow > 0 {
		startPos = max(0, pos-options.SlidingWindow+1)
	}

	var vec simd.Float32s
	lanes := vec.Len()

	for h := 0; h < numHeads; h++ {
		kvHead := h / headRatio
		qHead := q[h*headDim : (h+1)*headDim]

		maxScore := float32(math.Inf(-1))
		if len(sinks) > 0 {
			maxScore = sinks[h]
		}
		for p := startPos; p <= pos && p < cache.MaxPos; p++ {
			kHead := cache.Keys[layer][p][kvHead*headDim : (kvHead+1)*headDim]

			var acc simd.Float32s
			for offset := 0; offset < headDim; offset += lanes {
				end := min(offset+lanes, headDim)
				var w, v simd.Float32s
				if end-offset == lanes {
					w = simd.LoadFloat32s(qHead[offset:end])
					v = simd.LoadFloat32s(kHead[offset:end])
				} else {
					w, _ = simd.LoadFloat32sPart(qHead[offset:end])
					v, _ = simd.LoadFloat32sPart(kHead[offset:end])
				}
				acc = w.MulAdd(v, acc)
			}
			acc.Store(partials)
			var dot float32
			for _, val := range partials {
				dot += val
			}
			score := dot * scale
			scores[p] = score
			if score > maxScore {
				maxScore = score
			}
		}

		// Softmax
		var sumExp float32
		if len(sinks) > 0 {
			sumExp = float32(math.Exp(float64(sinks[h] - maxScore)))
		}
		for p := startPos; p <= pos && p < cache.MaxPos; p++ {
			expVal := float32(math.Exp(float64(scores[p] - maxScore)))
			scores[p] = expVal
			sumExp += expVal
		}
		invSum := float32(1.0) / sumExp

		// Weighted sum of V
		outHead := out[h*headDim : (h+1)*headDim]
		var outAcc [32]simd.Float32s
		for p := startPos; p <= pos && p < cache.MaxPos; p++ {
			vHead := cache.Values[layer][p][kvHead*headDim : (kvHead+1)*headDim]
			sVec := simd.BroadcastFloat32s(scores[p] * invSum)
			for vIdx, offset := 0, 0; offset < headDim; vIdx, offset = vIdx+1, offset+lanes {
				end := min(offset+lanes, headDim)
				if end-offset == lanes {
					vPart := simd.LoadFloat32s(vHead[offset:end])
					outAcc[vIdx] = vPart.MulAdd(sVec, outAcc[vIdx])
				} else {
					vPart, _ := simd.LoadFloat32sPart(vHead[offset:end])
					outAcc[vIdx] = vPart.MulAdd(sVec, outAcc[vIdx])
				}
			}
		}
		for vIdx, offset := 0, 0; offset < headDim; vIdx, offset = vIdx+1, offset+lanes {
			end := min(offset+lanes, headDim)
			if end-offset == lanes {
				outAcc[vIdx].Store(outHead[offset:end])
			} else {
				outAcc[vIdx].StorePart(outHead[offset:end])
			}
		}
	}
}
