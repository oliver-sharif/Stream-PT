//go:build goexperiment.simd

package forward

import (
	"context"
	"fmt"
	"math"
	"simd"

	"Stream-PT/ggufindex"
	ggufmmap "Stream-PT/ggufmap"
)

type prefillScratch struct {
	batch                                    int
	x, normed, q, k, v, attn, projected, ffn []float32
	sinks, vector                            []float32
	attention                                AttentionScratch
	moe, tailMoE                             *moeBatchScratch
	tailBatch                                int
}

func newPrefillScratch(cfg *Config, batch int) (*prefillScratch, error) {
	maxInt := int(^uint(0) >> 1)
	if batch <= 0 || cfg.HiddenDim <= 0 || cfg.NumHeads <= 0 || cfg.NumKVHeads <= 0 ||
		cfg.HeadDim <= 0 || cfg.NumHeads > maxInt/cfg.HeadDim || cfg.NumKVHeads > maxInt/cfg.HeadDim {
		return nil, fmt.Errorf("invalid prefill dimensions")
	}
	qDim, kvDim := cfg.NumHeads*cfg.HeadDim, cfg.NumKVHeads*cfg.HeadDim
	if batch > maxInt/4/max(cfg.HiddenDim, qDim, kvDim) {
		return nil, fmt.Errorf("prefill activation storage exceeds platform limits")
	}
	moe, err := newMoEBatchScratch(cfg, batch)
	if err != nil {
		return nil, err
	}
	return &prefillScratch{
		batch: batch, x: make([]float32, batch*cfg.HiddenDim), normed: make([]float32, batch*cfg.HiddenDim),
		q: make([]float32, batch*qDim), k: make([]float32, batch*kvDim), v: make([]float32, batch*kvDim),
		attn: make([]float32, batch*qDim), projected: make([]float32, batch*cfg.HiddenDim),
		ffn: make([]float32, batch*cfg.HiddenDim), sinks: make([]float32, cfg.NumHeads),
		vector: make([]float32, max(cfg.HiddenDim, qDim, kvDim)),
		moe:    moe,
	}, nil
}

func forwardLayerBatch(ctx context.Context, reader *ggufmmap.Reader, layer int, lw *LayerWeights,
	x []float32, cache *KVCache, startPos int, cfg *Config, scratch *prefillScratch, batch int, options Q40Options) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	dim, qDim, kvDim := cfg.HiddenDim, cfg.NumHeads*cfg.HeadDim, cfg.NumKVHeads*cfg.HeadDim
	normed := scratch.normed[:batch*dim]
	q, k, v := scratch.q[:batch*qDim], scratch.k[:batch*kvDim], scratch.v[:batch*kvDim]
	attn, projected, ffn := scratch.attn[:batch*qDim], scratch.projected[:batch*dim], scratch.ffn[:batch*dim]
	var vector simd.Float32s
	scratch.attention.prepare(cache.MaxPos, vector.Len())
	if err := normBatch(ctx, reader, lw.AttnNorm, x, normed, batch, cfg.RMSNormEps, scratch.vector, scratch.attention.partials); err != nil {
		return fmt.Errorf("attn norm: %w", err)
	}
	for _, projection := range []struct {
		weight, bias ggufindex.Tensor
		out          []float32
	}{
		{lw.AttnQ, lw.AttnQBias, q}, {lw.AttnK, lw.AttnKBias, k}, {lw.AttnV, lw.AttnVBias, v},
	} {
		if err := MulQ40BatchInto(ctx, reader, projection.weight, normed, projection.out, batch, options); err != nil {
			return err
		}
		if err := biasBatch(reader, projection.bias, projection.out, batch, scratch.vector); err != nil {
			return err
		}
	}
	if err := readFloatVector(reader, lw.AttnSinks, scratch.sinks); err != nil {
		return fmt.Errorf("attention sinks: %w", err)
	}
	window := 0
	if layer%2 == 0 {
		window = cfg.SlidingWindow
	}
	for item := 0; item < batch; item++ {
		if err := ctx.Err(); err != nil {
			return err
		}
		qItem, kItem, vItem := q[item*qDim:(item+1)*qDim], k[item*kvDim:(item+1)*kvDim], v[item*kvDim:(item+1)*kvDim]
		pos := startPos + item
		ApplyRoPE(qItem, kItem, pos, cfg.NumHeads, cfg.NumKVHeads, cfg.HeadDim, cfg.RopeFreqBase,
			RoPEOptions{ScalingFactor: cfg.RopeScalingFactor, OriginalContext: cfg.RopeOriginalContext})
		// Populate and attend in position order; no future key/value is visible.
		forwardAttention(qItem, kItem, vItem, cache, layer, pos, cfg.NumHeads, cfg.NumKVHeads, cfg.HeadDim,
			attn[item*qDim:(item+1)*qDim], AttentionOptions{Sinks: scratch.sinks, SlidingWindow: window},
			scratch.attention.scores, scratch.attention.partials)
	}
	if err := MulQ40BatchInto(ctx, reader, lw.AttnOutput, attn, projected, batch, options); err != nil {
		return err
	}
	if err := biasBatch(reader, lw.AttnOutputBias, projected, batch, scratch.vector); err != nil {
		return err
	}
	addVector(x, projected)
	if err := normBatch(ctx, reader, lw.PostAttnNorm, x, normed, batch, cfg.RMSNormEps, scratch.vector, scratch.attention.partials); err != nil {
		return fmt.Errorf("post attn norm: %w", err)
	}
	moe := scratch.moe
	if batch != scratch.batch {
		if scratch.tailMoE == nil || scratch.tailBatch != batch {
			var err error
			scratch.tailMoE, err = newMoEBatchScratch(cfg, batch)
			if err != nil {
				return err
			}
			scratch.tailBatch = batch
		}
		moe = scratch.tailMoE
	}
	if err := forwardMoEBatch(ctx, reader, normed, cfg, lw, ffn, batch, options, moe); err != nil {
		return fmt.Errorf("MoE: %w", err)
	}
	addVector(x, ffn)
	return nil
}

func biasBatch(reader *ggufmmap.Reader, tensor ggufindex.Tensor, dst []float32, batch int, buffer []float32) error {
	if tensor.Name == "" && tensor.Range.End <= tensor.Range.Start {
		return nil
	}
	dim := len(dst) / batch
	bias := buffer[:dim]
	if err := readFloatVector(reader, tensor, bias); err != nil {
		return err
	}
	for item := 0; item < batch; item++ {
		addVector(dst[item*dim:(item+1)*dim], bias)
	}
	return nil
}

func normBatch(ctx context.Context, reader *ggufmmap.Reader, weight ggufindex.Tensor, x, y []float32,
	batch int, epsilon float32, buffer, partials []float32) error {
	if epsilon < 0 || math.IsNaN(float64(epsilon)) || math.IsInf(float64(epsilon), 0) {
		return fmt.Errorf("invalid epsilon: %g", epsilon)
	}
	dim := len(x) / batch
	if weight.Type != 0 && weight.Type != 1 && weight.Type != 28 {
		return fmt.Errorf("%q: unsupported normalization weight type %d", weight.Name, weight.Type)
	}
	// readFloatVector uses the bias tensor's BF16 tag, while RMSNorm uses tag 28.
	if weight.Type == 28 {
		weight.Type = 30
	}
	weights := buffer[:dim]
	if err := readFloatVector(reader, weight, weights); err != nil {
		return err
	}
	var vector simd.Float32s
	lanes := vector.Len()
	for item := 0; item < batch; item++ {
		if err := ctx.Err(); err != nil {
			return err
		}
		input, output := x[item*dim:(item+1)*dim], y[item*dim:(item+1)*dim]
		var squares simd.Float32s
		for offset := 0; offset < dim; offset += lanes {
			end := min(offset+lanes, dim)
			var values simd.Float32s
			if end-offset == lanes {
				values = simd.LoadFloat32s(input[offset:end])
			} else {
				values, _ = simd.LoadFloat32sPart(input[offset:end])
			}
			squares = values.MulAdd(values, squares)
		}
		squares.Store(partials)
		var sum float32
		for _, value := range partials {
			sum += value
		}
		factor := simd.BroadcastFloat32s(float32(1) / float32(math.Sqrt(float64(sum/float32(dim)+epsilon))))
		for offset := 0; offset < dim; offset += lanes {
			end := min(offset+lanes, dim)
			var values, scales simd.Float32s
			if end-offset == lanes {
				values = simd.LoadFloat32s(input[offset:end])
				scales = simd.LoadFloat32s(weights[offset:end])
				values.Mul(scales).Mul(factor).Store(output[offset:end])
			} else {
				values, _ = simd.LoadFloat32sPart(input[offset:end])
				scales, _ = simd.LoadFloat32sPart(weights[offset:end])
				values.Mul(scales).Mul(factor).StorePart(output[offset:end])
			}
		}
	}
	return nil
}
