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

// LayerWeights holds the tensor descriptors for a single transformer block.
type LayerWeights struct {
	AttnNorm        ggufindex.Tensor
	AttnQ           ggufindex.Tensor
	AttnQBias       ggufindex.Tensor
	AttnK           ggufindex.Tensor
	AttnKBias       ggufindex.Tensor
	AttnV           ggufindex.Tensor
	AttnVBias       ggufindex.Tensor
	AttnSinks       ggufindex.Tensor
	AttnOutput      ggufindex.Tensor
	AttnOutputBias  ggufindex.Tensor
	PostAttnNorm    ggufindex.Tensor
	FFNGateInp      ggufindex.Tensor
	FFNGateInpBias  ggufindex.Tensor
	FFNGateExps     ggufindex.Tensor
	FFNGateExpsBias ggufindex.Tensor
	FFNUpExps       ggufindex.Tensor
	FFNUpExpsBias   ggufindex.Tensor
	FFNDownExps     ggufindex.Tensor
	FFNDownExpsBias ggufindex.Tensor
}

// NewLayerWeights builds a fast lookup structure from a GGUF Layer.
func NewLayerWeights(layer ggufindex.Layer) LayerWeights {
	var lw LayerWeights
	for _, t := range layer.Tensors {
		switch {
		case hasSuffix(t.Name, "attn_norm.weight"):
			lw.AttnNorm = t
		case hasSuffix(t.Name, "attn_q.weight"):
			lw.AttnQ = t
		case hasSuffix(t.Name, "attn_q.bias"):
			lw.AttnQBias = t
		case hasSuffix(t.Name, "attn_k.weight"):
			lw.AttnK = t
		case hasSuffix(t.Name, "attn_k.bias"):
			lw.AttnKBias = t
		case hasSuffix(t.Name, "attn_v.weight"):
			lw.AttnV = t
		case hasSuffix(t.Name, "attn_v.bias"):
			lw.AttnVBias = t
		case hasSuffix(t.Name, "attn_sinks.weight"):
			lw.AttnSinks = t
		case hasSuffix(t.Name, "attn_output.weight"):
			lw.AttnOutput = t
		case hasSuffix(t.Name, "attn_output.bias"):
			lw.AttnOutputBias = t
		case hasSuffix(t.Name, "post_attention_norm.weight"):
			lw.PostAttnNorm = t
		case hasSuffix(t.Name, "ffn_gate_inp.weight"):
			lw.FFNGateInp = t
		case hasSuffix(t.Name, "ffn_gate_inp.bias"):
			lw.FFNGateInpBias = t
		case hasSuffix(t.Name, "ffn_gate_exps.weight"):
			lw.FFNGateExps = t
		case hasSuffix(t.Name, "ffn_gate_exps.bias"):
			lw.FFNGateExpsBias = t
		case hasSuffix(t.Name, "ffn_up_exps.weight"):
			lw.FFNUpExps = t
		case hasSuffix(t.Name, "ffn_up_exps.bias"):
			lw.FFNUpExpsBias = t
		case hasSuffix(t.Name, "ffn_down_exps.weight"):
			lw.FFNDownExps = t
		case hasSuffix(t.Name, "ffn_down_exps.bias"):
			lw.FFNDownExpsBias = t
		}
	}
	return lw
}

func hasSuffix(s, suffix string) bool {
	return len(s) >= len(suffix) && s[len(s)-len(suffix):] == suffix
}

// ForwardLayer executes a single transformer block in-place on x.
func ForwardLayer(
	ctx context.Context,
	reader *ggufmmap.Reader,
	layerIdx int,
	lw *LayerWeights,
	x []float32,
	cache *KVCache,
	pos int,
	cfg *Config,
	scratch *LayerScratch,
	options Q40Options,
) error {
	if ctx == nil {
		return fmt.Errorf("nil context")
	}
	if err := ctx.Err(); err != nil {
		return err
	}

	// 1. Attention Norm: RMSNorm(x) -> scratch.NormedX
	if err := RMSNormInto(ctx, reader, lw.AttnNorm, x, scratch.NormedX, cfg.RMSNormEps); err != nil {
		return fmt.Errorf("layer %d attn norm: %w", layerIdx, err)
	}

	// 2. Q, K, V Projections
	if err := MulQ40Into(ctx, reader, lw.AttnQ, scratch.NormedX, scratch.Q, options); err != nil {
		return fmt.Errorf("layer %d attn Q: %w", layerIdx, err)
	}
	if err := addBias(reader, lw.AttnQBias, scratch.Q); err != nil {
		return err
	}

	if err := MulQ40Into(ctx, reader, lw.AttnK, scratch.NormedX, scratch.K, options); err != nil {
		return fmt.Errorf("layer %d attn K: %w", layerIdx, err)
	}
	if err := addBias(reader, lw.AttnKBias, scratch.K); err != nil {
		return err
	}

	if err := MulQ40Into(ctx, reader, lw.AttnV, scratch.NormedX, scratch.V, options); err != nil {
		return fmt.Errorf("layer %d attn V: %w", layerIdx, err)
	}
	if err := addBias(reader, lw.AttnVBias, scratch.V); err != nil {
		return err
	}

	// 3. RoPE
	ApplyRoPE(scratch.Q, scratch.K, pos, cfg.NumHeads, cfg.NumKVHeads, cfg.HeadDim, cfg.RopeFreqBase,
		RoPEOptions{ScalingFactor: cfg.RopeScalingFactor, OriginalContext: cfg.RopeOriginalContext})

	// 4. Attention
	if err := readFloatVector(reader, lw.AttnSinks, scratch.Sinks); err != nil {
		return fmt.Errorf("attention sinks: %w", err)
	}
	window := 0
	if layerIdx%2 == 0 {
		window = cfg.SlidingWindow
	}
	var attentionVector simd.Float32s
	scratch.attention.prepare(cache.MaxPos, attentionVector.Len())
	forwardAttention(scratch.Q, scratch.K, scratch.V, cache, layerIdx, pos,
		cfg.NumHeads, cfg.NumKVHeads, cfg.HeadDim, scratch.AttnCtx,
		AttentionOptions{Sinks: scratch.Sinks, SlidingWindow: window}, scratch.attention.scores, scratch.attention.partials)

	// 5. Attention Output Projection
	if err := MulQ40Into(ctx, reader, lw.AttnOutput, scratch.AttnCtx, scratch.AttnProj, options); err != nil {
		return fmt.Errorf("layer %d attn output: %w", layerIdx, err)
	}
	if err := addBias(reader, lw.AttnOutputBias, scratch.AttnProj); err != nil {
		return err
	}

	// Residual connection: x += AttnProj
	addVector(x, scratch.AttnProj)

	// 6. Post Attention Norm: RMSNorm(x) -> scratch.NormedX
	if err := RMSNormInto(ctx, reader, lw.PostAttnNorm, x, scratch.NormedX, cfg.RMSNormEps); err != nil {
		return fmt.Errorf("layer %d post attn norm: %w", layerIdx, err)
	}

	// 7. MoE FFN
	if cfg.NumExpertsUsed <= 0 || cfg.NumExpertsUsed > cfg.NumExperts {
		return fmt.Errorf("layer %d MoE: invalid expert count %d or top-k %d", layerIdx, cfg.NumExperts, cfg.NumExpertsUsed)
	}
	if scratch.moe == nil || len(scratch.moe.gateOut) != cfg.ExpertHiddenDim ||
		len(scratch.moe.downOut) != cfg.HiddenDim || len(scratch.moe.routerLogits) != cfg.NumExperts ||
		len(scratch.moe.selected) != cfg.NumExpertsUsed {
		scratch.moe = newMoEScratch(cfg)
	}
	if err := forwardMoE(ctx, reader, scratch.NormedX, cfg,
		lw.FFNGateInp, lw.FFNGateInpBias,
		lw.FFNGateExps, lw.FFNGateExpsBias,
		lw.FFNUpExps, lw.FFNUpExpsBias,
		lw.FFNDownExps, lw.FFNDownExpsBias,
		scratch.FFNOut,
		options,
		scratch.moe,
	); err != nil {
		return fmt.Errorf("layer %d MoE: %w", layerIdx, err)
	}

	// Residual connection: x += FFNOut
	addVector(x, scratch.FFNOut)

	return nil
}

// LayerScratch holds reusable temporary buffers to avoid allocations per layer.
type LayerScratch struct {
	NormedX   []float32
	Q         []float32
	K         []float32
	V         []float32
	Sinks     []float32
	AttnCtx   []float32
	AttnProj  []float32
	FFNOut    []float32
	moe       *moeScratch
	attention AttentionScratch
}

// NewLayerScratch allocates scratch buffers for a forward step.
func NewLayerScratch(cfg *Config) *LayerScratch {
	return &LayerScratch{
		NormedX:  make([]float32, cfg.HiddenDim),
		Q:        make([]float32, cfg.NumHeads*cfg.HeadDim),
		K:        make([]float32, cfg.NumKVHeads*cfg.HeadDim),
		V:        make([]float32, cfg.NumKVHeads*cfg.HeadDim),
		Sinks:    make([]float32, cfg.NumHeads),
		AttnCtx:  make([]float32, cfg.NumHeads*cfg.HeadDim),
		AttnProj: make([]float32, cfg.HiddenDim),
		FFNOut:   make([]float32, cfg.HiddenDim),
	}
}

func addBias(reader *ggufmmap.Reader, biasTensor ggufindex.Tensor, dst []float32) error {
	if biasTensor.Range.End <= biasTensor.Range.Start {
		if biasTensor.Name == "" {
			return nil
		}
		return fmt.Errorf("%q: invalid bias range", biasTensor.Name)
	}
	bias := make([]float32, len(dst))
	if err := readFloatVector(reader, biasTensor, bias); err != nil {
		return err
	}
	addVector(dst, bias)
	return nil
}

func readFloatVector(reader *ggufmmap.Reader, tensor ggufindex.Tensor, dst []float32) error {
	width := 4
	if tensor.Type == 1 || tensor.Type == 30 {
		width = 2
	} else if tensor.Type != 0 {
		return fmt.Errorf("%q: unsupported float tensor type %d", tensor.Name, tensor.Type)
	}
	if len(tensor.Shape) != 1 || tensor.Shape[0] != uint64(len(dst)) ||
		tensor.Range.End < tensor.Range.Start || tensor.Range.End-tensor.Range.Start != uint64(width*len(dst)) {
		return fmt.Errorf("%q: invalid float vector shape or range", tensor.Name)
	}
	return reader.WithTensor(tensor, func(data []byte) error {
		for i := range dst {
			switch tensor.Type {
			case 0:
				dst[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[i*4:]))
			case 1:
				dst[i] = float16(binary.LittleEndian.Uint16(data[i*2:]))
			case 30:
				dst[i] = math.Float32frombits(uint32(binary.LittleEndian.Uint16(data[i*2:])) << 16)
			}
		}
		return nil
	})
}

func addVector(dst, src []float32) {
	var vec simd.Float32s
	lanes := vec.Len()
	n := min(len(dst), len(src))
	for offset := 0; offset < n; offset += lanes {
		end := min(offset+lanes, n)
		if end-offset == lanes {
			d := simd.LoadFloat32s(dst[offset:end])
			s := simd.LoadFloat32s(src[offset:end])
			res := d.Add(s)
			res.Store(dst[offset:end])
		} else {
			d, _ := simd.LoadFloat32sPart(dst[offset:end])
			s, _ := simd.LoadFloat32sPart(src[offset:end])
			res := d.Add(s)
			res.StorePart(dst[offset:end])
		}
	}
}
