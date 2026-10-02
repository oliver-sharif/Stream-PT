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

// ExpertSelection stores selected expert index and its softmax weight.
type ExpertSelection struct {
	Index  int
	Weight float32
}

type moeScratch struct {
	routerBias, routerLogits []float32
	selected                 []ExpertSelection
	gateOut, upOut, act      []float32
	downOut                  []float32
	gateBias, upBias         []float32
	downBias                 []float32
}

func newMoEScratch(cfg *Config) *moeScratch {
	return &moeScratch{
		routerBias: make([]float32, cfg.NumExperts), routerLogits: make([]float32, cfg.NumExperts),
		selected: make([]ExpertSelection, cfg.NumExpertsUsed),
		gateOut:  make([]float32, cfg.ExpertHiddenDim), upOut: make([]float32, cfg.ExpertHiddenDim),
		act: make([]float32, cfg.ExpertHiddenDim), downOut: make([]float32, cfg.HiddenDim),
		gateBias: make([]float32, cfg.ExpertHiddenDim), upBias: make([]float32, cfg.ExpertHiddenDim),
		downBias: make([]float32, cfg.HiddenDim),
	}
}

// ForwardMoE routes the normalized activation vector to top-k experts and computes the FFN output.
func ForwardMoE(
	ctx context.Context,
	reader *ggufmmap.Reader,
	x []float32,
	cfg *Config,
	gateInpWeight, gateInpBias ggufindex.Tensor,
	gateExpsWeight, gateExpsBias ggufindex.Tensor,
	upExpsWeight, upExpsBias ggufindex.Tensor,
	downExpsWeight, downExpsBias ggufindex.Tensor,
	out []float32, // length = cfg.HiddenDim
	options Q40Options,
) error {
	return forwardMoE(ctx, reader, x, cfg, gateInpWeight, gateInpBias,
		gateExpsWeight, gateExpsBias, upExpsWeight, upExpsBias,
		downExpsWeight, downExpsBias, out, options, nil)
}

func forwardMoE(
	ctx context.Context,
	reader *ggufmmap.Reader,
	x []float32,
	cfg *Config,
	gateInpWeight, gateInpBias ggufindex.Tensor,
	gateExpsWeight, gateExpsBias ggufindex.Tensor,
	upExpsWeight, upExpsBias ggufindex.Tensor,
	downExpsWeight, downExpsBias ggufindex.Tensor,
	out []float32,
	options Q40Options,
	scratch *moeScratch,
) error {
	if ctx == nil {
		return fmt.Errorf("nil context")
	}
	if reader == nil {
		return fmt.Errorf("nil reader")
	}
	if err := ctx.Err(); err != nil {
		return err
	}

	numExperts := cfg.NumExperts
	topK := cfg.NumExpertsUsed
	if topK <= 0 || topK > numExperts {
		return fmt.Errorf("invalid expert count %d or top-k %d", numExperts, topK)
	}
	hiddenDim := cfg.HiddenDim
	expertDim := cfg.ExpertHiddenDim
	if len(x) != hiddenDim || len(out) != hiddenDim {
		return fmt.Errorf("MoE activation dimension mismatch")
	}
	if gateInpWeight.Type != 0 || len(gateInpWeight.Shape) != 2 ||
		gateInpWeight.Shape[0] != uint64(hiddenDim) || gateInpWeight.Shape[1] != uint64(numExperts) ||
		gateInpWeight.Range.End < gateInpWeight.Range.Start ||
		gateInpWeight.Range.End-gateInpWeight.Range.Start != uint64(hiddenDim*numExperts*4) {
		return fmt.Errorf("%q: expected F32 router matrix [%d %d]", gateInpWeight.Name, hiddenDim, numExperts)
	}
	if scratch == nil {
		scratch = newMoEScratch(cfg)
	}

	// 1. Read router bias if present.
	routerBias := scratch.routerBias
	clear(routerBias)
	if gateInpBias.Range.End > gateInpBias.Range.Start {
		if err := readFloatVector(reader, gateInpBias, routerBias); err != nil {
			return fmt.Errorf("router bias: %w", err)
		}
	}

	// 2. Compute router logits: gateInpWeight * x + routerBias.
	routerLogits := scratch.routerLogits
	if gateInpWeight.Range.End > gateInpWeight.Range.Start {
		err := reader.WithTensor(gateInpWeight, func(data []byte) error {
			rowBytes := hiddenDim * 4
			for e := 0; e < numExperts && (e+1)*rowBytes <= len(data); e++ {
				rowData := data[e*rowBytes : (e+1)*rowBytes]
				var acc simd.Float32s
				lanes := acc.Len()
				partials := make([]float32, lanes)
				for offset := 0; offset < hiddenDim; offset += lanes {
					end := min(offset+lanes, hiddenDim)
					var w simd.Float32s
					for l := offset; l < end; l++ {
						bits := binary.LittleEndian.Uint32(rowData[l*4 : (l+1)*4])
						partials[l-offset] = math.Float32frombits(bits)
					}
					var v simd.Float32s
					if end-offset == lanes {
						w = simd.LoadFloat32s(partials)
						v = simd.LoadFloat32s(x[offset:end])
					} else {
						w, _ = simd.LoadFloat32sPart(partials[:end-offset])
						v, _ = simd.LoadFloat32sPart(x[offset:end])
					}
					acc = w.MulAdd(v, acc)
				}
				acc.Store(partials)
				var sum float32
				for _, p := range partials {
					sum += p
				}
				routerLogits[e] = sum + routerBias[e]
			}
			return nil
		})
		if err != nil {
			return fmt.Errorf("router computation failed: %w", err)
		}
	}

	// 3. Find top-K experts.
	selected := scratch.selected
	for k := 0; k < topK; k++ {
		bestIdx := -1
		bestLogit := float32(math.Inf(-1))
		for e := 0; e < numExperts; e++ {
			already := false
			for prev := 0; prev < k; prev++ {
				if selected[prev].Index == e {
					already = true
					break
				}
			}
			if !already && routerLogits[e] > bestLogit {
				bestLogit = routerLogits[e]
				bestIdx = e
			}
		}
		if bestIdx < 0 {
			bestIdx = k
			bestLogit = 0
		}
		selected[k] = ExpertSelection{Index: bestIdx, Weight: bestLogit}
	}

	// 4. Softmax over top-K logits.
	maxLogit := float32(math.Inf(-1))
	for _, s := range selected {
		if s.Weight > maxLogit {
			maxLogit = s.Weight
		}
	}
	var sumExp float32
	for i := range selected {
		expVal := float32(math.Exp(float64(selected[i].Weight - maxLogit)))
		selected[i].Weight = expVal
		sumExp += expVal
	}
	if sumExp > 0 {
		invSum := float32(1.0) / sumExp
		for i := range selected {
			selected[i].Weight *= invSum
		}
	}

	// 5. Evaluate selected experts and accumulate into out.
	clear(out)
	gateOut, upOut, act := scratch.gateOut, scratch.upOut, scratch.act
	downOut := scratch.downOut
	gateBias, upBias, downBias := scratch.gateBias, scratch.upBias, scratch.downBias

	for _, sel := range selected {
		if err := ctx.Err(); err != nil {
			return err
		}
		if sel.Weight <= 0 {
			continue
		}
		if err := readExpertBias(reader, gateExpsBias, sel.Index, numExperts, gateBias); err != nil {
			return err
		}
		if err := readExpertBias(reader, upExpsBias, sel.Index, numExperts, upBias); err != nil {
			return err
		}
		if err := readExpertBias(reader, downExpsBias, sel.Index, numExperts, downBias); err != nil {
			return err
		}

		// Gate projection.
		if err := MulMXFP4Expert(ctx, reader, gateExpsWeight, sel.Index, x, gateBias, gateOut, options); err != nil {
			return fmt.Errorf("gate expert %d: %w", sel.Index, err)
		}

		// Up projection.
		if err := MulMXFP4Expert(ctx, reader, upExpsWeight, sel.Index, x, upBias, upOut, options); err != nil {
			return fmt.Errorf("up expert %d: %w", sel.Index, err)
		}

		// GPT-OSS uses clamped SwiGLU with alpha=1.702 and an up offset of 1.
		for i := 0; i < expertDim; i++ {
			act[i] = gptOSSSwiGLU(gateOut[i], upOut[i])
		}

		// Down projection.
		if err := MulMXFP4Expert(ctx, reader, downExpsWeight, sel.Index, act, downBias, downOut, options); err != nil {
			return fmt.Errorf("down expert %d: %w", sel.Index, err)
		}

		// Accumulate weighted output into out using SIMD.
		wVec := simd.BroadcastFloat32s(sel.Weight)
		var dummy simd.Float32s
		lanes := dummy.Len()
		for offset := 0; offset < hiddenDim; offset += lanes {
			end := min(offset+lanes, hiddenDim)
			if end-offset == lanes {
				dPart := simd.LoadFloat32s(downOut[offset:end])
				oPart := simd.LoadFloat32s(out[offset:end])
				res := dPart.MulAdd(wVec, oPart)
				res.Store(out[offset:end])
			} else {
				dPart, _ := simd.LoadFloat32sPart(downOut[offset:end])
				oPart, _ := simd.LoadFloat32sPart(out[offset:end])
				res := dPart.MulAdd(wVec, oPart)
				res.StorePart(out[offset:end])
			}
		}
	}

	return nil
}

func gptOSSSwiGLU(gate, up float32) float32 {
	gate = min(gate, 7)
	up = max(-7, min(up, 7))
	return gate / (1 + float32(math.Exp(float64(-1.702*gate)))) * (up + 1)
}

func readExpertBias(reader *ggufmmap.Reader, tensor ggufindex.Tensor, expert, numExperts int, dst []float32) error {
	if tensor.Name == "" {
		clear(dst)
		return nil
	}
	width := 4
	if tensor.Type == 1 || tensor.Type == 30 {
		width = 2
	}
	rowBytes := uint64(len(dst) * width)
	if len(tensor.Shape) != 2 || tensor.Shape[0] != uint64(len(dst)) || tensor.Shape[1] != uint64(numExperts) ||
		expert < 0 || expert >= numExperts || tensor.Range.End < tensor.Range.Start ||
		tensor.Range.End-tensor.Range.Start != rowBytes*uint64(numExperts) {
		return fmt.Errorf("%q: invalid expert bias shape or range", tensor.Name)
	}
	row := tensor
	row.Shape = []uint64{uint64(len(dst))}
	row.Range.Start += uint64(expert) * rowBytes
	row.Range.End = row.Range.Start + rowBytes
	return readFloatVector(reader, row, dst)
}
