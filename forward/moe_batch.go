//go:build goexperiment.simd

package forward

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"simd"
	"unsafe"

	"Stream-PT/ggufindex"
	ggufmmap "Stream-PT/ggufmap"
)

type moeBatchScratch struct {
	batch, hidden, expertDim, experts, topK int
	routerBias, logits                      []float32
	selected                                []ExpertSelection
	heads, next                             []int
	results                                 []float32
	gate, up, act                           []float32
	gateBias, upBias, downBias              []float32
}

func moeBatchProduct(values ...int) (int, error) {
	n := 1
	for _, v := range values {
		if v <= 0 || n > int(^uint(0)>>1)/v {
			return 0, fmt.Errorf("invalid MoE dimension or overflow")
		}
		n *= v
	}
	return n, nil
}

func newMoEBatchScratch(cfg *Config, batch int) (*moeBatchScratch, error) {
	if cfg == nil || cfg.NumExpertsUsed <= 0 || cfg.NumExpertsUsed > cfg.NumExperts {
		return nil, fmt.Errorf("invalid MoE configuration")
	}
	selections, err := moeBatchProduct(batch, cfg.NumExpertsUsed)
	if err != nil {
		return nil, err
	}
	counts := [][]int{
		{batch, cfg.HiddenDim}, {batch, cfg.NumExperts}, {batch, cfg.NumExpertsUsed},
		{batch, cfg.NumExpertsUsed, cfg.HiddenDim}, {min(selections, moeBatchTile), cfg.ExpertHiddenDim},
	}
	sizes := make([]int, len(counts))
	for i, factors := range counts {
		n, err := moeBatchProduct(append(factors, 4)...)
		if err != nil {
			return nil, err
		}
		sizes[i] = n / 4
	}
	if _, err := moeBatchProduct(batch, cfg.NumExpertsUsed, int(unsafe.Sizeof(ExpertSelection{}))); err != nil {
		return nil, err
	}
	if _, err := moeBatchProduct(max(cfg.NumExperts, batch*cfg.NumExpertsUsed), int(unsafe.Sizeof(int(0)))); err != nil {
		return nil, err
	}
	return &moeBatchScratch{
		batch: batch, hidden: cfg.HiddenDim, expertDim: cfg.ExpertHiddenDim, experts: cfg.NumExperts, topK: cfg.NumExpertsUsed,
		routerBias: make([]float32, cfg.NumExperts), logits: make([]float32, sizes[1]),
		selected: make([]ExpertSelection, sizes[2]), heads: make([]int, cfg.NumExperts), next: make([]int, sizes[2]),
		results: make([]float32, sizes[3]), gate: make([]float32, sizes[4]), up: make([]float32, sizes[4]), act: make([]float32, sizes[4]),
		gateBias: make([]float32, cfg.ExpertHiddenDim), upBias: make([]float32, cfg.ExpertHiddenDim), downBias: make([]float32, cfg.HiddenDim),
	}, nil
}

// ForwardMoEBatch computes MoE for flattened [batch, HiddenDim] activations.
// Input and output must not overlap. Expert activation storage is tile bounded.
func ForwardMoEBatch(ctx context.Context, reader *ggufmmap.Reader, x []float32, cfg *Config,
	lw *LayerWeights, out []float32, batch int, options Q40Options,
) error {
	return forwardMoEBatch(ctx, reader, x, cfg, lw, out, batch, options, nil)
}

func forwardMoEBatch(ctx context.Context, reader *ggufmmap.Reader, x []float32, cfg *Config,
	lw *LayerWeights, out []float32, batch int, options Q40Options, scratch *moeBatchScratch,
) error {
	if ctx == nil || reader == nil || cfg == nil || lw == nil {
		return fmt.Errorf("nil MoE context, reader, configuration or weights")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	h, d, experts, k := cfg.HiddenDim, cfg.ExpertHiddenDim, cfg.NumExperts, cfg.NumExpertsUsed
	n, err := moeBatchProduct(batch, h, 4)
	if err != nil || len(x) != n/4 || len(out) != n/4 || k <= 0 || k > experts || d <= 0 {
		return fmt.Errorf("invalid MoE batch dimensions")
	}
	xStart, yStart := uintptr(unsafe.Pointer(&x[0])), uintptr(unsafe.Pointer(&out[0]))
	if (xStart <= yStart && yStart-xStart < uintptr(n)) || (yStart < xStart && xStart-yStart < uintptr(n)) {
		return fmt.Errorf("MoE input and output overlap")
	}
	router := lw.FFNGateInp
	routerBytes, err := moeBatchProduct(h, experts, 4)
	if err != nil || router.Type != 0 || len(router.Shape) != 2 || router.Shape[0] != uint64(h) ||
		router.Shape[1] != uint64(experts) || router.Range.End < router.Range.Start || router.Range.End-router.Range.Start != uint64(routerBytes) {
		return fmt.Errorf("%q: invalid F32 router shape or range", router.Name)
	}
	for _, projection := range []struct {
		t       ggufindex.Tensor
		in, out int
	}{
		{lw.FFNGateExps, h, d}, {lw.FFNUpExps, h, d}, {lw.FFNDownExps, d, h},
	} {
		if err := validateMoEBatchExpert(projection.t, projection.in, projection.out, experts); err != nil {
			return err
		}
	}
	for _, bias := range []struct {
		t          ggufindex.Tensor
		dim, count int
	}{
		{lw.FFNGateInpBias, experts, 1}, {lw.FFNGateExpsBias, d, experts},
		{lw.FFNUpExpsBias, d, experts}, {lw.FFNDownExpsBias, h, experts},
	} {
		if err := validateMoEBatchBias(bias.t, bias.dim, bias.count); err != nil {
			return err
		}
	}
	if scratch == nil {
		scratch, err = newMoEBatchScratch(cfg, batch)
		if err != nil {
			return err
		}
	} else if scratch.batch != batch || scratch.hidden != h || scratch.expertDim != d || scratch.experts != experts || scratch.topK != k {
		return fmt.Errorf("MoE batch scratch dimensions mismatch")
	}
	clear(scratch.routerBias)
	if lw.FFNGateInpBias.Name != "" {
		if err := readFloatVector(reader, lw.FFNGateInpBias, scratch.routerBias); err != nil {
			return err
		}
	}
	if err := reader.WithTensor(router, func(data []byte) error {
		return routeMoEBatch(ctx, data, x, scratch)
	}); err != nil {
		return fmt.Errorf("router: %w", err)
	}
	for expert := range scratch.heads {
		scratch.heads[expert] = -1
	}
	for token := 0; token < batch; token++ {
		if err := ctx.Err(); err != nil {
			return err
		}
		selectMoEBatch(scratch.logits[token*experts:(token+1)*experts], scratch.selected[token*k:(token+1)*k])
		for rank := 0; rank < k; rank++ {
			idx := token*k + rank
			sel := scratch.selected[idx]
			if sel.Weight <= 0 {
				continue
			}
			scratch.next[idx] = scratch.heads[sel.Index]
			scratch.heads[sel.Index] = idx
		}
	}
	for expert, head := range scratch.heads {
		if head < 0 {
			continue
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		if err := readExpertBias(reader, lw.FFNGateExpsBias, expert, experts, scratch.gateBias); err != nil {
			return err
		}
		if err := readExpertBias(reader, lw.FFNUpExpsBias, expert, experts, scratch.upBias); err != nil {
			return err
		}
		if err := readExpertBias(reader, lw.FFNDownExpsBias, expert, experts, scratch.downBias); err != nil {
			return err
		}
		err := withMoEBatchExpert(reader, lw.FFNGateExps, expert, func(gate []byte) error {
			return withMoEBatchExpert(reader, lw.FFNUpExps, expert, func(up []byte) error {
				return withMoEBatchExpert(reader, lw.FFNDownExps, expert, func(down []byte) error {
					return evaluateMoEBatchExpert(ctx, gate, up, down, x, head, scratch, options)
				})
			})
		})
		if err != nil {
			return fmt.Errorf("expert %d: %w", expert, err)
		}
	}
	clear(out)
	var dummy simd.Float32s
	lanes := dummy.Len()
	for token := 0; token < batch; token++ {
		if err := ctx.Err(); err != nil {
			return err
		}
		for rank := 0; rank < k; rank++ {
			idx := token*k + rank
			weight := scratch.selected[idx].Weight
			if weight <= 0 {
				continue
			}
			w := simd.BroadcastFloat32s(weight)
			for offset := 0; offset < h; offset += lanes {
				end := min(offset+lanes, h)
				if end-offset == lanes {
					d := simd.LoadFloat32s(scratch.results[idx*h+offset : idx*h+end])
					o := simd.LoadFloat32s(out[token*h+offset : token*h+end])
					result := d.MulAdd(w, o)
					result.Store(out[token*h+offset : token*h+end])
				} else {
					d, _ := simd.LoadFloat32sPart(scratch.results[idx*h+offset : idx*h+end])
					o, _ := simd.LoadFloat32sPart(out[token*h+offset : token*h+end])
					result := d.MulAdd(w, o)
					result.StorePart(out[token*h+offset : token*h+end])
				}
			}
		}
	}
	return ctx.Err()
}

func validateMoEBatchBias(t ggufindex.Tensor, dim, count int) error {
	if t.Name == "" {
		if t.Range.End != t.Range.Start {
			return fmt.Errorf("unnamed MoE bias has nonempty range")
		}
		return nil
	}
	width := 4
	if t.Type == 1 || t.Type == 30 {
		width = 2
	} else if t.Type != 0 {
		return fmt.Errorf("%q: unsupported bias type", t.Name)
	}
	n, err := moeBatchProduct(dim, count, width)
	wantDims := 2
	if count == 1 && len(t.Shape) == 1 {
		wantDims = 1
	}
	if err != nil || len(t.Shape) != wantDims || t.Shape[0] != uint64(dim) ||
		(wantDims == 2 && t.Shape[1] != uint64(count)) || t.Range.End < t.Range.Start || t.Range.End-t.Range.Start != uint64(n) {
		return fmt.Errorf("%q: invalid bias shape or range", t.Name)
	}
	return nil
}

func routeMoEBatch(ctx context.Context, data []byte, x []float32, s *moeBatchScratch) error {
	var dummy simd.Float32s
	lanes := dummy.Len()
	partials := make([]float32, lanes)
	acc := make([]float32, moeBatchTile*lanes)
	for expert := 0; expert < s.experts; expert++ {
		row := data[expert*s.hidden*4 : (expert+1)*s.hidden*4]
		for begin := 0; begin < s.batch; begin += moeBatchTile {
			if err := ctx.Err(); err != nil {
				return err
			}
			count := min(moeBatchTile, s.batch-begin)
			clear(acc)
			for offset := 0; offset < s.hidden; offset += lanes {
				end := min(offset+lanes, s.hidden)
				for i := offset; i < end; i++ {
					partials[i-offset] = math.Float32frombits(binary.LittleEndian.Uint32(row[i*4:]))
				}
				var w simd.Float32s
				if end-offset == lanes {
					w = simd.LoadFloat32s(partials)
				} else {
					w, _ = simd.LoadFloat32sPart(partials[:end-offset])
				}
				for token := 0; token < count; token++ {
					base := (begin + token) * s.hidden
					var v simd.Float32s
					if end-offset == lanes {
						v = simd.LoadFloat32s(x[base+offset : base+end])
					} else {
						v, _ = simd.LoadFloat32sPart(x[base+offset : base+end])
					}
					storage := acc[token*lanes : (token+1)*lanes]
					a := simd.LoadFloat32s(storage)
					result := w.MulAdd(v, a)
					result.Store(storage)
				}
			}
			for token := 0; token < count; token++ {
				copy(partials, acc[token*lanes:(token+1)*lanes])
				var sum float32
				for _, p := range partials {
					sum += p
				}
				s.logits[(begin+token)*s.experts+expert] = sum + s.routerBias[expert]
			}
		}
	}
	return ctx.Err()
}

func selectMoEBatch(logits []float32, selected []ExpertSelection) {
	for k := range selected {
		best, value := -1, float32(math.Inf(-1))
		for expert, logit := range logits {
			already := false
			for prev := 0; prev < k; prev++ {
				if selected[prev].Index == expert {
					already = true
					break
				}
			}
			if !already && logit > value {
				best, value = expert, logit
			}
		}
		if best < 0 {
			best, value = k, 0
		}
		selected[k] = ExpertSelection{Index: best, Weight: value}
	}
	maxLogit := float32(math.Inf(-1))
	for _, sel := range selected {
		if sel.Weight > maxLogit {
			maxLogit = sel.Weight
		}
	}
	var sum float32
	for i := range selected {
		selected[i].Weight = float32(math.Exp(float64(selected[i].Weight - maxLogit)))
		sum += selected[i].Weight
	}
	if sum > 0 {
		inv := float32(1) / sum
		for i := range selected {
			selected[i].Weight *= inv
		}
	}
}

func evaluateMoEBatchExpert(ctx context.Context, gate, up, down []byte, x []float32,
	head int, s *moeBatchScratch, options Q40Options,
) error {
	var inputs, activations, outputs [moeBatchTile]int
	for head >= 0 {
		count := 0
		for head >= 0 && count < moeBatchTile {
			inputs[count] = head / s.topK * s.hidden
			activations[count] = count * s.expertDim
			outputs[count] = head * s.hidden
			head = s.next[head]
			count++
		}
		if err := mulMoEBatchExpert(ctx, gate, s.hidden, s.expertDim, x, inputs[:count], s.gateBias, s.gate, activations[:count], options); err != nil {
			return err
		}
		if err := mulMoEBatchExpert(ctx, up, s.hidden, s.expertDim, x, inputs[:count], s.upBias, s.up, activations[:count], options); err != nil {
			return err
		}
		for i := 0; i < count*s.expertDim; i++ {
			s.act[i] = gptOSSSwiGLU(s.gate[i], s.up[i])
		}
		if err := mulMoEBatchExpert(ctx, down, s.expertDim, s.hidden, s.act, activations[:count], s.downBias, s.results, outputs[:count], options); err != nil {
			return err
		}
	}
	return ctx.Err()
}
