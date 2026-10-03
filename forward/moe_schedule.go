//go:build goexperiment.simd

package forward

import (
	"context"
	"fmt"

	"Stream-PT/ggufindex"
	ggufmmap "Stream-PT/ggufmap"
)

type moeProjectionJob struct {
	expert int
	up     bool
	both   bool
}

func selectedExpertRange(t ggufindex.Tensor, expert int) ggufindex.Range {
	size := t.Shape[0] / mxfp4Elements * mxfp4BlockBytes * t.Shape[1]
	start := t.Range.Start + uint64(expert)*size
	return ggufindex.Range{File: t.Range.File, Start: start, End: start + size}
}

func evaluateSelectedMoE(ctx context.Context, reader *ggufmmap.Reader, lw *LayerWeights,
	x []float32, s *moeBatchScratch, options Q40Options,
) error {
	// Gate and up are independent. Schedule both together by physical file offset;
	// only the down phase must wait for the complete SwiGLU activations.
	s.ranges, s.jobs = s.ranges[:0], s.jobs[:0]
	for expert, head := range s.heads {
		if head < 0 {
			continue
		}
		gateSpan, upSpan := selectedExpertRange(lw.FFNGateExps, expert), selectedExpertRange(lw.FFNUpExps, expert)
		if gateSpan == upSpan {
			s.ranges = append(s.ranges, gateSpan)
			s.jobs = append(s.jobs, moeProjectionJob{expert: expert, both: true})
		} else {
			s.ranges = append(s.ranges, gateSpan, upSpan)
			s.jobs = append(s.jobs, moeProjectionJob{expert: expert}, moeProjectionJob{expert: expert, up: true})
		}
	}
	if err := reader.WithExpertRanges(s.ranges, func(index int, data []byte) error {
		if err := ctx.Err(); err != nil {
			return err
		}
		job := s.jobs[index]
		biasTensor, bias, dst := lw.FFNGateExpsBias, s.gateBias, s.gate
		if job.up {
			biasTensor, bias, dst = lw.FFNUpExpsBias, s.upBias, s.up
		}
		if err := readExpertBias(reader, biasTensor, job.expert, s.experts, bias); err != nil {
			return err
		}
		if err := projectSelectedMoE(ctx, data, x, bias, dst, s.heads[job.expert], s, false, options); err != nil {
			return err
		}
		if job.both {
			if err := readExpertBias(reader, lw.FFNUpExpsBias, job.expert, s.experts, s.upBias); err != nil {
				return err
			}
			return projectSelectedMoE(ctx, data, x, s.upBias, s.up, s.heads[job.expert], s, false, options)
		}
		return nil
	}); err != nil {
		return fmt.Errorf("selected gate/up experts: %w", err)
	}
	for idx, sel := range s.selected {
		if err := ctx.Err(); err != nil {
			return err
		}
		if sel.Weight <= 0 {
			continue
		}
		base := idx * s.expertDim
		for i := base; i < base+s.expertDim; i++ {
			s.act[i] = gptOSSSwiGLU(s.gate[i], s.up[i])
		}
	}
	s.ranges, s.jobs = s.ranges[:0], s.jobs[:0]
	for expert, head := range s.heads {
		if head >= 0 {
			s.ranges = append(s.ranges, selectedExpertRange(lw.FFNDownExps, expert))
			s.jobs = append(s.jobs, moeProjectionJob{expert: expert})
		}
	}
	if err := reader.WithExpertRanges(s.ranges, func(index int, data []byte) error {
		if err := ctx.Err(); err != nil {
			return err
		}
		expert := s.jobs[index].expert
		if err := readExpertBias(reader, lw.FFNDownExpsBias, expert, s.experts, s.downBias); err != nil {
			return err
		}
		return projectSelectedMoE(ctx, data, s.act, s.downBias, s.results, s.heads[expert], s, true, options)
	}); err != nil {
		return fmt.Errorf("selected down experts: %w", err)
	}
	return ctx.Err()
}

func projectSelectedMoE(ctx context.Context, data []byte, x, bias, dst []float32,
	head int, s *moeBatchScratch, down bool, options Q40Options,
) error {
	var inputs, outputs [moeBatchTile]int
	input, output := s.hidden, s.expertDim
	if down {
		input, output = s.expertDim, s.hidden
	}
	for head >= 0 {
		count := 0
		for head >= 0 && count < moeBatchTile {
			if down {
				inputs[count], outputs[count] = head*s.expertDim, head*s.hidden
			} else {
				inputs[count], outputs[count] = head/s.topK*s.hidden, head*s.expertDim
			}
			head = s.next[head]
			count++
		}
		if err := mulMoEBatchExpert(ctx, data, input, output, x, inputs[:count], bias, dst, outputs[:count], options); err != nil {
			return err
		}
	}
	return ctx.Err()
}
