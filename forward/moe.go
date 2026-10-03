//go:build goexperiment.simd

package forward

import (
	"context"
	"fmt"
	"math"

	"Stream-PT/ggufindex"
	ggufmmap "Stream-PT/ggufmap"
)

// ExpertSelection stores selected expert index and its softmax weight.
type ExpertSelection struct {
	Index  int
	Weight float32
}

type moeScratch struct {
	batch *moeBatchScratch
}

func newMoEScratch(cfg *Config) *moeScratch {
	return &moeScratch{}
}

// ForwardMoE routes the normalized activation vector to top-k experts and computes the FFN output.
func ForwardMoE(ctx context.Context, reader *ggufmmap.Reader, x []float32, cfg *Config,
	gateInpWeight, gateInpBias ggufindex.Tensor,
	gateExpsWeight, gateExpsBias ggufindex.Tensor,
	upExpsWeight, upExpsBias ggufindex.Tensor,
	downExpsWeight, downExpsBias ggufindex.Tensor,
	out []float32, options Q40Options,
) error {
	return forwardMoE(ctx, reader, x, cfg, gateInpWeight, gateInpBias,
		gateExpsWeight, gateExpsBias, upExpsWeight, upExpsBias,
		downExpsWeight, downExpsBias, out, options, nil)
}

func forwardMoE(ctx context.Context, reader *ggufmmap.Reader, x []float32, cfg *Config,
	gateInpWeight, gateInpBias ggufindex.Tensor,
	gateExpsWeight, gateExpsBias ggufindex.Tensor,
	upExpsWeight, upExpsBias ggufindex.Tensor,
	downExpsWeight, downExpsBias ggufindex.Tensor,
	out []float32, options Q40Options, scratch *moeScratch,
) error {
	if cfg == nil {
		return fmt.Errorf("nil MoE configuration")
	}
	var batchScratch *moeBatchScratch
	if scratch != nil {
		if scratch.batch == nil || scratch.batch.hidden != cfg.HiddenDim ||
			scratch.batch.expertDim != cfg.ExpertHiddenDim || scratch.batch.experts != cfg.NumExperts ||
			scratch.batch.topK != cfg.NumExpertsUsed {
			var err error
			scratch.batch, err = newMoEBatchScratch(cfg, 1)
			if err != nil {
				return err
			}
		}
		batchScratch = scratch.batch
	}
	lw := LayerWeights{
		FFNGateInp: gateInpWeight, FFNGateInpBias: gateInpBias,
		FFNGateExps: gateExpsWeight, FFNGateExpsBias: gateExpsBias,
		FFNUpExps: upExpsWeight, FFNUpExpsBias: upExpsBias,
		FFNDownExps: downExpsWeight, FFNDownExpsBias: downExpsBias,
	}
	// Decode callers use an empty name to disable an optional expert bias.
	for _, bias := range []*ggufindex.Tensor{&lw.FFNGateExpsBias, &lw.FFNUpExpsBias, &lw.FFNDownExpsBias} {
		if bias.Name == "" {
			*bias = ggufindex.Tensor{}
		}
	}
	return forwardMoEBatch(ctx, reader, x, cfg, &lw, out, 1, options, batchScratch)
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
