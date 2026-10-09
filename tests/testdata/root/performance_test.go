//go:build goexperiment.simd

package main

import (
	"testing"

	"Stream-PT/forward"
	"Stream-PT/ggufindex"
)

func TestCacheMemoryBudget(t *testing.T) {
	for _, tc := range []struct{ requested, available, want uint64 }{
		{2048, 16 << 30, 2048}, {2048, 2 << 30, 512}, {2048, 0, 0}, {0, 16 << 30, 0}, {2048, 1 << 20, 0},
	} {
		if got := cacheMiBForMemory(tc.requested, tc.available); got != tc.want {
			t.Fatalf("budget %+v: got %d", tc, got)
		}
	}
}

func TestResidentWeightCandidates(t *testing.T) {
	model := &ggufindex.Model{
		Shared: []ggufindex.Tensor{{Name: "token_embd.weight"}, {Name: "output_norm.weight"}, {Name: "output.weight"}},
		Layers: []ggufindex.Layer{{Tensors: []ggufindex.Tensor{{Name: "blk.0.attn_q.weight"}, {Name: "blk.0.ffn_gate_inp.weight"}, {Name: "blk.0.ffn_gate_exps.weight"}, {Name: "blk.0.ffn_up_exps.bias"}}}},
	}
	got := residentWeightCandidates(model)
	want := []string{"output.weight", "output_norm.weight", "blk.0.attn_q.weight", "blk.0.ffn_gate_inp.weight"}
	if len(got) != len(want) {
		t.Fatalf("got %v", got)
	}
	for i := range want {
		if got[i].Name != want[i] {
			t.Fatalf("priority %d got %s want %s", i, got[i].Name, want[i])
		}
	}
}

func TestInferencePerformance(t *testing.T) {
	metrics := forward.GenerationMetrics{PromptTokens: 32, ReusedPromptTokens: 20, DecodeSteps: 4, DecodeSeconds: 2}
	p := inferencePerformance(metrics, 3, processIO{readBytes: 10, majorFaults: 2, available: true}, processIO{readBytes: 30, majorFaults: 5, available: true})
	if p.ReadBytes != 20 || p.MajorFaults != 3 || !p.IOAvailable || p.FirstVisibleSeconds != 3 || p.ReusedPromptTokens != 20 {
		t.Fatalf("%+v", p)
	}
	if p := inferencePerformance(metrics, 0, processIO{}, processIO{available: true}); p.IOAvailable {
		t.Fatalf("unavailable %+v", p)
	}
	if p := inferencePerformance(metrics, 0, processIO{readBytes: 20, available: true}, processIO{readBytes: 10, available: true}); p.IOAvailable {
		t.Fatalf("underflow %+v", p)
	}
}
