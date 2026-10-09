//go:build goexperiment.simd

package forward

import (
	"bytes"
	"context"
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"Stream-PT/ggufindex"
	ggufmmap "Stream-PT/ggufmap"
)

func TestGPTOSSModelParameters(t *testing.T) {
	if _, err := os.Stat(filepath.Join("..", "model", "gpt-oss-120b-Q4_0-00001-of-00002.gguf")); os.IsNotExist(err) {
		t.Skip("Model files not present")
	}
	model, err := ggufindex.Open([]string{
		filepath.Join("..", "model", "gpt-oss-120b-Q4_0-00001-of-00002.gguf"),
		filepath.Join("..", "model", "gpt-oss-120b-Q4_0-00002-of-00002.gguf"),
	})
	if err != nil {
		t.Fatal(err)
	}
	for key, value := range model.Metadata {
		if strings.HasPrefix(key, "gpt-oss.") {
			var buf bytes.Buffer
			if err := value.WriteJSON(&buf); err != nil {
				t.Fatal(err)
			}
			t.Logf("%s = %s", key, buf.String())
		}
	}
	for _, tensor := range model.Layers[0].Tensors {
		t.Logf("%s: type=%d shape=%v", tensor.Name, tensor.Type, tensor.Shape)
	}
}

func TestGPTOSSExpertBiasesAndSwiGLU(t *testing.T) {
	const dim, experts = 32, 2
	path := filepath.Join(t.TempDir(), "experts.bin")
	var data []byte
	add := func(name string, typ uint32, shape []uint64, values []byte) ggufindex.Tensor {
		start := len(data)
		data = append(data, values...)
		return ggufindex.Tensor{Name: name, Type: typ, Shape: shape,
			Range: ggufindex.Range{File: path, Start: uint64(start), End: uint64(len(data))}}
	}
	f32 := func(values []float32) []byte {
		buf := make([]byte, len(values)*4)
		for i, v := range values {
			binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(v))
		}
		return buf
	}
	router := add("router", 0, []uint64{dim, experts}, f32(make([]float32, dim*experts)))
	routerBias := add("router.bias", 0, []uint64{experts}, f32([]float32{0, 1}))
	zeroWeights := make([]byte, experts*dim*mxfp4BlockBytes)
	for block := range experts * dim {
		zeroWeights[block*mxfp4BlockBytes] = 127
	}
	gate := add("gate", 39, []uint64{dim, dim, experts}, zeroWeights)
	up := add("up", 39, []uint64{dim, dim, experts}, zeroWeights)
	downWeights := append([]byte(nil), zeroWeights...)
	for row := range experts * dim {
		downWeights[row*mxfp4BlockBytes+1] = 2 // First coefficient is 1.
	}
	down := add("down", 39, []uint64{dim, dim, experts}, downWeights)
	bias := func(name string, value float32) ggufindex.Tensor {
		values := make([]float32, dim*experts)
		for i := dim; i < len(values); i++ {
			values[i] = value
		}
		return add(name, 0, []uint64{dim, experts}, f32(values))
	}
	gateBias, upBias, downBias := bias("gate.bias", 10), bias("up.bias", 12), bias("down.bias", 5)
	if err := os.WriteFile(path, data, 0600); err != nil {
		t.Fatal(err)
	}
	reader, err := ggufmmap.Open(&ggufindex.Model{Paths: []string{path}})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = reader.Close() })
	cfg := &Config{HiddenDim: dim, ExpertHiddenDim: dim, NumExperts: experts, NumExpertsUsed: 1}
	out := make([]float32, dim)
	err = moeForTest(context.Background(), reader, make([]float32, dim), cfg,
		router, routerBias, gate, gateBias, up, upBias, down, downBias, out, Q40Options{Workers: 2})
	if err != nil {
		t.Fatal(err)
	}
	want := float32(7/(1+math.Exp(-1.702*7))*8 + 5)
	for i, got := range out {
		if math.Abs(float64(got-want)) > 1e-5 {
			t.Fatalf("expert output[%d] = %g, want %g (expert biases and clamped GPT-OSS SwiGLU)", i, got, want)
		}
	}
}

func TestGPTOSSAttentionSink(t *testing.T) {
	cache := NewKVCache(1, 1, 1, 2)
	out := make([]float32, 2)
	attentionForTest([]float32{0, 0}, []float32{0, 0}, []float32{2, 4}, cache, 0, 0, 1, 1, 2, out,
		AttentionOptions{Sinks: []float32{0}})
	// A zero-logit sink takes half of the probability mass, with a zero value.
	compareVectors(t, out, []float32{1, 2})
}

func TestGPTOSSYaRNConcentration(t *testing.T) {
	q, k := []float32{1, 2, 3, 4}, []float32{1, 2, 3, 4}
	ApplyRoPE(q, k, 0, 1, 1, 4, 150000, RoPEOptions{ScalingFactor: 32, OriginalContext: 4096})
	concentration := float32(1 + 0.1*math.Log(32))
	want := []float32{concentration, 2 * concentration, 3 * concentration, 4 * concentration}
	compareVectors(t, q, want)
	compareVectors(t, k, want)
}
