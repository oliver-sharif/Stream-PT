//go:build goexperiment.simd

package tests

import (
	"context"
	"encoding/binary"
	"errors"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	. "Stream-PT/forward"
	"Stream-PT/ggufindex"
	ggufmmap "Stream-PT/ggufmap"
)

func prefillEngineFixture(t *testing.T) *Engine {
	t.Helper()
	const dim, vocab = 32, 3
	path := filepath.Join(t.TempDir(), "prefill.bin")
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
	ones := make([]float32, dim)
	for i := range ones {
		ones[i] = 1
	}
	norm := add("norm", 0, []uint64{dim}, f32(ones))
	embeddings := make([]byte, vocab*q40BlockBytes)
	head := make([]byte, vocab*34)
	for row := 0; row < vocab; row++ {
		binary.LittleEndian.PutUint16(embeddings[row*q40BlockBytes:], 0x3c00)
		binary.LittleEndian.PutUint16(head[row*34:], 0x3c00)
		for i := 0; i < dim; i++ {
			value := int8(1)
			if row == 1 || row == 2 && i >= dim/2 {
				value = -1
			}
			head[row*34+2+i] = byte(value)
			shift := uint(i / 16 * 4)
			embeddings[row*q40BlockBytes+2+i%16] |= byte(value+8) << shift
		}
	}
	tokenEmbd := add("token_embd.weight", 2, []uint64{dim, vocab}, embeddings)
	outputWeight := add("output.weight", 8, []uint64{dim, vocab}, head)
	identity := make([]byte, dim*q40BlockBytes)
	for row := 0; row < dim; row++ {
		block := identity[row*q40BlockBytes : (row+1)*q40BlockBytes]
		binary.LittleEndian.PutUint16(block, 0x3c00)
		for i := 2; i < len(block); i++ {
			block[i] = 0x88
		}
		block[2+row%16] += 1 << uint(row/16*4)
	}
	projection := add("projection", 2, []uint64{dim, dim}, identity)
	router := add("router", 0, []uint64{dim, 1}, f32(make([]float32, dim)))
	expertData := make([]byte, dim*mxfp4BlockBytes)
	for row := 0; row < dim; row++ {
		expertData[row*mxfp4BlockBytes] = 127
	}
	expert := add("expert", 39, []uint64{dim, dim, 1}, expertData)
	sinks := add("sinks", 0, []uint64{1}, f32([]float32{0}))
	if err := os.WriteFile(path, data, 0600); err != nil {
		t.Fatal(err)
	}
	reader, err := ggufmmap.Open(&ggufindex.Model{Paths: []string{path}})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := reader.Close(); err != nil {
			t.Error(err)
		}
	})
	cfg := &Config{HiddenDim: dim, NumLayers: 1, NumHeads: 1, NumKVHeads: 1, HeadDim: dim,
		NumExperts: 1, NumExpertsUsed: 1, ExpertHiddenDim: dim, VocabSize: vocab,
		RMSNormEps: 1e-5, RopeFreqBase: 10000, EOS: -1}
	return &Engine{
		Reader: reader, Config: cfg, Options: EngineOptions{Workers: 1},
		TokenEmbd: tokenEmbd, OutputNorm: norm, OutputWeight: outputWeight,
		Layers: []LayerWeights{{AttnNorm: norm, AttnQ: projection, AttnK: projection,
			AttnV: projection, AttnSinks: sinks, AttnOutput: projection, PostAttnNorm: norm,
			FFNGateInp: router, FFNGateExps: expert, FFNUpExps: expert, FFNDownExps: expert}},
		KVCache: NewKVCache(1, 16, 1, dim), X: make([]float32, dim), Scratch: NewLayerScratch(cfg),
		Tokenizer: tokenizerFixture(t, []string{"a", "b", "c"}),
	}
}

func TestPrefillSkipsIntermediateOutputProjection(t *testing.T) {
	for _, stage := range []string{"final norm", "output argmax"} {
		t.Run(stage, func(t *testing.T) {
			engine := prefillEngineFixture(t)
			baseline := prefillEngineFixture(t)
			if stage == "final norm" {
				engine.OutputNorm = ggufindex.Tensor{}
			} else {
				engine.OutputWeight = ggufindex.Tensor{}
			}
			// Public ForwardToken must still run both output stages, even at position zero.
			if _, err := engine.ForwardToken(context.Background(), 0, 0); err == nil || !strings.Contains(err.Error(), stage) {
				t.Fatalf("ForwardToken error = %v, want %s", err, stage)
			}
			prompt := []int{0, 1, 2}
			for pos, tok := range prompt {
				if _, err := baseline.ForwardToken(context.Background(), tok, pos); err != nil {
					t.Fatal(err)
				}
			}
			calls := 0
			got, err := engine.Generate(context.Background(), prompt, 1, func(int, string) bool {
				calls++
				return true
			})
			// Invalid output tensors expose any premature norm/head call without timing assertions.
			if err == nil || !strings.Contains(err.Error(), "prefill step 2 (token 2): "+stage) {
				t.Fatalf("Generate error = %v, want output failure only at the last prompt token", err)
			}
			if got != nil || calls != 0 {
				t.Fatalf("failed prefill emitted tokens %v, callback count %d", got, calls)
			}
			if !reflect.DeepEqual(engine.KVCache, baseline.KVCache) || !reflect.DeepEqual(engine.X, baseline.X) {
				t.Fatal("skipping intermediate outputs changed layer state or KV cache")
			}
			for pos := range prompt {
				if engine.KVCache.Values[0][pos][0] == 0 {
					t.Fatalf("prefill did not populate KV position %d", pos)
				}
			}
		})
	}
}

func fullProjectionGenerate(t *testing.T, engine *Engine, prompt []int, maxNewTokens int,
	onToken func(int, string) bool) []int {
	t.Helper()
	if maxNewTokens <= 0 {
		maxNewTokens = 64
	}
	var next int
	var err error
	for pos, tok := range prompt {
		next, err = engine.ForwardToken(context.Background(), tok, pos)
		if err != nil {
			t.Fatal(err)
		}
	}
	var generated []int
	for i := 0; i < maxNewTokens; i++ {
		if engine.IsEOS(next) {
			break
		}
		generated = append(generated, next)
		if onToken != nil && !onToken(next, engine.Tokenizer.Decode([]int{next})) {
			break
		}
		if i+1 == maxNewTokens {
			break
		}
		next, err = engine.ForwardToken(context.Background(), next, len(prompt)+i)
		if err != nil {
			t.Fatal(err)
		}
	}
	return generated
}

func TestPrefillMatchesFullProjection(t *testing.T) {
	for _, tc := range []struct {
		name   string
		prompt []int
		limit  int
		stopAt int
		eos    bool
	}{
		{name: "single prompt", prompt: []int{1}, limit: 4},
		{name: "multiple prompt", prompt: []int{0, 1, 2}, limit: 4},
		{name: "one new token", prompt: []int{2, 0, 1}, limit: 1},
		{name: "callback stop", prompt: []int{0, 1, 2}, limit: 4, stopAt: 2},
		{name: "EOS", prompt: []int{0, 1, 2}, limit: 4, eos: true},
		{name: "default limit with EOS", prompt: []int{1, 2}, eos: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			engine, baseline := prefillEngineFixture(t), prefillEngineFixture(t)
			if tc.eos {
				probe := prefillEngineFixture(t)
				var next int
				for pos, tok := range tc.prompt {
					var err error
					next, err = probe.ForwardToken(context.Background(), tok, pos)
					if err != nil {
						t.Fatal(err)
					}
				}
				engine.Config.EOS, baseline.Config.EOS = next, next
			}
			var gotText, wantText []string
			want := fullProjectionGenerate(t, baseline, tc.prompt, tc.limit, func(_ int, text string) bool {
				wantText = append(wantText, text)
				return tc.stopAt == 0 || len(wantText) < tc.stopAt
			})
			got, err := engine.Generate(context.Background(), tc.prompt, tc.limit, func(_ int, text string) bool {
				gotText = append(gotText, text)
				return tc.stopAt == 0 || len(gotText) < tc.stopAt
			})
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got, want) || !reflect.DeepEqual(gotText, wantText) {
				t.Fatalf("tokens/text = %v/%v, want %v/%v", got, gotText, want, wantText)
			}
			if !reflect.DeepEqual(engine.X, baseline.X) || !reflect.DeepEqual(engine.KVCache, baseline.KVCache) {
				t.Fatal("generated tokens match but layer state or KV cache differs")
			}
		})
	}
}

func TestPrefillRetainsValidation(t *testing.T) {
	t.Run("invalid intermediate token", func(t *testing.T) {
		engine := prefillEngineFixture(t)
		engine.OutputNorm = ggufindex.Tensor{}
		_, err := engine.Generate(context.Background(), []int{0, -1, 2}, 1, nil)
		if err == nil || !strings.Contains(err.Error(), "prefill step 1 (token -1): embedding lookup:") {
			t.Fatalf("error = %v, want intermediate embedding validation", err)
		}
	})
	t.Run("intermediate capacity", func(t *testing.T) {
		engine := prefillEngineFixture(t)
		engine.KVCache.MaxPos = 1
		engine.OutputNorm = ggufindex.Tensor{}
		_, err := engine.Generate(context.Background(), []int{0, 1, 2}, 1, nil)
		if err == nil || !strings.Contains(err.Error(), "prefill step 1 (token 1): position 1 outside KV cache capacity 1") {
			t.Fatalf("error = %v, want intermediate position validation", err)
		}
	})
	t.Run("nil context", func(t *testing.T) {
		engine := prefillEngineFixture(t)
		_, err := engine.Generate(nil, []int{0, 1}, 1, nil)
		if err == nil || !strings.Contains(err.Error(), "prefill step 0 (token 0): nil context") {
			t.Fatalf("error = %v, want nil context validation", err)
		}
	})
	t.Run("cancelled context", func(t *testing.T) {
		engine := prefillEngineFixture(t)
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		_, err := engine.Generate(ctx, []int{0, 1}, 1, nil)
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("error = %v, want cancellation", err)
		}
	})
}
