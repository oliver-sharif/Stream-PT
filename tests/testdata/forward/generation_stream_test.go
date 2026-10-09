//go:build goexperiment.simd

package forward

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"unicode/utf8"

	"Stream-PT/ggufindex"
	ggufmmap "Stream-PT/ggufmap"
)

func streamTestEngine(t *testing.T, tokens []string) *Engine {
	t.Helper()
	const dim = 32
	path := filepath.Join(t.TempDir(), "stream.bin")
	var data []byte
	add := func(typ uint32, shape []uint64, values []byte) ggufindex.Tensor {
		start := len(data)
		data = append(data, values...)
		return ggufindex.Tensor{Type: typ, Shape: shape, Range: ggufindex.Range{File: path, Start: uint64(start), End: uint64(len(data))}}
	}
	embeddings := make([]byte, len(tokens)*18)
	for id := range tokens {
		binary.LittleEndian.PutUint16(embeddings[id*18:], 0x3c00)
		for i := 2; i < 18; i++ {
			embeddings[id*18+i] = 0x88
		}
		embeddings[id*18+2+id] = 0x89
	}
	embd := add(2, []uint64{dim, uint64(len(tokens))}, embeddings)
	norms := make([]byte, dim*4)
	for i := range dim {
		binary.LittleEndian.PutUint32(norms[i*4:], math.Float32bits(1))
	}
	norm := add(0, []uint64{dim}, norms)
	weights := make([]byte, len(tokens)*34)
	for id := 1; id < len(tokens); id++ {
		binary.LittleEndian.PutUint16(weights[id*34:], 0x3c00)
		weights[id*34+2+id-1] = 1
	}
	output := add(8, []uint64{dim, uint64(len(tokens))}, weights)
	if err := os.WriteFile(path, data, 0600); err != nil {
		t.Fatal(err)
	}
	reader, err := ggufmmap.Open(&ggufindex.Model{Paths: []string{path}})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = reader.Close() })
	tokenMap := make(map[string]int)
	for id, token := range tokens {
		tokenMap[token] = id
	}
	return &Engine{
		Reader: reader, Config: &Config{EOS: 99, RMSNormEps: 1e-5},
		TokenEmbd: embd, OutputNorm: norm, OutputWeight: output,
		KVCache: NewKVCache(0, len(tokens), 1, 2), X: make([]float32, dim),
		Scratch:   &LayerScratch{NormedX: make([]float32, dim)},
		Tokenizer: &Tokenizer{Tokens: tokens, TokenMap: tokenMap},
	}
}

func TestHarmonyMessageEndContinuesGeneration(t *testing.T) {
	engine := streamTestEngine(t, []string{"prompt", "analysis", "<|end|>", "final", "<|return|>", "<|call|>"})
	engine.Config.EOS = 2
	engine.Config.EOSTokens = []int{2, 4, 5}
	options := DefaultSamplingOptions()
	options.Temperature = 0
	var emitted []int
	got, err := engine.GenerateWithSampling(context.Background(), []int{0}, 10, options, func(id int, text string) bool {
		emitted = append(emitted, id)
		return true
	})
	if err != nil || !slices.Equal(got, []int{1, 2, 3}) || !slices.Equal(got, emitted) {
		t.Fatalf("generated %v, emitted %v, error %v; want continuation through message end", got, emitted, err)
	}
}

func TestGenerationUTF8Streaming(t *testing.T) {
	b2u, u2b := initByteToUnicode()
	tokens := []string{"prompt"}
	for _, b := range []byte("\u00e4🙂") {
		tokens = append(tokens, string(b2u[b]))
	}
	tokens = append(tokens, "<|return|>")
	engine := streamTestEngine(t, tokens)
	engine.Tokenizer.unicodeToByte = u2b
	options := DefaultSamplingOptions()
	options.Temperature = 0
	var text strings.Builder
	_, err := engine.GenerateWithSampling(context.Background(), []int{0}, 10, options, func(id int, chunk string) bool {
		if !utf8.ValidString(chunk) {
			t.Errorf("invalid UTF-8 chunk for token %d: %x", id, chunk)
		}
		data, err := json.Marshal(chunk)
		if err != nil {
			t.Fatal(err)
		}
		var received string
		if err := json.Unmarshal(data, &received); err != nil {
			t.Fatal(err)
		}
		text.WriteString(received)
		return true
	})
	if err != nil || text.String() != "\u00e4🙂" {
		t.Fatalf("text %q, error %v", text.String(), err)
	}
}

func TestGenerationFinishReasons(t *testing.T) {
	for _, tc := range []struct {
		name, stop, reason string
		budget, capacity   int
		callback           bool
	}{
		{"completion", "<|return|>", "stop", 10, 3, true},
		{"handoff", "<|call|>", "tool_calls", 10, 3, true},
		{"budget", "<|return|>", "length", 1, 3, true},
		{"capacity", "<|return|>", "context_length", 10, 1, true},
		{"callback", "<|return|>", "callback", 10, 3, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			engine := streamTestEngine(t, []string{"prompt", "answer", tc.stop})
			engine.KVCache.MaxPos = tc.capacity
			options := DefaultSamplingOptions()
			options.Temperature = 0
			result, err := engine.GenerateWithSamplingResult(context.Background(), []int{0}, tc.budget, options, func(int, string) bool { return tc.callback })
			if err != nil || result.FinishReason != tc.reason || !slices.Equal(result.Tokens, []int{1}) {
				t.Fatalf("result %+v, error %v", result, err)
			}
			if tc.reason == "stop" || tc.reason == "tool_calls" {
				if result.StopToken == nil || *result.StopToken != 2 {
					t.Fatalf("missing stop token: %+v", result)
				}
			} else if result.StopToken != nil {
				t.Fatalf("unexpected stop token: %+v", result)
			}
		})
	}
	engine := streamTestEngine(t, []string{"prompt", "answer", "<|return|>"})
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	result, err := engine.GenerateWithSamplingResult(ctx, []int{0}, 10, DefaultSamplingOptions(), nil)
	if !errors.Is(err, context.Canceled) || result.FinishReason != "cancelled" {
		t.Fatalf("cancelled result %+v, error %v", result, err)
	}
}

func TestUTF8StreamIncompleteAndInvalid(t *testing.T) {
	var decoder utf8Stream
	if got := decoder.push("a\xf0\x9f", false); got != "a" {
		t.Fatalf("got %q", got)
	}
	if got := decoder.push("", true); got != "��" || !utf8.ValidString(got) {
		t.Fatalf("incomplete tail %q", got)
	}
	if got := decoder.push("\xffb", false); got != "�b" {
		t.Fatalf("invalid byte %q", got)
	}
}
