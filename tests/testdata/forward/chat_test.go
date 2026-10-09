//go:build goexperiment.simd

package forward

import (
	"context"
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"Stream-PT/ggufindex"
	ggufmmap "Stream-PT/ggufmap"
)

func TestEncodeChatPromptMissingSpecialTokens(t *testing.T) {
	for _, tok := range []*Tokenizer{nil, {TokenMap: map[string]int{}}} {
		if _, err := tok.EncodeChatPrompt("Yes!"); err == nil {
			t.Fatal("expected error for missing Harmony tokens")
		}
	}
}

func TestGPTOSSStopTokenAliases(t *testing.T) {
	engine := &Engine{Tokenizer: &Tokenizer{Tokens: []string{"Yes", "!", "\n\n", "<|fim_suffix|>", "<|ghissue|>"}}}
	for id := range engine.Tokenizer.Tokens {
		if got, want := engine.IsEOS(id), id >= 3; got != want {
			t.Errorf("IsEOS(%q) = %v, want %v", engine.Tokenizer.Tokens[id], got, want)
		}
	}
}

func TestGenerateStopsAtGPTOSSCompletion(t *testing.T) {
	const dim, vocab = 32, 5
	path := filepath.Join(t.TempDir(), "generation.bin")
	var data []byte
	add := func(typ uint32, shape []uint64, values []byte) ggufindex.Tensor {
		start := len(data)
		data = append(data, values...)
		return ggufindex.Tensor{Type: typ, Shape: shape,
			Range: ggufindex.Range{File: path, Start: uint64(start), End: uint64(len(data))}}
	}
	embeddings := make([]byte, vocab*18)
	for id := range vocab {
		binary.LittleEndian.PutUint16(embeddings[id*18:], 0x3c00)
		for i := 2; i < 18; i++ {
			embeddings[id*18+i] = 0x88
		}
		embeddings[id*18+2+id] = 0x89
	}
	embd := add(2, []uint64{dim, vocab}, embeddings)
	norms := make([]byte, dim*4)
	for i := range dim {
		binary.LittleEndian.PutUint32(norms[i*4:], math.Float32bits(1))
	}
	norm := add(0, []uint64{dim}, norms)
	weights := make([]byte, vocab*34)
	for id := 1; id < vocab; id++ {
		binary.LittleEndian.PutUint16(weights[id*34:], 0x3c00)
		weights[id*34+2+id-1] = 1
	}
	output := add(8, []uint64{dim, vocab}, weights)
	if err := os.WriteFile(path, data, 0600); err != nil {
		t.Fatal(err)
	}
	reader, err := ggufmmap.Open(&ggufindex.Model{Paths: []string{path}})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = reader.Close() })
	engine := &Engine{
		Reader: reader, Config: &Config{EOS: 99, RMSNormEps: 1e-5},
		TokenEmbd: embd, OutputNorm: norm, OutputWeight: output,
		KVCache: NewKVCache(0, 3, 1, 2), X: make([]float32, dim),
		Scratch:   &LayerScratch{NormedX: make([]float32, dim)},
		Tokenizer: &Tokenizer{Tokens: []string{"prompt", "Yes", "!", "<|fim_suffix|>", "It"}},
	}
	var emitted []int
	got, err := engine.Generate(context.Background(), []int{0}, 10, func(id int, text string) bool {
		emitted = append(emitted, id)
		return true
	})
	if err != nil {
		t.Fatal(err)
	}
	// A further forward pass after EOS would exceed the three-position cache.
	want := []int{1, 2}
	if !slices.Equal(got, want) || !slices.Equal(emitted, want) {
		t.Fatalf("generated = %v, emitted = %v, want %v (no EOS or trailing text)", got, emitted, want)
	}
}
