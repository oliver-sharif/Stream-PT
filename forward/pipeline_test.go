//go:build goexperiment.simd

package forward

import (
	"context"
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"

	"Stream-PT/ggufindex"
	ggufmmap "Stream-PT/ggufmap"
)

func TestConfigAndTokenizer(t *testing.T) {
	paths := []string{
		filepath.Join("..", "model", "gpt-oss-120b-Q4_0-00001-of-00002.gguf"),
		filepath.Join("..", "model", "gpt-oss-120b-Q4_0-00002-of-00002.gguf"),
	}
	for _, p := range paths {
		if _, err := os.Stat(p); err != nil {
			t.Skip("Model files not present")
		}
	}

	model, err := ggufindex.Open(paths)
	if err != nil {
		t.Fatal(err)
	}

	cfg, err := NewConfigFromModel(model)
	if err != nil {
		t.Fatalf("NewConfigFromModel failed: %v", err)
	}
	if cfg.HiddenDim != 2880 || cfg.NumLayers != 36 || cfg.NumHeads != 64 {
		t.Fatalf("Unexpected config: %+v", cfg)
	}

	tok, err := LoadTokenizer(model)
	if err != nil {
		t.Fatalf("LoadTokenizer failed: %v", err)
	}
	if len(tok.Tokens) != cfg.VocabSize {
		t.Fatalf("Vocab size mismatch: %d != %d", len(tok.Tokens), cfg.VocabSize)
	}

	encoded := tok.Encode("hello")
	if len(encoded) == 0 {
		t.Fatalf("Encoding failed for 'hello'")
	}
	chat, err := tok.EncodeChatPrompt("Hello, just say Yes!")
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("Chat prompt tokens: %v", chat)
	if len(chat) <= len(tok.Encode("Hello, just say Yes!")) {
		t.Fatal("chat prompt is missing the Harmony framing")
	}
}

func TestEmbeddingLookup(t *testing.T) {
	paths := []string{
		filepath.Join("..", "model", "gpt-oss-120b-Q4_0-00001-of-00002.gguf"),
		filepath.Join("..", "model", "gpt-oss-120b-Q4_0-00002-of-00002.gguf"),
	}
	for _, p := range paths {
		if _, err := os.Stat(p); err != nil {
			t.Skip("Model files not present")
		}
	}

	model, err := ggufindex.Open(paths)
	if err != nil {
		t.Fatal(err)
	}

	reader, err := ggufmmap.Open(model)
	if err != nil {
		t.Fatal(err)
	}
	defer reader.Close()

	embdTensor, ok := model.TensorByName("token_embd.weight")
	if !ok {
		t.Fatal("missing token_embd.weight")
	}

	out := make([]float32, 2880)
	if err := ReadEmbedding(context.Background(), reader, embdTensor, 10, out); err != nil {
		t.Fatalf("ReadEmbedding failed: %v", err)
	}

	var hasNonZero bool
	for _, v := range out {
		if v != 0 {
			hasNonZero = true
			break
		}
	}
	if !hasNonZero {
		t.Fatal("All embedding values are zero")
	}
}

func TestRoPEAndAttention(t *testing.T) {
	const (
		numHeads   = 4
		numKVHeads = 2
		headDim    = 8
		hiddenDim  = numHeads * headDim
		kvDim      = numKVHeads * headDim
	)

	q := make([]float32, hiddenDim)
	k := make([]float32, kvDim)
	v := make([]float32, kvDim)
	for i := range q {
		q[i] = float32(i + 1)
	}
	for i := range k {
		k[i] = float32(i + 1)
		v[i] = float32(i + 2)
	}

	ApplyRoPE(q, k, 0, numHeads, numKVHeads, headDim, 10000.0)

	cache := NewKVCache(1, 10, numKVHeads, headDim)
	out := make([]float32, hiddenDim)

	attentionForTest(q, k, v, cache, 0, 0, numHeads, numKVHeads, headDim, out, AttentionOptions{})

	var hasNonZero bool
	for _, val := range out {
		if val != 0 && !math.IsNaN(float64(val)) {
			hasNonZero = true
			break
		}
	}
	if !hasNonZero {
		t.Fatal("Attention output is zero or NaN")
	}
}

func TestQ80Dot(t *testing.T) {
	const elements = 32
	row := make([]byte, 34)
	// Scale = 1.0 (fp16 0x3c00)
	binary.LittleEndian.PutUint16(row[:2], 0x3c00)
	for i := range elements {
		row[2+i] = byte(int8(i - 16))
	}

	x := make([]float32, elements)
	for i := range x {
		x[i] = 1.0
	}

	reader, tensor := tensorFixture(t, 8, []uint64{elements, 1}, row)
	y := make([]float32, 1)
	if err := q80KernelRowsForTest(reader, tensor, x, y); err != nil {
		t.Fatal(err)
	}
	got := y[0]
	var want float32
	for i := range elements {
		want += float32(int8(row[2+i]))
	}

	if math.Abs(float64(got-want)) > 1e-4 {
		t.Fatalf("dotQ80 = %g, want %g", got, want)
	}
}

func TestMXFP4Dot(t *testing.T) {
	const elements = 32
	row := make([]byte, 17)
	// Scale = 127 -> 2^(127-127) = 1.0
	row[0] = 127
	for i := range 16 {
		row[1+i] = 0x22 // 0x2 = 1.0 for low and high nibble
	}

	x := make([]float32, elements)
	for i := range x {
		x[i] = 2.0
	}

	reader, tensor := tensorFixture(t, 39, []uint64{elements, 1, 1}, row)
	y := make([]float32, 1)
	if err := mulMXFP4ForTest(context.Background(), reader, tensor, 0, x, nil, y, Q40Options{}); err != nil {
		t.Fatal(err)
	}
	got := y[0]
	want := float32(elements * 2) // 32 * 1.0 * 2.0 = 64.0

	if math.Abs(float64(got-want)) > 1e-4 {
		t.Fatalf("dotMXFP4 = %g, want %g", got, want)
	}
}
