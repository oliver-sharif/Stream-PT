//go:build goexperiment.simd

package forward

import (
	"encoding/binary"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"Stream-PT/ggufindex"
)

func bpeMetadata(t *testing.T, model *ggufindex.Model, key string, typ uint32, value []byte) {
	t.Helper()
	path := filepath.Join(t.TempDir(), "metadata.bin")
	if err := os.WriteFile(path, value, 0600); err != nil {
		t.Fatal(err)
	}
	model.Metadata[key] = ggufindex.MetadataValue{Type: typ, Range: ggufindex.Range{File: path, End: uint64(len(value))}}
}

func bpeModel(t *testing.T, tokens, merges []string) *ggufindex.Model {
	t.Helper()
	model := optimizationTokenizerModel(t, tokens)
	data := binary.LittleEndian.AppendUint32(nil, 8)
	data = binary.LittleEndian.AppendUint64(data, uint64(len(merges)))
	for _, merge := range merges {
		data = binary.LittleEndian.AppendUint64(data, uint64(len(merge)))
		data = append(data, merge...)
	}
	bpeMetadata(t, model, "tokenizer.ggml.merges", 9, data)
	bpeMetadata(t, model, "tokenizer.ggml.pre", 8, append(binary.LittleEndian.AppendUint64(nil, 6), "gpt-4o"...))
	return model
}

func TestTokenizerBPERankReproduction(t *testing.T) {
	model := bpeModel(t, []string{"a", "b", "c", "ab", "bc"}, []string{"b c", "a b"})
	tok, err := LoadTokenizer(model)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := tok.Encode("abc"), []int{0, 4}; !slices.Equal(got, want) {
		t.Fatalf("rank-based BPE = %v, want %v (not greedy ab+c)", got, want)
	}
}

func TestTokenizerBPEPreBoundaries(t *testing.T) {
	model := bpeModel(t, []string{"a", "1", "a1"}, []string{"a 1"})
	tok, err := LoadTokenizer(model)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := tok.Encode("a1"), []int{0, 1}; !slices.Equal(got, want) {
		t.Fatalf("pretoken boundaries = %v, want %v", got, want)
	}
}
