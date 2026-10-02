//go:build goexperiment.simd

package tests

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"
	"unicode/utf8"

	. "Stream-PT/forward"
	"Stream-PT/ggufindex"
)

func optimizationByteAlphabet() [256]rune {
	var alphabet [256]rune
	next := rune(256)
	for b := range alphabet {
		if b >= '!' && b <= '~' || b >= '¡' && b <= '¬' || b >= '®' && b <= 'ÿ' {
			alphabet[b] = rune(b)
		} else {
			alphabet[b] = next
			next++
		}
	}
	return alphabet
}

func optimizationBPE(text string) string {
	alphabet := optimizationByteAlphabet()
	runes := make([]rune, len(text))
	for i := 0; i < len(text); i++ {
		runes[i] = alphabet[text[i]]
	}
	return string(runes)
}

func optimizationLegacyEncode(tok *Tokenizer, text string) []int {
	alphabet := optimizationByteAlphabet()
	runes := make([]rune, len(text))
	for i := 0; i < len(text); i++ {
		runes[i] = alphabet[text[i]]
	}
	maxLen := 1
	for token := range tok.TokenMap {
		maxLen = max(maxLen, utf8.RuneCountInString(token))
	}
	var ids []int
	for i := 0; i < len(runes); {
		matched := false
		for j := min(len(runes), i+maxLen); j > i; j-- {
			if id, ok := tok.TokenMap[string(runes[i:j])]; ok {
				ids = append(ids, id)
				i = j
				matched = true
				break
			}
		}
		if !matched {
			i++
		}
	}
	return ids
}

func optimizationTokenizerModel(tb testing.TB, tokens []string) *ggufindex.Model {
	tb.Helper()
	var data bytes.Buffer
	data.WriteString("metadata offset")
	start := data.Len()
	if err := binary.Write(&data, binary.LittleEndian, uint32(8)); err != nil {
		tb.Fatal(err)
	}
	if err := binary.Write(&data, binary.LittleEndian, uint64(len(tokens))); err != nil {
		tb.Fatal(err)
	}
	for _, token := range tokens {
		if err := binary.Write(&data, binary.LittleEndian, uint64(len(token))); err != nil {
			tb.Fatal(err)
		}
		data.WriteString(token)
	}
	path := filepath.Join(tb.TempDir(), "tokenizer.bin")
	if err := os.WriteFile(path, data.Bytes(), 0600); err != nil {
		tb.Fatal(err)
	}
	return &ggufindex.Model{Metadata: map[string]ggufindex.MetadataValue{
		"tokenizer.ggml.tokens": {Type: 9, Range: ggufindex.Range{
			File: path, Start: uint64(start), End: uint64(data.Len()),
		}},
	}}
}

func TestTokenizerOptimizationGreedy(t *testing.T) {
	tokens := []string{"", "a", "ab", "abc", "abcdx", "b", "c", "d", "!", "<|end|>", "a"}
	alphabet := optimizationByteAlphabet()
	for _, r := range alphabet {
		tokens = append(tokens, string(r))
	}
	tokens = append(tokens, optimizationBPE(" äöüß €\n\t"), "abc")
	tok, err := LoadTokenizer(optimizationTokenizerModel(t, tokens))
	if err != nil {
		t.Fatal(err)
	}
	for _, text := range []string{"", "abcabcd!", "abczabcdx", "<|end|>abc", " äöüß €\n\t", "🙂", "\x00\xff\xfe", strings.Repeat("abc?", 200)} {
		t.Run(fmt.Sprintf("%q", text), func(t *testing.T) {
			got, want := tok.Encode(text), optimizationLegacyEncode(tok, text)
			if !slices.Equal(got, want) {
				t.Fatalf("Encode = %v, want %v", got, want)
			}
			if decoded := tok.Decode(got); decoded != text {
				t.Fatalf("Decode = %q, want %q", decoded, text)
			}
		})
	}
	if got := tok.Encode("abc"); !slices.Equal(got, []int{len(tokens) - 1}) {
		t.Fatalf("duplicate longest token = %v", got)
	}
}

func TestTokenizerOptimizationFallback(t *testing.T) {
	tok, err := LoadTokenizer(optimizationTokenizerModel(t, []string{"", "a", "abc", "abcdx", "b", "d", "<|end|>"}))
	if err != nil {
		t.Fatal(err)
	}
	if got, want := tok.Encode("abcd?abz<|end|>"), []int{2, 5, 1, 4, 6}; !slices.Equal(got, want) {
		t.Fatalf("fallback = %v, want %v", got, want)
	}
	var nilTokenizer *Tokenizer
	if nilTokenizer.Encode("a") != nil || nilTokenizer.Decode([]int{0}) != "" {
		t.Fatal("nil tokenizer behavior changed")
	}
}

func TestTokenizerOptimizationReference(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	for vocab := 0; vocab < 8; vocab++ {
		var tokens []string
		for range 80 {
			text := make([]byte, rng.Intn(12)+1)
			for i := range text {
				text[i] = byte('a' + rng.Intn(4))
			}
			tokens = append(tokens, string(text))
		}
		tok, err := LoadTokenizer(optimizationTokenizerModel(t, tokens))
		if err != nil {
			t.Fatal(err)
		}
		for range 40 {
			text := make([]byte, rng.Intn(128))
			for i := range text {
				text[i] = byte('a' + rng.Intn(6))
			}
			if got, want := tok.Encode(string(text)), optimizationLegacyEncode(tok, string(text)); !slices.Equal(got, want) {
				t.Fatalf("vocabulary %d, text %q: %v, want %v", vocab, text, got, want)
			}
		}
	}
}

func TestTokenizerOptimizationBufferedLoad(t *testing.T) {
	text := strings.Repeat(" ä", 2048)
	tokens := []string{optimizationBPE(text), "<|end|>", "", "<|end|>"}
	tok, err := LoadTokenizer(optimizationTokenizerModel(t, tokens))
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(tok.Tokens, tokens) || tok.TokenMap["<|end|>"] != 3 {
		t.Fatal("vocabulary across reader buffers or duplicate ID changed")
	}
	if got := tok.Encode(text + "<|end|>"); !slices.Equal(got, []int{0, 3}) {
		t.Fatalf("long multi-byte token = %v, want [0 3]", got)
	}
	var ids []int
	allocs := testing.AllocsPerRun(20, func() { ids = tok.Encode(text) })
	if !slices.Equal(ids, []int{0}) || allocs > 1 {
		t.Fatalf("single-token Encode = %v, %.0f allocations; only the result slice may allocate", ids, allocs)
	}
	empty, err := LoadTokenizer(optimizationTokenizerModel(t, nil))
	if err != nil {
		t.Fatal(err)
	}
	if got := empty.Encode("abc"); got != nil {
		t.Fatalf("empty vocabulary = %v", got)
	}
}

func TestTokenizerOptimizationPublicLiteral(t *testing.T) {
	tok := &Tokenizer{Tokens: []string{"a", "ab", "<|end|>", "Ġ"}, TokenMap: map[string]int{"a": 0, "ab": 1, "<|end|>": 2, "Ġ": 3}}
	var wg sync.WaitGroup
	for range 16 {
		wg.Go(func() {
			if got, want := tok.Encode("ab <|end|>"), []int{1, 3, 2}; !slices.Equal(got, want) {
				t.Errorf("literal Encode = %v, want %v", got, want)
			}
		})
	}
	wg.Wait()
	if got := tok.Decode([]int{-1, 1, 3, 99}); got != "abĠ" {
		t.Fatalf("literal Decode semantics changed: %q", got)
	}
	if got := (&Tokenizer{Tokens: []string{"a"}}).Encode("a"); len(got) != 0 {
		t.Fatalf("Tokens-only literal unexpectedly encodes: %v", got)
	}
}

func TestTokenizerOptimizationLoadErrors(t *testing.T) {
	for _, tc := range []struct {
		name string
		data []byte
	}{
		{"header", []byte{8}},
		{"type", binary.LittleEndian.AppendUint64(binary.LittleEndian.AppendUint32(nil, 4), 0)},
		{"count", binary.LittleEndian.AppendUint64(binary.LittleEndian.AppendUint32(nil, 8), 1000001)},
		{"length", binary.LittleEndian.AppendUint64(binary.LittleEndian.AppendUint32(nil, 8), 1)},
		{"body", append(binary.LittleEndian.AppendUint64(binary.LittleEndian.AppendUint64(binary.LittleEndian.AppendUint32(nil, 8), 1), 4), 'a')},
	} {
		t.Run(tc.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "invalid.bin")
			if err := os.WriteFile(path, tc.data, 0600); err != nil {
				t.Fatal(err)
			}
			model := &ggufindex.Model{Metadata: map[string]ggufindex.MetadataValue{
				"tokenizer.ggml.tokens": {Range: ggufindex.Range{File: path, End: uint64(len(tc.data))}},
			}}
			if _, err := LoadTokenizer(model); err == nil {
				t.Fatal("expected malformed vocabulary error")
			}
		})
	}
	if _, err := LoadTokenizer(&ggufindex.Model{}); err == nil {
		t.Fatal("expected missing vocabulary error")
	}
}

func BenchmarkTokenizerOptimizationEncode(b *testing.B) {
	tokens := []string{"a", "ab", "abc", strings.Repeat("a", 128) + "z"}
	tok, err := LoadTokenizer(optimizationTokenizerModel(b, tokens))
	if err != nil {
		b.Fatal(err)
	}
	for _, text := range []struct{ name, value string }{
		{"prefix_miss", strings.Repeat("a", 1024)},
		{"short_matches", strings.Repeat("abc?", 256)},
	} {
		b.Run(text.name, func(b *testing.B) {
			for _, impl := range []struct {
				name   string
				encode func(string) []int
			}{{"Encode", tok.Encode}, {"Legacy", func(text string) []int { return optimizationLegacyEncode(tok, text) }}} {
				b.Run(impl.name, func(b *testing.B) {
					b.ReportAllocs()
					b.SetBytes(int64(len(text.value)))
					for b.Loop() {
						_ = impl.encode(text.value)
					}
				})
			}
		})
	}
}

func BenchmarkTokenizerOptimizationLoad(b *testing.B) {
	tokens := make([]string, 8192)
	for i := range tokens {
		tokens[i] = fmt.Sprintf("Ġtoken%d", i)
	}
	model := optimizationTokenizerModel(b, tokens)
	b.ReportAllocs()
	for b.Loop() {
		if _, err := LoadTokenizer(model); err != nil {
			b.Fatal(err)
		}
	}
}

func optimizationLargeVocabulary(count int) ([]string, map[string]int) {
	tokens := make([]string, count)
	tokenMap := make(map[string]int, count)
	for i := range tokens {
		// Shared prefixes and all 256 BPE bytes, including multi-byte Unicode glyphs.
		text := fmt.Sprintf(" group%03d/token%06d/", i%1000, i)
		text += string([]byte{byte(i), byte(i >> 8)})
		tokens[i] = optimizationBPE(text)
		tokenMap[tokens[i]] = i
	}
	return tokens, tokenMap
}

func TestTokenizerOptimizationStartupAllocations(t *testing.T) {
	tokens, tokenMap := optimizationLargeVocabulary(20000)
	var ids []int
	allocs := testing.AllocsPerRun(3, func() {
		tok := &Tokenizer{Tokens: tokens, TokenMap: tokenMap}
		ids = tok.Encode(" group000/token000000/\x00\x00")
	})
	if !slices.Equal(ids, []int{0}) || allocs > 100 {
		t.Fatalf("startup = %v, %.0f allocations; arena must not allocate per prefix", ids, allocs)
	}
}

func TestTokenizerOptimizationWideLookup(t *testing.T) {
	tokens := []string{"", "🙂", "a🙂", "a\xff", "�"}
	for b := range 256 {
		tokens = append(tokens, optimizationBPE(string([]byte{byte(b)})))
		tokens = append(tokens, optimizationBPE(string([]byte{'a', byte(b)})))
	}
	tok, err := LoadTokenizer(optimizationTokenizerModel(t, tokens))
	if err != nil {
		t.Fatal(err)
	}
	for b := range 256 {
		text := string([]byte{'a', byte(b), byte(b), 'a', byte(255 - b)})
		if got, want := tok.Encode(text), optimizationLegacyEncode(tok, text); !slices.Equal(got, want) {
			t.Fatalf("byte %d: %v, want %v", b, got, want)
		}
	}
}

func TestTokenizerOptimizationSparseLookup(t *testing.T) {
	for _, count := range []int{1, 8, 9, 128} {
		t.Run(fmt.Sprintf("edges_%d", count), func(t *testing.T) {
			tokens := []string{"a"}
			for i := range count {
				tokens = append(tokens, optimizationBPE(string([]byte{'a', byte(2 * i)})))
			}
			tok, err := LoadTokenizer(optimizationTokenizerModel(t, tokens))
			if err != nil {
				t.Fatal(err)
			}
			for b := range 256 {
				text := string([]byte{'a', byte(b), 'a'})
				if got, want := tok.Encode(text), optimizationLegacyEncode(tok, text); !slices.Equal(got, want) {
					t.Fatalf("byte %d: %v, want %v", b, got, want)
				}
			}
		})
	}
}

func BenchmarkTokenizerOptimizationStartup(b *testing.B) {
	for _, count := range []int{8192, 200000} {
		b.Run(fmt.Sprintf("tokens_%d", count), func(b *testing.B) {
			tokens, tokenMap := optimizationLargeVocabulary(count)
			b.Run("InitEncoder", func(b *testing.B) {
				b.ReportAllocs()
				for b.Loop() {
					tok := &Tokenizer{Tokens: tokens, TokenMap: tokenMap}
					if ids := tok.Encode(" group000/token000000/\x00\x00"); !slices.Equal(ids, []int{0}) {
						b.Fatal(ids)
					}
				}
			})
			model := optimizationTokenizerModel(b, tokens)
			b.Run("LoadTokenizer", func(b *testing.B) {
				b.ReportAllocs()
				for b.Loop() {
					if _, err := LoadTokenizer(model); err != nil {
						b.Fatal(err)
					}
				}
			})
		})
	}
}
