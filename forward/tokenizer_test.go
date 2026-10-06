//go:build goexperiment.simd

package forward

import (
	"os"
	"path/filepath"
	"testing"

	"Stream-PT/ggufindex"
)

func TestTokenizerInspection(t *testing.T) {
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

	for k, v := range model.Metadata {
		if len(k) > 9 && k[:9] == "tokenizer" {
			t.Logf("tokenizer meta key: %s = %+v", k, v)
		}
	}

	cfg, _ := NewConfigFromModel(model)
	t.Logf("Config BOS: %d, EOS: %d", cfg.BOS, cfg.EOS)

	tok, err := LoadTokenizer(model)
	if err != nil {
		t.Fatal(err)
	}

	t.Logf("Loaded %d tokens", len(tok.Tokens))

	testCases := []string{
		"Hallo, sag nur Ja!",
		"Guten Tag! Wie geht es dir?\nSehr gut, danke.",
		"Special: äöüß € 123 \t \r\n",
	}

	if tt, ok := model.Metadata["tokenizer.ggml.token_type"]; ok {
		t.Logf("token_type Range: %+v", tt.Range)
	}

	for _, tc := range testCases {
		enc := tok.Encode(tc)
		dec := tok.Decode(enc)
		t.Logf("Original: %q", tc)
		t.Logf("Encoded:  %v", enc)
		t.Logf("Decoded:  %q", dec)
		if dec != tc {
			t.Errorf("Mismatch! Got %q, want %q", dec, tc)
		}
	}
}

func TestEOSDetection(t *testing.T) {
	cfg := &Config{
		EOS:       200002,
		EOSTokens: []int{200002, 199999, 200007},
	}

	if !cfg.IsEOS(200002) {
		t.Errorf("expected 200002 to be EOS")
	}
	if !cfg.IsEOS(199999) {
		t.Errorf("expected 199999 to be EOS")
	}
	if !cfg.IsEOS(200007) {
		t.Errorf("expected 200007 to be EOS")
	}
	if cfg.IsEOS(100) {
		t.Errorf("token 100 should not be EOS")
	}

	engine := &Engine{
		Config: cfg,
		Tokenizer: &Tokenizer{
			Tokens: []string{"hello", "<|endoftext|>", "world", "<|end|>", "</s>"},
		},
	}

	if !engine.IsEOS(200002) {
		t.Errorf("engine should recognize config EOS 200002")
	}
	if !engine.IsEOS(1) { // "<|endoftext|>"
		t.Errorf("engine should recognize token 1 (<|endoftext|>) as EOS")
	}
	if !engine.IsEOS(3) { // "<|end|>"
		t.Errorf("engine should recognize token 3 (<|end|>) as EOS")
	}
	if !engine.IsEOS(4) { // "</s>"
		t.Errorf("engine should recognize token 4 (</s>) as EOS")
	}
	if engine.IsEOS(0) { // "hello"
		t.Errorf("token 0 ('hello') should not be EOS")
	}
	if engine.IsEOS(2) { // "world"
		t.Errorf("token 2 ('world') should not be EOS")
	}
}
