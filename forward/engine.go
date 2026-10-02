//go:build goexperiment.simd

package forward

import (
	"bufio"
	"context"
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"runtime"
	"unicode/utf8"

	"Stream-PT/ggufindex"
	ggufmmap "Stream-PT/ggufmap"
)

// EngineOptions configures runtime concurrency and streaming memory limits.
type EngineOptions struct {
	Workers          int    // Number of worker goroutines for matrix multiplications (default: runtime.GOMAXPROCS(0))
	WindowBytes      uint64 // Chunk window size in bytes for streaming mmap (default: 8 MiB)
	PrefillBatchSize int    // Prompt positions per weight pass (default: 32; one uses the decode path).
}

// Engine manages the end-to-end streaming inference pipeline.
type Engine struct {
	Model          *ggufindex.Model
	Reader         *ggufmmap.Reader
	Config         *Config
	Options        EngineOptions
	Layers         []LayerWeights
	TokenEmbd      ggufindex.Tensor
	OutputNorm     ggufindex.Tensor
	OutputWeight   ggufindex.Tensor
	KVCache        *KVCache
	X              []float32
	Scratch        *LayerScratch
	Tokenizer      *Tokenizer
	prefillScratch *prefillScratch
}

// NewEngine initializes the inference engine with default options.
func NewEngine(model *ggufindex.Model, reader *ggufmmap.Reader) (*Engine, error) {
	return NewEngineWithOptions(model, reader, EngineOptions{})
}

// NewEngineWithOptions initializes the inference engine with user-defined options.
func NewEngineWithOptions(model *ggufindex.Model, reader *ggufmmap.Reader, options EngineOptions) (*Engine, error) {
	if model == nil {
		return nil, fmt.Errorf("nil model")
	}
	if reader == nil {
		return nil, fmt.Errorf("nil reader")
	}

	if options.Workers <= 0 {
		options.Workers = runtime.GOMAXPROCS(0)
	}
	if options.WindowBytes == 0 {
		options.WindowBytes = DefaultQ40WindowBytes
	}
	if options.PrefillBatchSize < 0 {
		return nil, fmt.Errorf("invalid prefill batch size %d", options.PrefillBatchSize)
	}

	cfg, err := NewConfigFromModel(model)
	if err != nil {
		return nil, fmt.Errorf("config error: %w", err)
	}

	tokenEmbd, ok := model.TensorByName("token_embd.weight")
	if !ok {
		return nil, fmt.Errorf("missing token_embd.weight tensor")
	}

	outputNorm, ok := model.TensorByName("output_norm.weight")
	if !ok {
		return nil, fmt.Errorf("missing output_norm.weight tensor")
	}

	outputWeight, ok := model.TensorByName("output.weight")
	if !ok {
		return nil, fmt.Errorf("missing output.weight tensor")
	}

	layers := make([]LayerWeights, len(model.Layers))
	for i, l := range model.Layers {
		layers[i] = NewLayerWeights(l)
	}
	if len(layers) != cfg.NumLayers || cfg.NumHeads <= 0 || cfg.NumKVHeads <= 0 ||
		cfg.NumHeads%cfg.NumKVHeads != 0 || cfg.HeadDim <= 0 || cfg.HeadDim%2 != 0 {
		return nil, fmt.Errorf("invalid transformer dimensions or layer count")
	}

	const maxSeqLen = 2048
	kvCache := NewKVCache(cfg.NumLayers, maxSeqLen, cfg.NumKVHeads, cfg.HeadDim)

	tokenizer, err := LoadTokenizer(model)
	if err != nil {
		return nil, fmt.Errorf("tokenizer: %w", err)
	}

	if tokenizer != nil && cfg != nil {
		knownStopTokens := []string{
			"<|endoftext|>",
			"<|end|>",
			"<|return|>",
			"<|fim_suffix|>",
			"<|call|>",
			"<|ghissue|>",
			"<|eot_id|>",
			"<|im_end|>",
			"</s>",
			"<eos>",
			"<|end_of_text|>",
			"<|eom_id|>",
		}
		for _, stopStr := range knownStopTokens {
			if id, ok := tokenizer.TokenMap[stopStr]; ok {
				if !cfg.IsEOS(id) {
					cfg.EOSTokens = append(cfg.EOSTokens, id)
				}
			}
		}
	}

	return &Engine{
		Model:        model,
		Reader:       reader,
		Config:       cfg,
		Options:      options,
		Layers:       layers,
		TokenEmbd:    tokenEmbd,
		OutputNorm:   outputNorm,
		OutputWeight: outputWeight,
		KVCache:      kvCache,
		X:            make([]float32, cfg.HiddenDim),
		Scratch:      NewLayerScratch(cfg),
		Tokenizer:    tokenizer,
	}, nil
}

// ForwardToken processes a single token at position pos through all transformer layers
// and returns the sampled / argmax next token ID.
func (e *Engine) ForwardToken(ctx context.Context, tokenID, pos int) (int, error) {
	return e.forwardToken(ctx, tokenID, pos, true)
}

// forwardToken advances the layer state and KV cache, optionally projecting the next token.
func (e *Engine) forwardToken(ctx context.Context, tokenID, pos int, projectOutput bool) (int, error) {
	if ctx == nil {
		return 0, fmt.Errorf("nil context")
	}
	if err := ctx.Err(); err != nil {
		return 0, err
	}
	if pos < 0 || pos >= e.KVCache.MaxPos {
		return 0, fmt.Errorf("position %d outside KV cache capacity %d", pos, e.KVCache.MaxPos)
	}

	// 1. Read token embedding directly into X.
	if err := ReadEmbedding(ctx, e.Reader, e.TokenEmbd, tokenID, e.X); err != nil {
		return 0, fmt.Errorf("embedding lookup: %w", err)
	}

	opts := Q40Options{
		Workers:     e.Options.Workers,
		WindowBytes: e.Options.WindowBytes,
	}

	// 2. Stream through all transformer layers.
	for l := 0; l < len(e.Layers); l++ {
		if err := ForwardLayer(ctx, e.Reader, l, &e.Layers[l], e.X, e.KVCache, pos, e.Config, e.Scratch, opts); err != nil {
			return 0, fmt.Errorf("layer %d failed: %w", l, err)
		}
	}
	if !projectOutput {
		return 0, nil
	}
	return e.projectOutput(ctx, opts)
}

func (e *Engine) projectOutput(ctx context.Context, opts Q40Options) (int, error) {
	// 3. Final layer norm.
	if err := RMSNormInto(ctx, e.Reader, e.OutputNorm, e.X, e.Scratch.NormedX, e.Config.RMSNormEps); err != nil {
		return 0, fmt.Errorf("final norm: %w", err)
	}

	// 4. LM head projection & Argmax.
	bestToken, _, err := MulQ80Argmax(ctx, e.Reader, e.OutputWeight, e.Scratch.NormedX, opts)
	if err != nil {
		return 0, fmt.Errorf("output argmax: %w", err)
	}

	return bestToken, nil
}

// IsEOS checks if tokenID is considered an End-Of-Sequence (EOS) or stop token.
func (e *Engine) IsEOS(tokenID int) bool {
	if e == nil {
		return false
	}
	if e.Config != nil && e.Config.IsEOS(tokenID) {
		return true
	}
	if e.Tokenizer != nil && tokenID >= 0 && tokenID < len(e.Tokenizer.Tokens) {
		switch e.Tokenizer.Tokens[tokenID] {
		case "<|endoftext|>", "<|end|>", "<|return|>", "<|fim_suffix|>", "<|call|>", "<|ghissue|>", "<|eot_id|>", "<|im_end|>", "</s>", "<eos>", "<|end_of_text|>", "<|eom_id|>":
			return true
		}
	}
	return false
}

// Generate runs autoregressive generation given promptTokens, invoking onToken for each new token.
func (e *Engine) Generate(
	ctx context.Context,
	promptTokens []int,
	maxNewTokens int,
	onToken func(tokenID int, text string) bool,
) ([]int, error) {
	if len(promptTokens) == 0 {
		return nil, fmt.Errorf("empty prompt tokens")
	}
	if maxNewTokens <= 0 {
		maxNewTokens = 64
	}

	var generated []int
	var nextToken int
	var err error

	nextToken, err = e.Prefill(ctx, promptTokens, 0)
	if err != nil {
		return nil, err
	}

	pos := len(promptTokens)
	for i := 0; i < maxNewTokens; i++ {
		if e.IsEOS(nextToken) {
			break
		}
		generated = append(generated, nextToken)
		tokenText := ""
		if e.Tokenizer != nil {
			tokenText = e.Tokenizer.Decode([]int{nextToken})
		}
		if onToken != nil {
			if !onToken(nextToken, tokenText) {
				break
			}
		}
		if i+1 == maxNewTokens {
			break
		}
		nextToken, err = e.ForwardToken(ctx, nextToken, pos)
		if err != nil {
			return generated, fmt.Errorf("generation step %d (token %d): %w", i, nextToken, err)
		}
		pos++
	}

	return generated, nil
}

// Tokenizer decodes and encodes tokens from GGUF metadata using byte-level BPE.
type Tokenizer struct {
	Tokens        []string
	TokenMap      map[string]int
	byteToUnicode [256]rune
	unicodeToByte map[rune]byte
	maxTokenLen   int
	encoder       tokenizerEncoder
}

func initByteToUnicode() ([256]rune, map[rune]byte) {
	var b2u [256]rune
	u2b := make(map[rune]byte, 256)

	seen := make(map[byte]bool)
	var bs []int
	for b := int('!'); b <= int('~'); b++ {
		bs = append(bs, b)
	}
	for b := int('¡'); b <= int('¬'); b++ {
		bs = append(bs, b)
	}
	for b := int('®'); b <= int('ÿ'); b++ {
		bs = append(bs, b)
	}
	for _, b := range bs {
		b2u[byte(b)] = rune(b)
		u2b[rune(b)] = byte(b)
		seen[byte(b)] = true
	}
	n := 0
	for b := 0; b < 256; b++ {
		if !seen[byte(b)] {
			r := rune(256 + n)
			b2u[byte(b)] = r
			u2b[r] = byte(b)
			n++
		}
	}
	return b2u, u2b
}

// LoadTokenizer reads the token vocabulary from model metadata.
func LoadTokenizer(model *ggufindex.Model) (*Tokenizer, error) {
	meta, ok := model.Metadata["tokenizer.ggml.tokens"]
	if !ok {
		return nil, fmt.Errorf("missing tokenizer.ggml.tokens metadata")
	}

	f, err := os.Open(meta.Range.File)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	if _, err := f.Seek(int64(meta.Range.Start), io.SeekStart); err != nil {
		return nil, err
	}
	r := bufio.NewReader(f)

	var elemType uint32
	var count uint64
	if err := binary.Read(r, binary.LittleEndian, &elemType); err != nil {
		return nil, err
	}
	if err := binary.Read(r, binary.LittleEndian, &count); err != nil {
		return nil, err
	}
	if elemType != 8 || count > 1000000 {
		return nil, fmt.Errorf("unexpected token array type %d count %d", elemType, count)
	}

	tokens := make([]string, int(count))
	tokenMap := make(map[string]int, int(count))
	b2u, u2b := initByteToUnicode()
	maxTokenLen := 1

	for i := 0; i < int(count); i++ {
		var strLen uint64
		if err := binary.Read(r, binary.LittleEndian, &strLen); err != nil {
			return nil, err
		}
		buf := make([]byte, int(strLen))
		if _, err := io.ReadFull(r, buf); err != nil {
			return nil, err
		}
		str := string(buf)
		tokens[i] = str
		tokenMap[str] = i
		rCount := utf8.RuneCountInString(str)
		if rCount > maxTokenLen {
			maxTokenLen = rCount
		}
	}

	tokenizer := &Tokenizer{
		Tokens:        tokens,
		TokenMap:      tokenMap,
		byteToUnicode: b2u,
		unicodeToByte: u2b,
		maxTokenLen:   maxTokenLen,
	}
	tokenizer.initEncoder()
	return tokenizer, nil
}

// Decode converts token IDs into human-readable text.
func (t *Tokenizer) Decode(tokens []int) string {
	if t == nil {
		return ""
	}
	var rawBytes []byte
	for _, tok := range tokens {
		if tok >= 0 && tok < len(t.Tokens) {
			s := t.Tokens[tok]
			for _, r := range s {
				if b, ok := t.unicodeToByte[r]; ok {
					rawBytes = append(rawBytes, b)
				} else {
					rawBytes = append(rawBytes, []byte(string(r))...)
				}
			}
		}
	}
	return string(rawBytes)
}

// Encode converts input text into token IDs using greedy longest-prefix matching on byte-level BPE tokens.
func (t *Tokenizer) Encode(text string) []int {
	if t == nil || len(text) == 0 {
		return nil
	}
	t.initEncoder()

	var tokenIDs []int
	for i := 0; i < len(text); {
		node := t.encoder.root[text[i]]
		matchedEnd, matchedID := i, 0
		for j := i; j < len(text); j++ {
			if node == 0 {
				break
			}
			if n := t.encoder.nodes[node]; n.terminal {
				matchedEnd, matchedID = j+1, n.id
			}
			if j+1 < len(text) {
				node = t.encoder.next(node, text[j+1])
			}
		}
		if matchedEnd > i {
			tokenIDs = append(tokenIDs, matchedID)
			i = matchedEnd
		} else {
			i++
		}
	}
	return tokenIDs
}
