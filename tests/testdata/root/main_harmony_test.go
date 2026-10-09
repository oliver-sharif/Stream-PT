//go:build goexperiment.simd

package main

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"Stream-PT/forward"
	"Stream-PT/ggufindex"
	ggufmmap "Stream-PT/ggufmap"
)

func TestChatHarmonySSEContinuation(t *testing.T) {
	const dim = 32
	tokens := []string{"prompt", "<|channel|>", "analysis", "<|message|>", "hidden reasoning", "<|end|>", "<|start|>", "assist", "ant", "<|meta_sep|>", "assistant", "final", "<|im_sep|>", "Yes!", "<|return|>", "<|call|>"}
	path := filepath.Join(t.TempDir(), "harmony.bin")
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
	sequence := []int{10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14}
	for i, id := range sequence[1:] {
		binary.LittleEndian.PutUint16(weights[id*34:], 0x3c00)
		weights[id*34+2+sequence[i]] = 1
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
	engine := &forward.Engine{
		Reader: reader, Config: &forward.Config{EOS: 5, EOSTokens: []int{5, 14, 15}, RMSNormEps: 1e-5},
		TokenEmbd: embd, OutputNorm: norm, OutputWeight: output,
		Options: forward.EngineOptions{PrefillBatchSize: 1},
		KVCache: forward.NewKVCache(0, 256, 1, 2), X: make([]float32, dim),
		Scratch:   &forward.LayerScratch{NormedX: make([]float32, dim)},
		Tokenizer: &forward.Tokenizer{Tokens: tokens, TokenMap: tokenMap},
	}
	srv := &ChatServer{Engine: engine, DefaultMaxTokens: 32}
	for _, stop := range []string{"<|return|>", "<|call|>"} {
		t.Run(stop, func(t *testing.T) {
			// The same numerical model ends either with completion or handoff.
			engine.Tokenizer.Tokens[14] = stop
			req := httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(`{"prompt":"prompt","temperature":0}`))
			rec := httptest.NewRecorder()
			srv.routes().ServeHTTP(rec, req)
			var text strings.Builder
			var reasoning strings.Builder
			var done ChatEvent
			for _, line := range strings.Split(rec.Body.String(), "\n") {
				if payload, ok := strings.CutPrefix(line, "data: "); ok {
					var event ChatEvent
					if err := json.Unmarshal([]byte(payload), &event); err != nil {
						t.Fatal(err)
					}
					if event.Error != "" {
						t.Fatal(event.Error)
					}
					text.WriteString(event.Text)
					var fields map[string]any
					if err := json.Unmarshal([]byte(payload), &fields); err != nil {
						t.Fatal(err)
					}
					if chunk, ok := fields["reasoning"].(string); ok {
						reasoning.WriteString(chunk)
					}
					if !event.Done && event.GeneratedTokens == 0 {
						t.Fatal("missing live token progress")
					}
					if event.Done {
						done = event
					}
				}
			}
			if reasoning.String() != "hidden reasoning" {
				t.Fatalf("reasoning %q, want live analysis; SSE: %s", reasoning.String(), rec.Body.String())
			}
			wantReason := "stop"
			if stop == "<|call|>" {
				wantReason = "tool_calls"
			}
			if text.String() != "Yes!" || !done.Done || done.FinishReason != wantReason || done.StopToken == nil || *done.StopToken != 14 || done.GeneratedTokens != 12 {
				t.Fatalf("text %q, completion %+v; SSE: %s", text.String(), done, rec.Body.String())
			}
			if done.Performance == nil || done.Performance.PrefillSeconds <= 0 || done.Performance.DecodeSteps != 12 || done.Performance.DecodeTokensPerSec <= 0 || done.Performance.FirstVisibleSeconds <= 0 {
				t.Fatalf("missing inference diagnostics: %+v", done.Performance)
			}
		})
	}
	t.Run("session-prefix", func(t *testing.T) {
		for iteration := range 2 {
			req := httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(`{"session_id":"same-browser","prompt":"prompt","temperature":0,"max_tokens":4}`))
			rec := httptest.NewRecorder()
			srv.routes().ServeHTTP(rec, req)
			var done ChatEvent
			for line := range strings.SplitSeq(rec.Body.String(), "\n") {
				if payload, ok := strings.CutPrefix(line, "data: "); ok {
					var event ChatEvent
					if err := json.Unmarshal([]byte(payload), &event); err != nil {
						t.Fatal(err)
					}
					if event.Error != "" {
						t.Fatal(event.Error)
					}
					if event.Done {
						done = event
					}
				}
			}
			if done.Performance == nil || !done.Done {
				t.Fatalf("missing performance: %+v", done)
			}
			wantReuse := 0
			if iteration == 1 {
				wantReuse = done.Performance.PromptTokens - 1
			}
			if done.Performance.ReusedPromptTokens != wantReuse {
				t.Fatalf("iteration %d: %+v", iteration, done.Performance)
			}
		}
	})
	t.Run("budget-during-reasoning", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(`{"prompt":"prompt","temperature":0,"max_tokens":4}`))
		rec := httptest.NewRecorder()
		srv.routes().ServeHTTP(rec, req)
		var reasoning strings.Builder
		var done ChatEvent
		for _, line := range strings.Split(rec.Body.String(), "\n") {
			if payload, ok := strings.CutPrefix(line, "data: "); ok {
				var event ChatEvent
				if err := json.Unmarshal([]byte(payload), &event); err != nil {
					t.Fatal(err)
				}
				if event.Error != "" || event.Text != "" {
					t.Fatalf("unexpected event %+v", event)
				}
				reasoning.WriteString(event.Reasoning)
				if event.Done {
					done = event
				}
			}
		}
		if reasoning.String() != "hidden reasoning" || !done.Done || done.FinishReason != "length" || done.GeneratedTokens != 4 {
			t.Fatalf("reasoning %q, completion %+v", reasoning.String(), done)
		}
	})
}
