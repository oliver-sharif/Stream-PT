//go:build goexperiment.simd

package main

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"Stream-PT/forward"
	"Stream-PT/ggufindex"
	ggufmmap "Stream-PT/ggufmap"
)

func TestCLIThreadDefaults(t *testing.T) {
	cmd := exec.Command("go", "run", ".", "-h")
	for _, value := range os.Environ() {
		if !strings.HasPrefix(value, "GOMAXPROCS=") {
			cmd.Env = append(cmd.Env, value)
		}
	}
	cmd.Env = append(cmd.Env, "GOMAXPROCS=8")
	output, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("CLI help: %v\n%s", err, output)
	}
	for _, flag := range []string{"-threads int", "-t int"} {
		_, description, ok := strings.Cut(string(output), flag+"\n")
		if !ok || !strings.HasPrefix(strings.TrimSpace(description), "Number of parallel worker goroutines") ||
			!strings.Contains(strings.SplitN(description, "\n", 2)[0], "(default 8)") {
			t.Fatalf("%s must preserve GOMAXPROCS=8:\n%s", flag, output)
		}
	}
}

func TestChatServerIndex(t *testing.T) {
	srv := &ChatServer{DefaultMaxTokens: 10}
	ts := httptest.NewServer(srv.routes())
	defer ts.Close()

	resp, err := http.Get(ts.URL + "/")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want %d", resp.StatusCode, http.StatusOK)
	}

	contentType := resp.Header.Get("Content-Type")
	if !strings.Contains(contentType, "text/html") {
		t.Fatalf("Content-Type = %s, want text/html", contentType)
	}

	buf := new(bytes.Buffer)
	if _, err := buf.ReadFrom(resp.Body); err != nil {
		t.Fatal(err)
	}
	body := buf.String()
	for _, expected := range []string{"Stream-PT", "/api/chat", "/api/info", "max-tokens-slider", "system-prompt-input", "settings-modal"} {
		if !strings.Contains(body, expected) {
			t.Errorf("Index HTML missing %q", expected)
		}
	}
}

func TestChatServerInfo(t *testing.T) {
	srv := &ChatServer{
		DefaultMaxTokens: 128,
		ModelInfo: ServerInfo{
			ModelName:        "gpt-oss-test",
			LayerCount:       16,
			Workers:          4,
			WindowMB:         32,
			DefaultMaxTokens: 128,
			CPU:              "Test-CPU",
			RAM:              "16.0 GiB",
		},
	}
	ts := httptest.NewServer(srv.routes())
	defer ts.Close()

	t.Run("GET", func(t *testing.T) {
		resp, err := http.Get(ts.URL + "/api/info")
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("status = %d, want %d", resp.StatusCode, http.StatusOK)
		}
		var info ServerInfo
		if err := json.NewDecoder(resp.Body).Decode(&info); err != nil {
			t.Fatalf("decode: %v", err)
		}
		if info.ModelName != "gpt-oss-test" || info.DefaultMaxTokens != 128 || info.Workers != 4 {
			t.Fatalf("unexpected info response: %+v", info)
		}
	})

	t.Run("MethodNotAllowed", func(t *testing.T) {
		resp, err := http.Post(ts.URL+"/api/info", "application/json", strings.NewReader("{}"))
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusMethodNotAllowed {
			t.Fatalf("status = %d, want %d", resp.StatusCode, http.StatusMethodNotAllowed)
		}
	})
}

func TestChatServerValidation(t *testing.T) {
	srv := &ChatServer{DefaultMaxTokens: 10}
	handler := srv.routes()

	t.Run("MethodNotAllowed", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodGet, "/api/chat", nil)
		rec := httptest.NewRecorder()
		handler.ServeHTTP(rec, req)
		if rec.Code != http.StatusMethodNotAllowed {
			t.Fatalf("status = %d, want %d", rec.Code, http.StatusMethodNotAllowed)
		}
	})

	t.Run("InvalidJSON", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader("invalid json"))
		rec := httptest.NewRecorder()
		handler.ServeHTTP(rec, req)
		if rec.Code != http.StatusBadRequest {
			t.Fatalf("status = %d, want %d", rec.Code, http.StatusBadRequest)
		}
	})

	t.Run("EmptyMessages", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(`{"messages":[]}`))
		rec := httptest.NewRecorder()
		handler.ServeHTTP(rec, req)
		if rec.Code != http.StatusBadRequest {
			t.Fatalf("status = %d, want %d", rec.Code, http.StatusBadRequest)
		}
	})
}

func TestChatServerSSEStreaming(t *testing.T) {
	const dim, vocab = 32, 12
	path := filepath.Join(t.TempDir(), "server_gen.bin")
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
	// Ensure token 1 ("Ja") has highest logit after prompt
	weights[1*34+2+6] = 10
	output := add(8, []uint64{dim, vocab}, weights)
	if err := os.WriteFile(path, data, 0600); err != nil {
		t.Fatal(err)
	}
	reader, err := ggufmmap.Open(&ggufindex.Model{Paths: []string{path}})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = reader.Close() })

	engine := &forward.Engine{
		Reader: reader, Config: &forward.Config{EOS: 99, RMSNormEps: 1e-5, HiddenDim: dim, NumHeads: 1, NumKVHeads: 1, HeadDim: 2},
		TokenEmbd: embd, OutputNorm: norm, OutputWeight: output,
		Options: forward.EngineOptions{PrefillBatchSize: 1},
		KVCache: forward.NewKVCache(0, 20, 1, 2), X: make([]float32, dim),
		Scratch: &forward.LayerScratch{NormedX: make([]float32, dim)},
		Tokenizer: &forward.Tokenizer{
			Tokens: []string{"prompt", "Ja", "!", "test", "It", "<|start|>", "<|message|>", "<|end|>", "<|channel|>", "user", "assistant", "final"},
			TokenMap: map[string]int{
				"prompt": 0, "Ja": 1, "!": 2, "test": 3, "It": 4,
				"<|start|>": 5, "<|message|>": 6, "<|end|>": 7, "<|channel|>": 8,
				"user": 9, "assistant": 10, "final": 11,
			},
		},
	}

	srv := &ChatServer{Engine: engine, DefaultMaxTokens: 5}
	ts := httptest.NewServer(srv.routes())
	defer ts.Close()

	t.Run("DefaultStreaming", func(t *testing.T) {
		reqBody := `{"messages":[{"role":"user","content":"Ja"}]}`
		resp, err := http.Post(ts.URL+"/api/chat", "application/json", strings.NewReader(reqBody))
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			t.Fatalf("status = %d, want %d", resp.StatusCode, http.StatusOK)
		}
		if !strings.Contains(resp.Header.Get("Content-Type"), "text/event-stream") {
			t.Fatalf("Content-Type = %s, want text/event-stream", resp.Header.Get("Content-Type"))
		}

		buf := new(bytes.Buffer)
		if _, err := buf.ReadFrom(resp.Body); err != nil {
			t.Fatal(err)
		}
		sseOutput := buf.String()

		lines := strings.Split(sseOutput, "\n")
		var receivedDone bool
		var events []ChatEvent
		for _, line := range lines {
			trimmed := strings.TrimSpace(line)
			if strings.HasPrefix(trimmed, "data:") {
				dataJSON := strings.TrimSpace(strings.TrimPrefix(trimmed, "data:"))
				var ev ChatEvent
				if err := json.Unmarshal([]byte(dataJSON), &ev); err != nil {
					t.Fatalf("failed to parse SSE data %q: %v", dataJSON, err)
				}
				events = append(events, ev)
				if ev.Done {
					receivedDone = true
				}
			}
		}

		if !receivedDone {
			t.Fatalf("expected done event in SSE stream, got: %s", sseOutput)
		}
		if len(events) < 2 {
			t.Fatalf("expected at least 2 SSE events, got: %v", events)
		}
	})

	t.Run("CustomMaxTokensAndSystemPrompt", func(t *testing.T) {
		reqBody := `{"system_prompt":"You are helper","messages":[{"role":"user","content":"Ja"}],"max_tokens":2}`
		resp, err := http.Post(ts.URL+"/api/chat", "application/json", strings.NewReader(reqBody))
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			t.Fatalf("status = %d, want %d", resp.StatusCode, http.StatusOK)
		}

		buf := new(bytes.Buffer)
		if _, err := buf.ReadFrom(resp.Body); err != nil {
			t.Fatal(err)
		}
		sseOutput := buf.String()

		lines := strings.Split(sseOutput, "\n")
		var tokenEvents int
		var receivedDone bool
		for _, line := range lines {
			trimmed := strings.TrimSpace(line)
			if strings.HasPrefix(trimmed, "data:") {
				dataJSON := strings.TrimSpace(strings.TrimPrefix(trimmed, "data:"))
				var ev ChatEvent
				if err := json.Unmarshal([]byte(dataJSON), &ev); err != nil {
					t.Fatalf("failed to parse SSE data %q: %v", dataJSON, err)
				}
				if ev.Token > 0 || ev.Text != "" {
					tokenEvents++
				}
				if ev.Done {
					receivedDone = true
				}
			}
		}
		if !receivedDone {
			t.Fatalf("expected done event")
		}
		if tokenEvents > 2 {
			t.Fatalf("expected at most 2 token events for max_tokens:2, got %d", tokenEvents)
		}
	})
}
