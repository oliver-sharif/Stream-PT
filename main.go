//go:build goexperiment.simd

package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"Stream-PT/forward"
	"Stream-PT/ggufindex"
	ggufmmap "Stream-PT/ggufmap"
)

type ServerInfo struct {
	ModelName        string                  `json:"model_name"`
	LayerCount       int                     `json:"layer_count"`
	Workers          int                     `json:"workers"`
	WindowMB         int                     `json:"window_mb"`
	DefaultMaxTokens int                     `json:"default_max_tokens"`
	DefaultSampling  forward.SamplingOptions `json:"default_sampling"`
	CPU              string                  `json:"cpu"`
	RAM              string                  `json:"ram"`
}

type ChatServer struct {
	Engine           *forward.Engine
	DefaultMaxTokens int
	ModelInfo        ServerInfo
	mu               sync.Mutex
}

type ChatRequest struct {
	Messages         []forward.ChatMessage `json:"messages"`
	Prompt           string                `json:"prompt"`
	SystemPrompt     string                `json:"system_prompt,omitempty"`
	MaxTokens        int                   `json:"max_tokens"`
	Temperature      *float64              `json:"temperature"`
	TopK             *int                  `json:"top_k"`
	TopP             *float64              `json:"top_p"`
	MinP             *float64              `json:"min_p"`
	RepeatPenalty    *float64              `json:"repeat_penalty"`
	RepeatLastN      *int                  `json:"repeat_last_n"`
	FrequencyPenalty *float64              `json:"frequency_penalty"`
	PresencePenalty  *float64              `json:"presence_penalty"`
	Seed             *uint64               `json:"seed"`
}

func (s *ChatServer) samplingDefaults() forward.SamplingOptions {
	if s.Engine != nil && s.Engine.Options.Sampling != nil {
		return *s.Engine.Options.Sampling
	}
	return forward.DefaultSamplingOptions()
}

func (r ChatRequest) samplingOptions(options forward.SamplingOptions) forward.SamplingOptions {
	if r.Temperature != nil {
		options.Temperature = *r.Temperature
	}
	if r.TopK != nil {
		options.TopK = *r.TopK
	}
	if r.TopP != nil {
		options.TopP = *r.TopP
	}
	if r.MinP != nil {
		options.MinP = *r.MinP
	}
	if r.RepeatPenalty != nil {
		options.RepeatPenalty = *r.RepeatPenalty
	}
	if r.RepeatLastN != nil {
		options.RepeatLastN = *r.RepeatLastN
	}
	if r.FrequencyPenalty != nil {
		options.FrequencyPenalty = *r.FrequencyPenalty
	}
	if r.PresencePenalty != nil {
		options.PresencePenalty = *r.PresencePenalty
	}
	if r.Seed != nil {
		options.Seed = r.Seed
	}
	return options
}

type ChatEvent struct {
	Token int    `json:"token,omitempty"`
	Text  string `json:"text,omitempty"`
	Done  bool   `json:"done,omitempty"`
	Error string `json:"error,omitempty"`
}

func (s *ChatServer) routes() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/", s.handleIndex)
	mux.HandleFunc("/api/info", s.handleInfo)
	mux.HandleFunc("/api/chat", s.handleChat)
	return mux
}

func (s *ChatServer) handleIndex(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Write([]byte(indexHTML))
}

func (s *ChatServer) handleInfo(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	info := s.ModelInfo
	info.DefaultSampling = s.samplingDefaults()
	if info.DefaultMaxTokens == 0 {
		info.DefaultMaxTokens = s.DefaultMaxTokens
	}
	if info.DefaultMaxTokens <= 0 {
		info.DefaultMaxTokens = 512
	}
	if s.Engine != nil {
		if s.Engine.Config != nil && info.LayerCount == 0 {
			info.LayerCount = s.Engine.Config.NumLayers
		}
		if info.Workers == 0 {
			info.Workers = s.Engine.Options.Workers
		}
		if info.WindowMB == 0 && s.Engine.Options.WindowBytes > 0 {
			info.WindowMB = int(s.Engine.Options.WindowBytes / (1024 * 1024))
		}
	}
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	_ = json.NewEncoder(w).Encode(info)
}

func (s *ChatServer) handleChat(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "Streaming not supported", http.StatusInternalServerError)
		return
	}

	var req ChatRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("Invalid JSON request: %v", err), http.StatusBadRequest)
		return
	}
	sampling := req.samplingOptions(s.samplingDefaults())
	if err := sampling.Validate(); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	messages := req.Messages
	if len(messages) == 0 && req.Prompt != "" {
		messages = []forward.ChatMessage{{Role: "user", Content: req.Prompt}}
	}
	if req.SystemPrompt != "" {
		if len(messages) == 0 || messages[0].Role != "system" {
			messages = append([]forward.ChatMessage{{Role: "system", Content: req.SystemPrompt}}, messages...)
		}
	}
	if len(messages) == 0 {
		http.Error(w, "Empty messages or prompt", http.StatusBadRequest)
		return
	}

	maxTokens := req.MaxTokens
	if maxTokens <= 0 {
		maxTokens = s.DefaultMaxTokens
	}
	if maxTokens <= 0 {
		maxTokens = 512
	}

	w.Header().Set("Content-Type", "text/event-stream; charset=utf-8")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")

	sendSSE := func(event ChatEvent) error {
		data, err := json.Marshal(event)
		if err != nil {
			return err
		}
		if _, err := fmt.Fprintf(w, "data: %s\n\n", data); err != nil {
			return err
		}
		flusher.Flush()
		return nil
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	var promptTokens []int
	var err error
	if s.Engine != nil && s.Engine.Tokenizer != nil {
		promptTokens, err = s.Engine.Tokenizer.EncodeChatMessages(messages)
	} else {
		err = fmt.Errorf("tokenizer not initialized")
	}

	if err != nil {
		_ = sendSSE(ChatEvent{Error: err.Error(), Done: true})
		return
	}

	ctx := r.Context()
	_, err = s.Engine.GenerateWithSampling(ctx, promptTokens, maxTokens, sampling, func(tokenID int, text string) bool {
		if ctx.Err() != nil {
			return false
		}
		if err := sendSSE(ChatEvent{Token: tokenID, Text: text}); err != nil {
			return false
		}
		return true
	})

	if err != nil && ctx.Err() == nil {
		_ = sendSSE(ChatEvent{Error: err.Error(), Done: true})
		return
	}

	_ = sendSSE(ChatEvent{Done: true})
}

const indexHTML = `<!DOCTYPE html>
<html lang="de">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Stream-PT Chat</title>
  <style>
    :root {
      --bg-color: #0d1117;
      --card-bg: #161b22;
      --border-color: #30363d;
      --text-main: #f0f6fc;
      --text-muted: #8b949e;
      --user-bg: #1f6feb;
      --assistant-bg: #21262d;
      --accent: #238636;
      --accent-hover: #2ea043;
      --danger: #da3633;
      --danger-hover: #b62324;
    }
    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background-color: var(--bg-color);
      color: var(--text-main);
      display: flex;
      flex-direction: column;
      height: 100vh;
      overflow: hidden;
    }
    header {
      background-color: var(--card-bg);
      border-bottom: 1px solid var(--border-color);
      padding: 12px 20px;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    .header-title {
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .header-title h1 {
      font-size: 1.15rem;
      font-weight: 600;
    }
    .badge {
      font-size: 0.75rem;
      background-color: #238636;
      color: #fff;
      padding: 2px 8px;
      border-radius: 12px;
      font-weight: 500;
    }
    .header-actions {
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .status-text {
      font-size: 0.85rem;
      color: var(--text-muted);
    }
    button {
      cursor: pointer;
      font-family: inherit;
      font-size: 0.9rem;
      border-radius: 6px;
      border: 1px solid transparent;
      padding: 6px 14px;
      transition: background-color 0.15s ease, opacity 0.15s ease;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 6px;
    }
    .btn-secondary {
      background-color: #21262d;
      color: var(--text-main);
      border-color: var(--border-color);
    }
    .btn-secondary:hover {
      background-color: #30363d;
    }
    .btn-primary {
      background-color: var(--accent);
      color: #ffffff;
      font-weight: 600;
    }
    .btn-primary:hover:not(:disabled) {
      background-color: var(--accent-hover);
    }
    .btn-primary:disabled {
      opacity: 0.5;
      cursor: not-allowed;
    }
    .btn-danger {
      background-color: var(--danger);
      color: #ffffff;
    }
    .btn-danger:hover {
      background-color: var(--danger-hover);
    }
    .btn-icon {
      background: none;
      border: none;
      font-size: 1.25rem;
      color: var(--text-muted);
      cursor: pointer;
      padding: 4px 8px;
      border-radius: 4px;
    }
    .btn-icon:hover {
      color: var(--text-main);
      background-color: #30363d;
    }
    main {
      flex: 1;
      display: flex;
      flex-direction: column;
      max-width: 900px;
      width: 100%;
      margin: 0 auto;
      padding: 16px;
      overflow: hidden;
      gap: 12px;
    }
    #chat-container {
      flex: 1;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 14px;
      padding-right: 6px;
    }
    #chat-container::-webkit-scrollbar {
      width: 6px;
    }
    #chat-container::-webkit-scrollbar-thumb {
      background-color: var(--border-color);
      border-radius: 3px;
    }
    .message {
      display: flex;
      width: 100%;
      flex-direction: column;
    }
    .message.user {
      align-items: flex-end;
    }
    .message.assistant {
      align-items: flex-start;
    }
    .message-role {
      font-size: 0.75rem;
      color: var(--text-muted);
      margin-bottom: 4px;
      padding: 0 4px;
    }
    .bubble {
      max-width: 80%;
      padding: 10px 14px;
      border-radius: 10px;
      font-size: 0.95rem;
      line-height: 1.5;
      white-space: pre-wrap;
      word-break: break-word;
    }
    .message.user .bubble {
      background-color: var(--user-bg);
      color: #ffffff;
      border-bottom-right-radius: 2px;
    }
    .message.assistant .bubble {
      background-color: var(--assistant-bg);
      border: 1px solid var(--border-color);
      border-bottom-left-radius: 2px;
    }
    .cursor {
      display: inline-block;
      width: 8px;
      height: 15px;
      background-color: var(--text-main);
      margin-left: 2px;
      vertical-align: middle;
      animation: blink 0.8s infinite;
    }
    @keyframes blink {
      0%, 50% { opacity: 1; }
      51%, 100% { opacity: 0; }
    }
    .input-section {
      background-color: var(--card-bg);
      border: 1px solid var(--border-color);
      border-radius: 8px;
      padding: 10px 12px;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    .input-row {
      display: flex;
      gap: 8px;
      align-items: flex-end;
    }
    textarea {
      flex: 1;
      background: transparent;
      border: none;
      color: var(--text-main);
      font-family: inherit;
      font-size: 0.95rem;
      resize: none;
      outline: none;
      max-height: 150px;
      min-height: 24px;
      line-height: 1.4;
    }
    textarea::placeholder {
      color: var(--text-muted);
    }
    .input-footer {
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 0.8rem;
      color: var(--text-muted);
      gap: 10px;
      flex-wrap: wrap;
    }
    .input-footer-left {
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .settings-pill {
      background-color: var(--assistant-bg);
      border: 1px solid var(--border-color);
      color: var(--text-main);
      padding: 2px 8px;
      border-radius: 12px;
      cursor: pointer;
      font-size: 0.75rem;
      display: inline-flex;
      align-items: center;
      gap: 4px;
      transition: border-color 0.15s ease;
    }
    .settings-pill:hover {
      border-color: var(--text-muted);
    }

    /* Modal / Drawer */
    .modal-backdrop {
      position: fixed;
      top: 0;
      left: 0;
      width: 100vw;
      height: 100vh;
      background: rgba(0, 0, 0, 0.7);
      backdrop-filter: blur(4px);
      display: flex;
      align-items: center;
      justify-content: center;
      z-index: 1000;
    }
    .modal-card {
      background-color: var(--card-bg);
      border: 1px solid var(--border-color);
      border-radius: 12px;
      width: 90%;
      max-width: 520px;
      max-height: 90vh;
      display: flex;
      flex-direction: column;
      box-shadow: 0 12px 32px rgba(0, 0, 0, 0.6);
      overflow: hidden;
    }
    .modal-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 14px 18px;
      border-bottom: 1px solid var(--border-color);
    }
    .modal-header h2 {
      font-size: 1.1rem;
      font-weight: 600;
    }
    .modal-body {
      padding: 18px;
      display: flex;
      flex-direction: column;
      gap: 18px;
      overflow-y: auto;
    }
    .setting-group {
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    .setting-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 0.9rem;
      font-weight: 600;
    }
    .setting-value {
      color: #58a6ff;
      font-family: monospace;
      font-size: 0.95rem;
    }
    .slider-row {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    input[type="range"] {
      flex: 1;
      accent-color: var(--accent);
      cursor: pointer;
    }
    input[type="number"] {
      width: 80px;
      background-color: var(--bg-color);
      border: 1px solid var(--border-color);
      border-radius: 6px;
      color: var(--text-main);
      padding: 5px 8px;
      font-family: inherit;
      font-size: 0.9rem;
      outline: none;
    }
    input[type="number"]:focus {
      border-color: #58a6ff;
    }
    .preset-buttons {
      display: flex;
      gap: 6px;
      flex-wrap: wrap;
    }
    .preset-btn {
      padding: 3px 8px;
      font-size: 0.75rem;
      background-color: var(--assistant-bg);
      border: 1px solid var(--border-color);
      color: var(--text-muted);
      border-radius: 4px;
    }
    .preset-btn:hover {
      background-color: #30363d;
      color: var(--text-main);
    }
    .preset-btn.active {
      background-color: var(--accent);
      color: #fff;
      border-color: var(--accent);
    }
    .setting-help {
      font-size: 0.75rem;
      color: var(--text-muted);
      line-height: 1.3;
    }
    .form-textarea {
      background-color: var(--bg-color);
      border: 1px solid var(--border-color);
      border-radius: 6px;
      color: var(--text-main);
      padding: 8px 10px;
      font-family: inherit;
      font-size: 0.85rem;
      resize: vertical;
      min-height: 60px;
      outline: none;
    }
    .form-textarea:focus {
      border-color: #58a6ff;
    }
    .info-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      font-size: 0.8rem;
      background-color: var(--assistant-bg);
      border: 1px solid var(--border-color);
      border-radius: 6px;
      padding: 10px 12px;
    }
    .info-item {
      display: flex;
      flex-direction: column;
      gap: 2px;
    }
    .info-label {
      color: var(--text-muted);
      font-size: 0.72rem;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }
    .info-val {
      font-weight: 500;
      color: var(--text-main);
      word-break: break-all;
    }
    .modal-footer {
      display: flex;
      justify-content: space-between;
      padding: 12px 18px;
      border-top: 1px solid var(--border-color);
      background-color: rgba(0, 0, 0, 0.2);
    }
  </style>
</head>
<body>
  <header>
    <div class="header-title">
      <h1>Stream-PT</h1>
      <span class="badge">SSE Live</span>
    </div>
    <div class="header-actions">
      <span id="status-text" class="status-text">Bereit</span>
      <button id="settings-btn" class="btn-secondary" title="Parameter & Einstellungen">⚙ Einstellungen</button>
      <button id="clear-btn" class="btn-secondary" title="Konversation und Kontext leeren">Neuer Chat</button>
    </div>
  </header>

  <main>
    <div id="chat-container">
      <div class="message assistant">
        <div class="message-role">Stream-PT</div>
        <div class="bubble">Hallo! Wie kann ich Ihnen heute helfen?</div>
      </div>
    </div>

    <div class="input-section">
      <div class="input-row">
        <textarea id="prompt-input" rows="1" placeholder="Nachricht eingeben... (Enter zum Senden, Shift+Enter für Zeilenumbruch)"></textarea>
        <button id="stop-btn" class="btn-danger" style="display: none;">Stopp</button>
        <button id="send-btn" class="btn-primary">Senden</button>
      </div>
      <div class="input-footer">
        <div class="input-footer-left">
          <span>Kontext im Browser</span>
          <span id="footer-settings-badge" class="settings-pill" title="Klicken, um Einstellungen anzupassen">⚙ Max Tokens: <span id="footer-max-tokens">512</span></span>
        </div>
        <span id="token-counter"></span>
      </div>
    </div>
  </main>

  <!-- Settings Modal -->
  <div id="settings-modal" class="modal-backdrop" style="display: none;">
    <div class="modal-card">
      <div class="modal-header">
        <h2>⚙ Einstellungen & Parameter</h2>
        <button id="close-settings-btn" class="btn-icon" title="Schließen">&times;</button>
      </div>
      <div class="modal-body">
        <div class="setting-group">
          <div class="setting-header">
            <label for="max-tokens-input">Max New Tokens</label>
            <span class="setting-value" id="max-tokens-display">512</span>
          </div>
          <div class="slider-row">
            <input type="range" id="max-tokens-slider" min="1" max="2048" step="1" value="512">
            <input type="number" id="max-tokens-input" min="1" max="4096" value="512">
          </div>
          <div class="preset-buttons">
            <button type="button" class="preset-btn" data-val="64">64</button>
            <button type="button" class="preset-btn" data-val="128">128</button>
            <button type="button" class="preset-btn" data-val="256">256</button>
            <button type="button" class="preset-btn" data-val="512">512</button>
            <button type="button" class="preset-btn" data-val="1024">1024</button>
            <button type="button" class="preset-btn" data-val="2048">2048</button>
          </div>
          <p class="setting-help">Maximale Anzahl der zu generierenden Tokens pro Antwort.</p>
        </div>

        <div class="setting-group">
          <div class="setting-header">
            <label for="temperature-input">Temperature</label>
            <span class="setting-value" id="temperature-display">0.80</span>
          </div>
          <div class="slider-row">
            <input type="range" id="temperature-slider" min="0" max="2" step="0.01" value="0.8">
            <input type="number" id="temperature-input" min="0" max="2" step="0.01" value="0.8">
          </div>
          <p class="setting-help">0 = deterministische Auswahl mit Wiederholungsstrafe. Höhere Werte erzeugen vielfältigere Antworten; Standard: 0,8.</p>
        </div>

        <div class="setting-group">
          <div class="setting-header">
            <label for="repeat-penalty-input">Wiederholungsstrafe</label>
            <input type="number" id="repeat-penalty-input" min="1" max="2" step="0.01" value="1">
          </div>
          <p class="setting-help">Standard: 1 (deaktiviert, wie llama.cpp). Höhere Werte bestrafen die letzten 64 Kontext- und Antwort-Tokens und können Code und Formatierung beschädigen. Sampling: Top-k 40, Top-p 0,95, Min-p 0,05 (Server-Standardwerte).</p>
        </div>

        <div class="setting-group">
          <div class="setting-header">
            <label for="system-prompt-input">System-Prompt (Rolle / Anweisung)</label>
          </div>
          <textarea id="system-prompt-input" class="form-textarea" rows="2" placeholder="z. B. Du bist ein präziser, deutschsprachiger KI-Assistent..."></textarea>
          <p class="setting-help">Optionaler System-Prompt zur Steuerung des Modellverhaltens.</p>
        </div>

        <div class="setting-group">
          <div class="setting-header">
            <span>System & Modell-Info</span>
          </div>
          <div class="info-grid">
            <div class="info-item"><span class="info-label">Modell</span><span id="info-model" class="info-val">-</span></div>
            <div class="info-item"><span class="info-label">Layer</span><span id="info-layers" class="info-val">-</span></div>
            <div class="info-item"><span class="info-label">Workers (Threads)</span><span id="info-workers" class="info-val">-</span></div>
            <div class="info-item"><span class="info-label">Chunk Window</span><span id="info-window" class="info-val">-</span></div>
            <div class="info-item"><span class="info-label">CPU</span><span id="info-cpu" class="info-val">-</span></div>
            <div class="info-item"><span class="info-label">RAM</span><span id="info-ram" class="info-val">-</span></div>
          </div>
        </div>
      </div>
      <div class="modal-footer">
        <button id="reset-settings-btn" class="btn-secondary">Standard</button>
        <button id="save-settings-btn" class="btn-primary">Fertig</button>
      </div>
    </div>
  </div>

  <script>
    const chatContainer = document.getElementById('chat-container');
    const promptInput = document.getElementById('prompt-input');
    const sendBtn = document.getElementById('send-btn');
    const stopBtn = document.getElementById('stop-btn');
    const clearBtn = document.getElementById('clear-btn');
    const statusText = document.getElementById('status-text');
    const tokenCounter = document.getElementById('token-counter');

    // Settings elements
    const settingsBtn = document.getElementById('settings-btn');
    const footerSettingsBadge = document.getElementById('footer-settings-badge');
    const footerMaxTokens = document.getElementById('footer-max-tokens');
    const settingsModal = document.getElementById('settings-modal');
    const closeSettingsBtn = document.getElementById('close-settings-btn');
    const saveSettingsBtn = document.getElementById('save-settings-btn');
    const resetSettingsBtn = document.getElementById('reset-settings-btn');
    const maxTokensSlider = document.getElementById('max-tokens-slider');
    const maxTokensInput = document.getElementById('max-tokens-input');
    const maxTokensDisplay = document.getElementById('max-tokens-display');
    const systemPromptInput = document.getElementById('system-prompt-input');
    const temperatureSlider = document.getElementById('temperature-slider');
    const temperatureInput = document.getElementById('temperature-input');
    const temperatureDisplay = document.getElementById('temperature-display');
    const repeatPenaltyInput = document.getElementById('repeat-penalty-input');
    const presetButtons = document.querySelectorAll('.preset-btn');

    let serverDefaults = { default_max_tokens: 512, default_sampling: { temperature: 0.8, repeat_penalty: 1 } };
    let settings = {
      version: 2,
      maxTokens: 512,
      temperature: 0.8,
      repeatPenalty: 1,
      systemPrompt: ''
    };

    let messages = [];
    let isGenerating = false;
    let abortController = null;

    function readSavedSettings() {
      try {
        const parsed = JSON.parse(localStorage.getItem('stream_pt_settings') || '{}') || {};
        if (parsed.version !== 2) {
          // Replace only the old implicit defaults, preserving custom values.
          if (parsed.maxTokens === 5) delete parsed.maxTokens;
          if (parsed.repeatPenalty === 1.1) delete parsed.repeatPenalty;
        }
        return parsed;
      } catch (e) {
        return {};
      }
    }

    function loadSettings() {
      try {
        const parsed = readSavedSettings();
        if (parsed) {
          if (parsed.maxTokens && Number.isInteger(parsed.maxTokens) && parsed.maxTokens > 0) {
            settings.maxTokens = parsed.maxTokens;
          }
          if (typeof parsed.systemPrompt === 'string') {
            settings.systemPrompt = parsed.systemPrompt;
          }
          if (Number.isFinite(parsed.temperature) && parsed.temperature >= 0) {
            settings.temperature = parsed.temperature;
          }
          if (Number.isFinite(parsed.repeatPenalty) && parsed.repeatPenalty > 0) {
            settings.repeatPenalty = parsed.repeatPenalty;
          }
        }
      } catch (e) {
        console.error('Failed to load settings:', e);
      }
      applySettingsToUI();
    }

    function saveSettings() {
      try {
        localStorage.setItem('stream_pt_settings', JSON.stringify(settings));
      } catch (e) {
        console.error('Failed to save settings:', e);
      }
      applySettingsToUI();
    }

    function applySettingsToUI() {
      maxTokensSlider.value = settings.maxTokens;
      maxTokensInput.value = settings.maxTokens;
      maxTokensDisplay.textContent = settings.maxTokens;
      footerMaxTokens.textContent = settings.maxTokens;
      systemPromptInput.value = settings.systemPrompt || '';
      temperatureSlider.value = settings.temperature;
      temperatureInput.value = settings.temperature;
      temperatureDisplay.textContent = settings.temperature.toFixed(2);
      repeatPenaltyInput.value = settings.repeatPenalty;

      presetButtons.forEach(btn => {
        if (parseInt(btn.dataset.val, 10) === settings.maxTokens) {
          btn.classList.add('active');
        } else {
          btn.classList.remove('active');
        }
      });
    }

    function setMaxTokens(val) {
      let num = parseInt(val, 10);
      if (isNaN(num) || num < 1) num = 1;
      if (num > 4096) num = 4096;
      settings.maxTokens = num;
      saveSettings();
    }

    maxTokensSlider.addEventListener('input', (e) => setMaxTokens(e.target.value));
    maxTokensInput.addEventListener('change', (e) => setMaxTokens(e.target.value));
    function setTemperature(val) {
      const num = Number(val);
      if (!Number.isFinite(num)) return;
      settings.temperature = Math.max(0, Math.min(2, num));
      saveSettings();
    }
    temperatureSlider.addEventListener('input', (e) => setTemperature(e.target.value));
    temperatureInput.addEventListener('change', (e) => setTemperature(e.target.value));
    repeatPenaltyInput.addEventListener('change', (e) => {
      const num = Number(e.target.value);
      if (!Number.isFinite(num)) return;
      settings.repeatPenalty = Math.max(1, Math.min(2, num));
      saveSettings();
    });
    systemPromptInput.addEventListener('input', (e) => {
      settings.systemPrompt = e.target.value;
      saveSettings();
    });

    presetButtons.forEach(btn => {
      btn.addEventListener('click', () => {
        setMaxTokens(btn.dataset.val);
      });
    });

    function openSettings() {
      settingsModal.style.display = 'flex';
    }

    function closeSettings() {
      settingsModal.style.display = 'none';
    }

    settingsBtn.addEventListener('click', openSettings);
    footerSettingsBadge.addEventListener('click', openSettings);
    closeSettingsBtn.addEventListener('click', closeSettings);
    saveSettingsBtn.addEventListener('click', closeSettings);
    settingsModal.addEventListener('click', (e) => {
      if (e.target === settingsModal) closeSettings();
    });

    resetSettingsBtn.addEventListener('click', () => {
      settings.maxTokens = serverDefaults.default_max_tokens || 512;
      settings.temperature = serverDefaults.default_sampling?.temperature ?? 0.8;
      settings.repeatPenalty = serverDefaults.default_sampling?.repeat_penalty ?? 1;
      settings.systemPrompt = '';
      saveSettings();
    });

    async function fetchServerInfo() {
      try {
        const res = await fetch('/api/info');
        if (res.ok) {
          const info = await res.json();
          serverDefaults = info;
          const saved = readSavedSettings();
          if (!Number.isFinite(saved.temperature) || saved.temperature < 0) {
            settings.temperature = info.default_sampling?.temperature ?? 0.8;
          }
          if (!Number.isFinite(saved.repeatPenalty) || saved.repeatPenalty <= 0) {
            settings.repeatPenalty = info.default_sampling?.repeat_penalty ?? 1;
          }
          if (info.default_max_tokens && !(Number.isInteger(saved.maxTokens) && saved.maxTokens > 0)) {
            settings.maxTokens = info.default_max_tokens;
          }
          applySettingsToUI();
          if (info.model_name) document.getElementById('info-model').textContent = info.model_name;
          if (info.layer_count) document.getElementById('info-layers').textContent = info.layer_count;
          if (info.workers) document.getElementById('info-workers').textContent = info.workers;
          if (info.window_mb) document.getElementById('info-window').textContent = info.window_mb + ' MiB';
          if (info.cpu) document.getElementById('info-cpu').textContent = info.cpu;
          if (info.ram) document.getElementById('info-ram').textContent = info.ram;
        }
      } catch (e) {
        console.warn('Could not fetch /api/info:', e);
      }
    }

    function updateStatus() {
      const turns = messages.filter(m => m.role === 'user').length;
      if (turns === 0) {
        statusText.textContent = 'Bereit';
      } else {
        statusText.textContent = turns + (turns === 1 ? ' Nachricht im Kontext' : ' Nachrichten im Kontext');
      }
    }

    clearBtn.addEventListener('click', () => {
      if (isGenerating && abortController) {
        abortController.abort();
      }
      messages = [];
      chatContainer.innerHTML = '';
      tokenCounter.textContent = '';
      appendMessageBubble('assistant', 'Stream-PT', 'Neuer Chat gestartet. Wie kann ich helfen?');
      updateStatus();
    });

    promptInput.addEventListener('input', () => {
      promptInput.style.height = 'auto';
      promptInput.style.height = Math.min(promptInput.scrollHeight, 150) + 'px';
    });

    promptInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
      }
    });

    sendBtn.addEventListener('click', sendMessage);

    stopBtn.addEventListener('click', () => {
      if (abortController) {
        abortController.abort();
      }
    });

    function appendMessageBubble(role, roleName, text = '') {
      const msgDiv = document.createElement('div');
      msgDiv.className = 'message ' + role;

      const roleDiv = document.createElement('div');
      roleDiv.className = 'message-role';
      roleDiv.textContent = roleName;
      msgDiv.appendChild(roleDiv);

      const bubbleDiv = document.createElement('div');
      bubbleDiv.className = 'bubble';
      bubbleDiv.textContent = text;
      msgDiv.appendChild(bubbleDiv);

      chatContainer.appendChild(msgDiv);
      chatContainer.scrollTop = chatContainer.scrollHeight;
      return bubbleDiv;
    }

    async function sendMessage() {
      const text = promptInput.value.trim();
      if (!text || isGenerating) return;

      promptInput.value = '';
      promptInput.style.height = 'auto';

      messages.push({ role: 'user', content: text });
      appendMessageBubble('user', 'Sie', text);
      updateStatus();

      const assistantBubble = appendMessageBubble('assistant', 'Stream-PT', '');
      const cursor = document.createElement('span');
      cursor.className = 'cursor';
      assistantBubble.appendChild(cursor);

      isGenerating = true;
      sendBtn.disabled = true;
      stopBtn.style.display = 'inline-flex';
      abortController = new AbortController();

      let fullResponse = '';
      let generatedTokens = 0;
      const startTime = performance.now();
      tokenCounter.textContent = 'Generiere (max ' + settings.maxTokens + ' Tokens)...';

      const outgoingMessages = [];
      if (settings.systemPrompt && settings.systemPrompt.trim()) {
        outgoingMessages.push({ role: 'system', content: settings.systemPrompt.trim() });
      }
      outgoingMessages.push(...messages);

      try {
        const response = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            messages: outgoingMessages,
            max_tokens: settings.maxTokens,
            temperature: settings.temperature,
            repeat_penalty: settings.repeatPenalty,
            system_prompt: settings.systemPrompt
          }),
          signal: abortController.signal
        });

        if (!response.ok) {
          throw new Error('HTTP ' + response.status + ': ' + (await response.text()));
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let buffer = '';

        while (true) {
          const { value, done } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop();

          for (const line of lines) {
            const trimmed = line.trim();
            if (!trimmed.startsWith('data:')) continue;
            const payload = trimmed.substring(5).trim();
            if (payload === '[DONE]') break;
            try {
              const event = JSON.parse(payload);
              if (event.error) {
                fullResponse += '\n[Fehler: ' + event.error + ']';
                assistantBubble.textContent = fullResponse;
                assistantBubble.appendChild(cursor);
              } else if (event.text !== undefined) {
                generatedTokens++;
                fullResponse += event.text;
                assistantBubble.textContent = fullResponse;
                assistantBubble.appendChild(cursor);
                chatContainer.scrollTop = chatContainer.scrollHeight;
                const elapsedSec = (performance.now() - startTime) / 1000;
                const speed = elapsedSec > 0 ? (generatedTokens / elapsedSec).toFixed(1) : '0';
                tokenCounter.textContent = generatedTokens + ' / ' + settings.maxTokens + ' Tokens (' + speed + ' T/s)';
              }
              if (event.done) {
                break;
              }
            } catch (err) {
              console.error('SSE parse error:', err, payload);
            }
          }
        }

        cursor.remove();
        if (fullResponse) {
          messages.push({ role: 'assistant', content: fullResponse });
        }
        const totalElapsed = ((performance.now() - startTime) / 1000).toFixed(1);
        tokenCounter.textContent = generatedTokens + ' Tokens generiert in ' + totalElapsed + 's';
      } catch (err) {
        cursor.remove();
        if (err.name !== 'AbortError') {
          assistantBubble.textContent += '\n[Fehler: ' + err.message + ']';
        } else if (fullResponse) {
          messages.push({ role: 'assistant', content: fullResponse });
          tokenCounter.textContent = 'Gestoppt bei ' + generatedTokens + ' Tokens';
        }
      } finally {
        isGenerating = false;
        sendBtn.disabled = false;
        stopBtn.style.display = 'none';
        abortController = null;
        updateStatus();
        promptInput.focus();
      }
    }

    // Init
    loadSettings();
    fetchServerInfo();
  </script>
</body>
</html>
`

func main() {
	sampling := forward.DefaultSamplingOptions()
	flag.Float64Var(&sampling.Temperature, "temp", sampling.Temperature, "Sampling temperature (0 selects greedy decoding with penalties)")
	flag.IntVar(&sampling.TopK, "top-k", sampling.TopK, "Top-k candidate limit (0 disables)")
	flag.Float64Var(&sampling.TopP, "top-p", sampling.TopP, "Nucleus probability threshold (1 disables)")
	flag.Float64Var(&sampling.MinP, "min-p", sampling.MinP, "Minimum probability relative to the best token (0 disables)")
	flag.Float64Var(&sampling.RepeatPenalty, "repeat-penalty", sampling.RepeatPenalty, "Repetition penalty (1 disables)")
	flag.IntVar(&sampling.RepeatLastN, "repeat-last-n", sampling.RepeatLastN, "Penalty history length (0 disables, -1 uses all context)")
	flag.Float64Var(&sampling.FrequencyPenalty, "frequency-penalty", 0, "Penalty per previous token occurrence")
	flag.Float64Var(&sampling.PresencePenalty, "presence-penalty", 0, "Penalty for previously seen tokens")
	seedFlag := flag.String("seed", "", "Sampling seed (empty selects a random seed)")
	threadsFlag := flag.Int("threads", runtime.GOMAXPROCS(0), "Number of parallel worker goroutines / compute threads")
	flag.IntVar(threadsFlag, "t", runtime.GOMAXPROCS(0), "Number of parallel worker goroutines (shorthand)")

	windowMBFlag := flag.Int("window-mb", 64, "Streaming mmap chunk window size in MiB")
	flag.IntVar(windowMBFlag, "w", 64, "Streaming mmap chunk window size in MiB (shorthand)")

	maxTokensFlag := flag.Int("max-tokens", 512, "Maximum number of tokens to generate")
	flag.IntVar(maxTokensFlag, "n", 512, "Maximum number of tokens to generate (shorthand)")

	promptFlag := flag.String("prompt", "", "Prompt / input text for generation (if set, runs CLI generation instead of web server)")
	flag.StringVar(promptFlag, "p", "", "Prompt / input text for generation (shorthand)")

	addrFlag := flag.String("addr", ":8080", "HTTP server address for the chat web server")
	flag.StringVar(addrFlag, "a", ":8080", "HTTP server address (shorthand)")

	expertCacheMBFlag := flag.Uint64("expert-cache-mb", 0, "Maximum locked hot-expert cache in MiB (0 disables retention)")
	expertMinUsesFlag := flag.Uint64("expert-cache-min-uses", 8, "Expert matrix uses before hot-cache admission")
	expertStatsFlag := flag.Bool("expert-stats", false, "Track and report selected expert matrix frequencies")
	noExpertLookaheadFlag := flag.Bool("no-expert-lookahead", false, "Disable prefetch of the next selected expert run (for comparison)")

	flag.Parse()
	if *seedFlag != "" {
		seed, err := strconv.ParseUint(*seedFlag, 10, 64)
		if err != nil {
			log.Fatalf("Invalid seed: %v", err)
		}
		sampling.Seed = &seed
	}
	if err := sampling.Validate(); err != nil {
		log.Fatal(err)
	}
	if *expertCacheMBFlag > ^uint64(0)>>20 {
		log.Fatal("expert-cache-mb exceeds supported size")
	}

	paths := []string{
		"model/gpt-oss-120b-Q4_0-00001-of-00002.gguf",
		"model/gpt-oss-120b-Q4_0-00002-of-00002.gguf",
	}

	model, err := ggufindex.Open(paths)
	if err != nil {
		log.Fatal(err)
	}

	reader, err := ggufmmap.Open(model)
	if err != nil {
		log.Fatal(err)
	}
	defer func() {
		if err := reader.Close(); err != nil {
			log.Print(err)
		}
	}()
	if err := reader.ConfigureExpertLookahead(!*noExpertLookaheadFlag); err != nil {
		log.Fatalf("Failed to configure expert lookahead: %v", err)
	}
	if err := reader.ConfigureExpertCache(ggufmmap.ExpertCacheOptions{
		MaxBytes:   *expertCacheMBFlag << 20,
		MinUses:    *expertMinUsesFlag,
		TrackStats: *expertStatsFlag,
	}); err != nil {
		log.Fatalf("Failed to configure expert cache: %v", err)
	}

	fmt.Println("╭────────────────────────────────────────────────────────────╮")
	fmt.Println("│                    SYSTEM & MODELL                         │")
	fmt.Println("├────────────────────────────────────────────────────────────┤")
	fmt.Printf("│ CPU:    %-51s │\n", cpuModel())
	fmt.Printf("│ RAM:    %-51s │\n", totalRAM())
	fmt.Printf("│ Modell: %-51s │\n", modelDescription(paths))
	fmt.Printf("│ Layer:  %-51d │\n", model.LayerCount)
	fmt.Println("╰────────────────────────────────────────────────────────────╯")
	fmt.Printf("Gewichte aus %d Dateien bleiben dateibasiert geladen.\n", len(model.Paths))

	threads := *threadsFlag
	if threads <= 0 {
		threads = runtime.GOMAXPROCS(0)
	}
	runtime.GOMAXPROCS(threads)

	windowBytes := uint64(*windowMBFlag) * 1024 * 1024
	if windowBytes == 0 {
		windowBytes = forward.DefaultQ40WindowBytes
	}

	engineOpts := forward.EngineOptions{
		Workers:     threads,
		WindowBytes: windowBytes,
		Sampling:    &sampling,
	}

	engine, err := forward.NewEngineWithOptions(model, reader, engineOpts)
	if err != nil {
		log.Fatalf("Failed to initialize inference engine: %v", err)
	}

	prompt := *promptFlag
	if prompt == "" && flag.NArg() > 0 {
		prompt = strings.Join(flag.Args(), " ")
	}

	if prompt != "" {
		runCLI(engine, reader, prompt, *maxTokensFlag, *expertStatsFlag, *expertCacheMBFlag > 0)
		return
	}

	server := &ChatServer{
		Engine:           engine,
		DefaultMaxTokens: *maxTokensFlag,
		ModelInfo: ServerInfo{
			ModelName:        modelDescription(paths),
			LayerCount:       model.LayerCount,
			Workers:          engine.Options.Workers,
			WindowMB:         int(engine.Options.WindowBytes / (1024 * 1024)),
			DefaultMaxTokens: *maxTokensFlag,
			CPU:              cpuModel(),
			RAM:              totalRAM(),
		},
	}

	addr := *addrFlag
	fmt.Printf("\n--- Stream-PT Chat Server gestartet ---\n")
	fmt.Printf("Adresse:           http://localhost%s\n", formatAddr(addr))
	fmt.Printf("Workers (Threads): %d\n", engine.Options.Workers)
	fmt.Printf("Chunk Window:      %d MiB\n", engine.Options.WindowBytes/(1024*1024))
	fmt.Printf("Bereit für Browser-Verbindungen...\n\n")

	if err := http.ListenAndServe(addr, server.routes()); err != nil {
		log.Fatalf("HTTP-Server Fehler: %v", err)
	}
}

func formatAddr(addr string) string {
	if strings.HasPrefix(addr, ":") {
		return addr
	}
	if strings.HasPrefix(addr, "127.0.0.1:") {
		return strings.TrimPrefix(addr, "127.0.0.1")
	}
	if strings.HasPrefix(addr, "0.0.0.0:") {
		return strings.TrimPrefix(addr, "0.0.0.0")
	}
	return ":" + addr
}

func runCLI(engine *forward.Engine, reader *ggufmmap.Reader, prompt string, maxTokens int, expertStats bool, hasCache bool) {
	fmt.Printf("\n--- Streaming Inference Pipeline Ready ---\n")
	fmt.Printf("Workers (Threads): %d\n", engine.Options.Workers)
	fmt.Printf("Go compute slots:  %d\n", runtime.GOMAXPROCS(0))
	fmt.Printf("Chunk Window:      %d MiB\n", engine.Options.WindowBytes/(1024*1024))
	fmt.Printf("Prompt:            %s\n", prompt)

	promptTokens, err := engine.Tokenizer.EncodeChatPrompt(prompt)
	if err != nil {
		log.Fatalf("Failed to encode chat prompt: %v", err)
	}
	fmt.Printf("Encoded %d prompt tokens: %v\n", len(promptTokens), promptTokens)

	ctx := context.Background()
	fmt.Printf("\nGenerating tokens (streaming layer-by-layer):\n")
	inferenceStart := time.Now()
	_, err = engine.Generate(ctx, promptTokens, maxTokens, func(tokenID int, text string) bool {
		fmt.Printf("[Token %d: %q]\n", tokenID, text)
		return true
	})
	fmt.Printf("Inferenzzeit: %s\n", time.Since(inferenceStart).Round(time.Millisecond))
	if expertStats || hasCache {
		stats := reader.ExpertCacheStats()
		fmt.Printf("Expert weights: %d selected ranges, %.2f MiB; %d mapped runs; %d cache hits; %.2f MiB locked; %d lock failures\n",
			stats.SelectedRanges, float64(stats.SelectedBytes)/(1<<20), stats.MappedRuns,
			stats.CacheHits, float64(stats.CachedBytes)/(1<<20), stats.LockFailures)
		fmt.Printf("Expert prefetch: %d bounded requests; %d failures\n", stats.PrefetchCalls, stats.PrefetchFailures)
		if expertStats {
			ranges := make([]ggufindex.Range, 0, len(stats.RangeUses))
			for span := range stats.RangeUses {
				ranges = append(ranges, span)
			}
			sort.Slice(ranges, func(i, j int) bool {
				if stats.RangeUses[ranges[i]] != stats.RangeUses[ranges[j]] {
					return stats.RangeUses[ranges[i]] > stats.RangeUses[ranges[j]]
				}
				if ranges[i].File != ranges[j].File {
					return ranges[i].File < ranges[j].File
				}
				return ranges[i].Start < ranges[j].Start
			})
			for _, span := range ranges[:min(8, len(ranges))] {
				fmt.Printf("Hot expert matrix: %s [%d,%d), %d uses\n", filepath.Base(span.File), span.Start, span.End, stats.RangeUses[span])
			}
		}
	}
	if err != nil {
		log.Printf("Inference step info: %v", err)
	}
}

func cpuModel() string {
	data, err := os.ReadFile("/proc/cpuinfo")
	if err != nil {
		return runtime.GOARCH
	}

	for line := range strings.SplitSeq(string(data), "\n") {
		if key, value, ok := strings.Cut(line, ":"); ok && strings.TrimSpace(key) == "model name" {
			return strings.TrimSpace(value)
		}
	}
	return runtime.GOARCH
}

func totalRAM() string {
	data, err := os.ReadFile("/proc/meminfo")
	if err != nil {
		return "Unbekannt"
	}

	for _, line := range strings.Split(string(data), "\n") {
		fields := strings.Fields(line)
		if len(fields) >= 2 && fields[0] == "MemTotal:" {
			kilobytes, err := strconv.ParseUint(fields[1], 10, 64)
			if err == nil {
				return fmt.Sprintf("%.1f GiB", float64(kilobytes)/(1024*1024))
			}
		}
	}
	return "Unbekannt"
}

func modelDescription(paths []string) string {
	if len(paths) == 0 {
		return "Unbekannt"
	}

	name := strings.TrimSuffix(filepath.Base(paths[0]), ".gguf")
	if separator := strings.LastIndex(name, "-of-"); separator >= 0 {
		if shard := strings.LastIndex(name[:separator], "-"); shard >= 0 {
			name = name[:shard]
		}
	}
	if len(paths) > 1 {
		return fmt.Sprintf("%s (%d Dateien)", name, len(paths))
	}
	return name
}
