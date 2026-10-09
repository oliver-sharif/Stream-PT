//go:build goexperiment.simd

package forward

import (
	"fmt"
	"strings"
	"time"
)

// ChatMessage represents a single message in a conversation.
type ChatMessage struct {
	Role    string `json:"role"`    // "system", "developer", "user", "assistant"
	Content string `json:"content"` // message content
}

// EncodeChatMessages encodes tool-free GPT-OSS history with an open assistant header.
func (t *Tokenizer) EncodeChatMessages(messages []ChatMessage) ([]int, error) {
	if t == nil {
		return nil, fmt.Errorf("nil tokenizer")
	}
	if len(messages) == 0 {
		return nil, fmt.Errorf("empty messages")
	}
	for i, msg := range messages {
		switch msg.Role {
		case "system", "developer":
			if i != 0 {
				return nil, fmt.Errorf("%s instructions must be the first message", msg.Role)
			}
		case "user", "assistant":
		default:
			return nil, fmt.Errorf("unsupported chat role %q (tools are not supported)", msg.Role)
		}
		if msg.Role == "assistant" {
			for _, marker := range []string{"<|channel|>", "<|meta_sep|>"} {
				if strings.Contains(msg.Content, marker+"analysis<|message|>") ||
					strings.Contains(msg.Content, marker+"final<|message|>") ||
					strings.Contains(msg.Content, marker+"analysis<|im_sep|>") ||
					strings.Contains(msg.Content, marker+"final<|im_sep|>") {
					return nil, fmt.Errorf("assistant content must not contain Harmony channel headers")
				}
			}
		}
	}
	specialID := func(names ...string) (int, error) {
		for _, name := range names {
			if id, ok := t.TokenMap[name]; ok {
				return id, nil
			}
		}
		return 0, fmt.Errorf("missing Harmony token %s", names[0])
	}
	start, err := specialID("<|start|>", "<|im_start|>")
	if err != nil {
		return nil, err
	}
	message, err := specialID("<|message|>", "<|im_sep|>")
	if err != nil {
		return nil, err
	}
	end, err := specialID("<|end|>", "<|im_end|>")
	if err != nil {
		return nil, err
	}
	channel, err := specialID("<|channel|>", "<|meta_sep|>")
	if err != nil {
		return nil, err
	}

	var tokens []int
	appendMessage := func(role, content string) {
		tokens = append(tokens, start)
		tokens = append(tokens, t.Encode(role)...)
		if role == "assistant" {
			tokens = append(tokens, channel)
			tokens = append(tokens, t.Encode("final")...)
		}
		tokens = append(tokens, message)
		tokens = append(tokens, t.Encode(content)...)
		tokens = append(tokens, end)
	}
	// Based on openai/gpt-oss-120b/chat_template.jinja, without tools.
	appendMessage("system", "You are ChatGPT, a large language model trained by OpenAI.\n"+
		"Knowledge cutoff: 2024-06\nCurrent date: "+time.Now().Format("2006-01-02")+
		"\n\nReasoning: medium\n\n# Valid channels: analysis, commentary, final. Channel must be included for every message.")
	if messages[0].Role == "system" || messages[0].Role == "developer" {
		if messages[0].Content != "" {
			appendMessage("developer", "# Instructions\n\n"+messages[0].Content+"\n\n")
		}
		messages = messages[1:]
	}
	for _, msg := range messages {
		appendMessage(msg.Role, msg.Content)
	}

	// Leave the channel and message separator for the model to generate.
	tokens = append(tokens, start)
	tokens = append(tokens, t.Encode("assistant")...)
	return tokens, nil
}

// EncodeChatPrompt frames a user message and opens a GPT-OSS assistant header.
func (t *Tokenizer) EncodeChatPrompt(prompt string) ([]int, error) {
	return t.EncodeChatMessages([]ChatMessage{{Role: "user", Content: prompt}})
}
