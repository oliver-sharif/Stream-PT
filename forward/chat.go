//go:build goexperiment.simd

package forward

import "fmt"

// ChatMessage represents a single message in a conversation.
type ChatMessage struct {
	Role    string `json:"role"`    // "system", "user", "assistant"
	Content string `json:"content"` // message content
}

// EncodeChatMessages encodes a conversation history and starts a GPT-OSS Harmony final answer.
func (t *Tokenizer) EncodeChatMessages(messages []ChatMessage) ([]int, error) {
	if t == nil {
		return nil, fmt.Errorf("nil tokenizer")
	}
	if len(messages) == 0 {
		return nil, fmt.Errorf("empty messages")
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
	for _, msg := range messages {
		role := msg.Role
		if role == "" {
			role = "user"
		}
		tokens = append(tokens, start)
		tokens = append(tokens, t.Encode(role)...)
		if role == "assistant" {
			tokens = append(tokens, channel)
			tokens = append(tokens, t.Encode("final")...)
		}
		tokens = append(tokens, message)
		tokens = append(tokens, t.Encode(msg.Content)...)
		tokens = append(tokens, end)
	}

	// Trigger assistant completion
	tokens = append(tokens, start)
	tokens = append(tokens, t.Encode("assistant")...)
	tokens = append(tokens, channel)
	tokens = append(tokens, t.Encode("final")...)
	tokens = append(tokens, message)
	return tokens, nil
}

// EncodeChatPrompt frames a user message and starts a GPT-OSS Harmony final answer.
func (t *Tokenizer) EncodeChatPrompt(prompt string) ([]int, error) {
	return t.EncodeChatMessages([]ChatMessage{{Role: "user", Content: prompt}})
}
