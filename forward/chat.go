//go:build goexperiment.simd

package forward

import "fmt"

// EncodeChatPrompt frames a user message and starts a GPT-OSS Harmony final answer.
func (t *Tokenizer) EncodeChatPrompt(prompt string) ([]int, error) {
	if t == nil {
		return nil, fmt.Errorf("nil tokenizer")
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

	// Insert control token IDs directly; Encode treats the user input as ordinary text.
	tokens := []int{start}
	tokens = append(tokens, t.Encode("user")...)
	tokens = append(tokens, message)
	tokens = append(tokens, t.Encode(prompt)...)
	tokens = append(tokens, end, start)
	tokens = append(tokens, t.Encode("assistant")...)
	tokens = append(tokens, channel)
	tokens = append(tokens, t.Encode("final")...)
	tokens = append(tokens, message)
	return tokens, nil
}
