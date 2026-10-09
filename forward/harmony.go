//go:build goexperiment.simd

package forward

import "strings"

const harmonyHeaderLimit = 4096

// HarmonyStream filters a generated Harmony continuation to user-visible text.
// Text supplied to Push must already be valid UTF-8; decoding belongs to the caller.
type HarmonyStream struct {
	controls  map[int]string
	enabled   bool
	header    bool
	role      string
	channel   string
	format    string
	inChannel bool
	inFormat  bool
	visible   bool
	terminal  bool
}

// NewHarmonyStream begins in the open assistant header emitted by EncodeChatMessages.
// Vocabularies without Harmony framing pass through unchanged.
func NewHarmonyStream(tokenizer *Tokenizer) *HarmonyStream {
	s := &HarmonyStream{header: true, role: "assistant", controls: make(map[int]string)}
	if tokenizer == nil {
		return s
	}
	groups := [][]string{
		{"<|start|>", "<|im_start|>"},
		{"<|message|>", "<|im_sep|>"},
		{"<|end|>", "<|im_end|>"},
		{"<|channel|>", "<|meta_sep|>"},
		{"<|return|>", "<|fim_suffix|>"},
	}
	s.enabled = true
	for _, names := range groups {
		found := false
		for _, name := range names {
			if id, ok := tokenizer.TokenMap[name]; ok {
				s.controls[id] = names[0]
				found = true
			}
		}
		s.enabled = s.enabled && found
	}
	for name, id := range tokenizer.TokenMap {
		if strings.HasPrefix(name, "<|") && strings.HasSuffix(name, "|>") {
			if _, exists := s.controls[id]; !exists {
				s.controls[id] = name
			}
		}
	}
	return s
}

// Push consumes one token and returns only final or channel-less assistant body text.
// Headers, analysis, tool-directed messages, and control tokens are never returned.
func (s *HarmonyStream) Push(tokenID int, text string) string {
	if !s.enabled {
		return text
	}
	if control, ok := s.controls[tokenID]; ok {
		switch control {
		case "<|start|>":
			s.resetHeader("")
		case "<|end|>":
			s.resetHeader("assistant")
		case "<|return|>", "<|call|>", "<|endoftext|>":
			s.visible = false
			s.terminal = true
		case "<|channel|>":
			if s.header && !s.terminal && !s.inChannel && !s.inFormat {
				s.inChannel = true
				s.inFormat = false
			} else {
				s.visible = false
				if s.header {
					s.terminal = true
				}
			}
		case "<|meta_start|>":
			if s.header && !s.terminal && !s.inFormat {
				s.inFormat = true
			} else if s.header {
				s.terminal = true
			}
		case "<|message|>":
			if s.header && !s.terminal {
				role := strings.Fields(s.role)
				channel := strings.TrimSpace(s.channel)
				s.visible = len(role) == 1 && role[0] == "assistant" &&
					(channel == "" || channel == "final" || channel == "analysis") &&
					(!s.inFormat || len(strings.Fields(s.format)) == 1) &&
					!strings.Contains(s.format, "=") &&
					!strings.Contains(s.role+s.channel+s.format, "to=")
				s.header = false
			}
		default:
			if s.header {
				s.terminal = true
			}
		}
		return ""
	}
	if s.terminal {
		return ""
	}
	if s.header {
		if len(text) > harmonyHeaderLimit-len(s.role)-len(s.channel)-len(s.format) {
			s.resetHeader("")
			s.terminal = true
			return ""
		}
		if s.inFormat {
			s.format += text
		} else if s.inChannel {
			s.channel += text
		} else {
			s.role += text
		}
		return ""
	}
	if s.visible && strings.TrimSpace(s.channel) != "analysis" {
		return text
	}
	return ""
}

// PushWithReasoning separates analysis from the final answer without exposing headers or tools.
func (s *HarmonyStream) PushWithReasoning(tokenID int, text string) (answer, reasoning string) {
	answer = s.Push(tokenID, text)
	if _, control := s.controls[tokenID]; !control && s.enabled &&
		!s.header && !s.terminal && s.visible && strings.TrimSpace(s.channel) == "analysis" {
		reasoning = text
	}
	return answer, reasoning
}

func (s *HarmonyStream) resetHeader(role string) {
	s.header = true
	s.role = role
	s.channel = ""
	s.format = ""
	s.inChannel = false
	s.inFormat = false
	s.visible = false
	s.terminal = false
}
