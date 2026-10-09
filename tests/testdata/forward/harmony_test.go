//go:build goexperiment.simd

package forward

import (
	"reflect"
	"strings"
	"testing"
	"time"
)

func harmonyByteTokenizer(aliases bool) *Tokenizer {
	vocab := make(map[string]int)
	// GPT-2's byte alphabet: printable bytes retain their code points.
	n := 256
	for b := 0; b < 256; b++ {
		r := b
		if !((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || b >= 174) {
			r = n
			n++
		}
		vocab[string(rune(r))] = b
	}
	names := []string{"<|start|>", "<|message|>", "<|end|>", "<|channel|>", "<|return|>", "<|call|>", "<|meta_start|>"}
	if aliases {
		names = []string{"<|im_start|>", "<|im_sep|>", "<|im_end|>", "<|meta_sep|>", "<|fim_suffix|>", "<|call|>", "<|meta_start|>"}
	}
	for i, name := range names {
		vocab[name] = 256 + i
	}
	return &Tokenizer{TokenMap: vocab}
}

func harmonyBytes(text string) []int {
	ids := make([]int, len(text))
	for i := 0; i < len(text); i++ {
		ids[i] = int(text[i])
	}
	return ids
}

func TestHarmonyTemplateCanonical(t *testing.T) {
	// Source: https://huggingface.co/openai/gpt-oss-120b/raw/main/chat_template.jinja
	// Tool-free inference with add_generation_prompt=true: assistant history ends with end.
	for _, aliases := range []bool{false, true} {
		t.Run(map[bool]string{false: "canonical", true: "aliases"}[aliases], func(t *testing.T) {
			tokenizer := harmonyByteTokenizer(aliases)
			messages := []ChatMessage{{"system", "Be concise."}, {"user", "Hello"}, {"assistant", "Caf\u00e9"}, {"user", "Continue"}}
			before := time.Now().Format("2006-01-02")
			got, err := tokenizer.EncodeChatMessages(messages)
			if err != nil {
				t.Fatal(err)
			}
			wantForDate := func(date string) []int {
				var want []int
				frame := func(role, body string, channel bool, end int) {
					want = append(want, 256)
					want = append(want, harmonyBytes(role)...)
					if channel {
						want = append(want, 259)
						want = append(want, harmonyBytes("final")...)
					}
					want = append(want, 257)
					want = append(want, harmonyBytes(body)...)
					want = append(want, end)
				}
				frame("system", "You are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2024-06\nCurrent date: "+date+"\n\nReasoning: medium\n\n# Valid channels: analysis, commentary, final. Channel must be included for every message.", false, 258)
				frame("developer", "# Instructions\n\nBe concise.\n\n", false, 258)
				frame("user", "Hello", false, 258)
				frame("assistant", "Caf\u00e9", true, 258)
				frame("user", "Continue", false, 258)
				want = append(want, 256)
				return append(want, harmonyBytes("assistant")...)
			}
			want := wantForDate(before)
			if !reflect.DeepEqual(got, want) && !reflect.DeepEqual(got, wantForDate(time.Now().Format("2006-01-02"))) {
				t.Fatalf("canonical token IDs mismatch:\ngot  %v\nwant %v", got, want)
			}
		})
	}
}

func TestHarmonyTemplateLastAssistant(t *testing.T) {
	for _, aliases := range []bool{false, true} {
		ids, err := harmonyByteTokenizer(aliases).EncodeChatMessages([]ChatMessage{{"assistant", "answer"}})
		if err != nil {
			t.Fatal(err)
		}
		want := append([]int{256}, harmonyBytes("assistant")...)
		want = append(want, 259)
		want = append(want, harmonyBytes("final")...)
		want = append(want, 257)
		want = append(want, harmonyBytes("answer")...)
		want = append(want, 258, 256)
		want = append(want, harmonyBytes("assistant")...)
		if len(ids) < len(want) || !reflect.DeepEqual(ids[len(ids)-len(want):], want) {
			t.Fatalf("last assistant must end with end before the generation prefix: %v", ids)
		}
	}
}

func TestHarmonyTemplateRoles(t *testing.T) {
	tokenizer := harmonyByteTokenizer(false)
	for _, messages := range [][]ChatMessage{
		nil, {{"", "hello"}}, {{"tool", "hello"}}, {{"unknown", "hello"}},
		{{"user", "hello"}, {"system", "late"}}, {{"system", "one"}, {"developer", "two"}},
		{{"assistant", "<|channel|>analysis<|message|>secret"}},
	} {
		if _, err := tokenizer.EncodeChatMessages(messages); err == nil {
			t.Errorf("expected error for %#v", messages)
		}
	}
	for _, role := range []string{"system", "developer", "user", "assistant"} {
		ids, err := tokenizer.EncodeChatMessages([]ChatMessage{{role, "hello"}})
		if err != nil || len(ids) == 0 {
			t.Errorf("role %s: %v", role, err)
		}
	}
	var absent *Tokenizer
	if _, err := absent.EncodeChatPrompt("hello"); err == nil {
		t.Fatal("nil tokenizer accepted")
	}
	delete(tokenizer.TokenMap, "<|return|>")
	if _, err := tokenizer.EncodeChatPrompt("hello"); err != nil {
		t.Fatalf("inference template must not require return: %v", err)
	}
	delete(tokenizer.TokenMap, "<|end|>")
	if _, err := tokenizer.EncodeChatPrompt("hello"); err == nil {
		t.Fatal("missing Harmony end accepted")
	}
}

func TestHarmonyTemplateNoInstructions(t *testing.T) {
	tokenizer := harmonyByteTokenizer(false)
	plain, err := tokenizer.EncodeChatPrompt("Hello")
	if err != nil {
		t.Fatal(err)
	}
	empty, err := tokenizer.EncodeChatMessages([]ChatMessage{{"system", ""}, {"user", "Hello"}})
	if err != nil || !reflect.DeepEqual(plain, empty) {
		t.Fatalf("empty instructions must be omitted: %v", err)
	}
	if strings.Contains(stringFromHarmonyBytes(plain), "# Tools") {
		t.Fatal("tool-free template advertised tools")
	}
}

func stringFromHarmonyBytes(ids []int) string {
	var text []byte
	for _, id := range ids {
		if id < 256 {
			text = append(text, byte(id))
		}
	}
	return string(text)
}

func TestHarmonyStream(t *testing.T) {
	type token struct {
		id   int
		text string
	}
	text := func(value string) token { return token{0, value} }
	start, message, end := token{256, "<|start|>"}, token{257, "<|message|>"}, token{258, "<|end|>"}
	channel, ret, call := token{259, "<|channel|>"}, token{260, "<|return|>"}, token{261, "<|call|>"}
	cases := []struct {
		name   string
		tokens []token
		want   string
	}{
		{"channel-less", []token{message, text("Caf\u00e9 🌍"), ret, text("hidden")}, "Caf\u00e9 🌍"},
		{"analysis-to-final", []token{channel, text("analysis"), message, text("secret"), end, start, text("assistant"), channel, text("final"), message, text("answer"), ret}, "answer"},
		{"end-to-channel", []token{channel, text("analysis"), message, text("secret"), end, channel, text("final"), message, text("answer")}, "answer"},
		{"end-to-message", []token{channel, text("analysis"), message, text("secret"), end, message, text("answer")}, "answer"},
		{"commentary", []token{channel, text("commentary"), message, text("hidden")}, ""},
		{"tool-before-channel", []token{text(" to=python"), channel, text("final"), message, text("hidden"), call}, ""},
		{"tool-after-channel", []token{channel, text("final to=python"), message, text("hidden"), call}, ""},
		{"tool-channel-less", []token{text(" to=browser"), message, text("hidden")}, ""},
		{"tool-response", []token{start, text("python"), message, text("hidden"), end, start, text("assistant"), message, text("answer")}, "answer"},
		{"user-message", []token{start, text("user"), channel, text("final"), message, text("hidden")}, ""},
		{"incomplete-header", []token{channel, text("final")}, ""},
		{"unknown-channel", []token{channel, text("other"), message, text("hidden")}, ""},
		{"format-header", []token{{262, "<|meta_start|>"}, text("json"), message, text("{}")}, "{}"},
		{"final-format-header", []token{channel, text("final"), {262, "<|meta_start|>"}, text("json"), message, text("{}")}, "{}"},
		{"analysis-format-header", []token{channel, text("analysis"), {262, "<|meta_start|>"}, text("json"), message, text("hidden")}, ""},
		{"tool-format-header", []token{{262, "<|meta_start|>"}, text("json to=python"), message, text("hidden")}, ""},
		{"malformed-format-recipient", []token{{262, ""}, text("json to = python"), message, text("hidden")}, ""},
		{"recipient-across-controls", []token{text(" to"), channel, text("="), {262, ""}, text("python"), message, text("hidden")}, ""},
		{"duplicate-channel", []token{channel, text("fi"), channel, text("nal"), message, text("hidden")}, ""},
		{"duplicate-format", []token{{262, ""}, text("to"), {262, ""}, text("python"), message, text("hidden")}, ""},
		{"channel-after-format", []token{{262, ""}, text("json"), channel, text("final"), message, text("hidden")}, ""},
		{"control-text-independent", []token{{257, ""}, text("answer"), {260, "wrong text"}, text("hidden")}, "answer"},
		{"return-blocks-continuation", []token{message, text("answer"), ret, channel, text("final"), message, text("hidden")}, "answer"},
	}
	for _, aliases := range []bool{false, true} {
		for _, tc := range cases {
			t.Run(tc.name+map[bool]string{false: "/canonical", true: "/aliases"}[aliases], func(t *testing.T) {
				stream := NewHarmonyStream(harmonyByteTokenizer(aliases))
				var got string
				for _, tok := range tc.tokens {
					out := stream.Push(tok.id, tok.text)
					if tok.id >= 256 && out != "" {
						t.Fatalf("control token leaked: %q", out)
					}
					got += out
				}
				if got != tc.want {
					t.Fatalf("got %q, want %q", got, tc.want)
				}
			})
		}
	}
}

func TestHarmonyStreamReasoning(t *testing.T) {
	for _, aliases := range []bool{false, true} {
		for _, recipient := range []string{"", " to=python"} {
			stream := NewHarmonyStream(harmonyByteTokenizer(aliases))
			var answer, reasoning strings.Builder
			push := func(id int, text string) {
				a, r := stream.PushWithReasoning(id, text)
				answer.WriteString(a)
				reasoning.WriteString(r)
			}
			push(0, recipient)
			push(259, "")
			for _, r := range "analysis" {
				push(0, string(r))
			}
			push(257, "")
			push(0, "Caf\u00e9 🙂")
			push(258, "")
			push(256, "")
			push(0, "assistant")
			push(259, "")
			push(0, "final")
			push(257, "")
			push(0, "Answer")
			push(260, "")
			push(0, "hidden")
			wantReasoning := "Caf\u00e9 🙂"
			if recipient != "" {
				wantReasoning = ""
			}
			if answer.String() != "Answer" || reasoning.String() != wantReasoning {
				t.Fatalf("aliases %v, recipient %q: answer %q, reasoning %q", aliases, recipient, answer.String(), reasoning.String())
			}
		}
	}
	stream := NewHarmonyStream(nil)
	if answer, reasoning := stream.PushWithReasoning(0, "plain"); answer != "plain" || reasoning != "" {
		t.Fatalf("non-Harmony: answer %q, reasoning %q", answer, reasoning)
	}
}

func TestHarmonyStreamTokenwiseHeaderAndUTF8(t *testing.T) {
	stream := NewHarmonyStream(harmonyByteTokenizer(false))
	pushHeader := func(value string) {
		for _, r := range value {
			if out := stream.Push(0, string(r)); out != "" {
				t.Fatalf("header leaked: %q", out)
			}
		}
	}
	stream.Push(259, "")
	pushHeader("analysis")
	stream.Push(257, "")
	if out := stream.Push(0, "Geheim 🤫"); out != "" {
		t.Fatalf("analysis leaked: %q", out)
	}
	stream.Push(258, "")
	stream.Push(256, "")
	pushHeader("assistant")
	stream.Push(259, "")
	pushHeader("final")
	stream.Push(257, "")
	for _, chunk := range []string{"Ca", "f", "\u00e9", " ", "🌍"} {
		if out := stream.Push(0, chunk); out != chunk {
			t.Fatalf("valid UTF-8 chunk changed: got %q, want %q", out, chunk)
		}
	}
	stream.Push(258, "")
	pushHeader(" to=python")
	stream.Push(259, "")
	pushHeader("final")
	stream.Push(257, "")
	if out := stream.Push(0, "tool request"); out != "" {
		t.Fatalf("tokenwise recipient leaked: %q", out)
	}
}

func TestHarmonyStreamBoundedHeader(t *testing.T) {
	for _, chunk := range []string{"x", strings.Repeat("x", 8192)} {
		stream := NewHarmonyStream(harmonyByteTokenizer(false))
		stream.Push(262, "")
		for i := 0; i < 8192/len(chunk); i++ {
			if out := stream.Push(0, chunk); out != "" {
				t.Fatalf("header leaked: %q", out)
			}
			if size := len(stream.role) + len(stream.channel) + len(stream.format); size > 4096 {
				t.Fatalf("header buffer unbounded: %d bytes", size)
			}
		}
		stream.Push(257, "")
		if out := stream.Push(0, "hidden"); out != "" {
			t.Fatalf("oversized header body leaked: %q", out)
		}
		stream.Push(256, "")
		stream.Push(0, "assistant")
		stream.Push(257, "")
		if out := stream.Push(0, "answer"); out != "answer" {
			t.Fatalf("valid next message lost: %q", out)
		}
	}
}

func TestHarmonyStreamPassthrough(t *testing.T) {
	for _, tokenizer := range []*Tokenizer{nil, {TokenMap: map[string]int{"hello": 0}}, {TokenMap: map[string]int{"<|im_start|>": 1}}} {
		stream := NewHarmonyStream(tokenizer)
		for _, chunk := range []string{"hello", "Caf\u00e9 🌍", "<|start|>", ""} {
			if out := stream.Push(1, chunk); out != chunk {
				t.Fatalf("non-Harmony text changed: got %q, want %q", out, chunk)
			}
		}
	}
}

func TestHarmonyStreamMixedAliases(t *testing.T) {
	tokenizer := harmonyByteTokenizer(false)
	for i, alias := range []string{"<|im_start|>", "<|im_sep|>", "<|im_end|>", "<|meta_sep|>", "<|fim_suffix|>"} {
		tokenizer.TokenMap[alias] = 300 + i
	}
	stream := NewHarmonyStream(tokenizer)
	for _, tok := range []struct {
		id   int
		text string
	}{{303, ""}, {0, "analysis"}, {301, ""}, {0, "secret"}, {302, ""}, {300, ""}, {0, "assistant"}, {259, ""}, {0, "final"}, {257, ""}} {
		if out := stream.Push(tok.id, tok.text); out != "" {
			t.Fatalf("header or analysis leaked: %q", out)
		}
	}
	if out := stream.Push(0, "answer"); out != "answer" {
		t.Fatalf("final lost: %q", out)
	}
	if out := stream.Push(304, ""); out != "" {
		t.Fatalf("return alias leaked: %q", out)
	}
	if out := stream.Push(0, "hidden"); out != "" {
		t.Fatalf("post-return text leaked: %q", out)
	}
}
