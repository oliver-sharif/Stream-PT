//go:build goexperiment.simd

package forward

import (
	"bytes"
	"container/heap"
	"encoding/json"
	"fmt"
	"regexp"
	"strings"
	"unicode"
	"unicode/utf8"

	"Stream-PT/ggufindex"
)

type tokenizerPair struct{ left, right int }
type tokenizerMerge struct{ rank, id int }

type tokenizerBPE struct {
	pre     string
	merges  map[tokenizerPair]tokenizerMerge
	bytes   [256]int
	special *regexp.Regexp
}

func tokenizerMetadata(model *ggufindex.Model, key string, value any) error {
	meta, ok := model.Metadata[key]
	if !ok {
		return fmt.Errorf("missing %s metadata", key)
	}
	var data bytes.Buffer
	if err := meta.WriteJSON(&data); err != nil {
		return fmt.Errorf("%s: %w", key, err)
	}
	if err := json.Unmarshal(data.Bytes(), value); err != nil {
		return fmt.Errorf("%s: %w", key, err)
	}
	return nil
}

func (t *Tokenizer) loadBPE(model *ggufindex.Model) error {
	if _, ok := model.Metadata["tokenizer.ggml.merges"]; !ok {
		return nil
	}
	bpe := &tokenizerBPE{}
	if err := tokenizerMetadata(model, "tokenizer.ggml.pre", &bpe.pre); err != nil {
		return err
	}
	switch bpe.pre {
	case "gpt-4o", "o200k_base", "o200k_harmony":
	default:
		return fmt.Errorf("unsupported BPE pretokenizer %q", bpe.pre)
	}
	var merges []string
	if err := tokenizerMetadata(model, "tokenizer.ggml.merges", &merges); err != nil {
		return err
	}
	bpe.merges = make(map[tokenizerPair]tokenizerMerge, len(merges))
	for rank, merge := range merges {
		left, right, ok := strings.Cut(merge, " ")
		leftID, leftOK := t.TokenMap[left]
		rightID, rightOK := t.TokenMap[right]
		id, mergedOK := t.TokenMap[left+right]
		if !ok || left == "" || right == "" || !leftOK || !rightOK || !mergedOK {
			return fmt.Errorf("invalid tokenizer merge %d: %q", rank, merge)
		}
		pair := tokenizerPair{leftID, rightID}
		if _, exists := bpe.merges[pair]; !exists {
			bpe.merges[pair] = tokenizerMerge{rank, id}
		}
	}
	for b, r := range t.byteToUnicode {
		bpe.bytes[b] = -1
		if id, ok := t.TokenMap[string(r)]; ok {
			bpe.bytes[b] = id
		}
	}
	if _, ok := model.Metadata["tokenizer.ggml.token_type"]; ok {
		var types []int
		if err := tokenizerMetadata(model, "tokenizer.ggml.token_type", &types); err != nil {
			return err
		}
		if len(types) != len(t.Tokens) {
			return fmt.Errorf("tokenizer token_type length %d, want %d", len(types), len(t.Tokens))
		}
		var specials []string
		for id, typ := range types {
			if (typ == 3 || typ == 4) && t.Tokens[id] != "" {
				specials = append(specials, regexp.QuoteMeta(t.Tokens[id]))
			}
		}
		if len(specials) > 0 {
			bpe.special = regexp.MustCompile(strings.Join(specials, "|"))
			bpe.special.Longest()
		}
	}
	t.bpe = bpe
	return nil
}

// OpenAI's o200k_base pattern, including Unicode White_Space rather than RE2's
// ASCII-only \s: https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
// The last two alternatives (\s+(?!\S)|\s+) are handled below, since RE2 has no lookahead.
const tokenizerSpace = `\t\n\v\f\r \x{0085}\p{Z}`

var tokenizerO200k = regexp.MustCompile(`^(?:` +
	`[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|` +
	`[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|` +
	`\p{N}{1,3}| ?[^` + tokenizerSpace + `\p{L}\p{N}]+[\r\n/]*|[` + tokenizerSpace + `]*[\r\n]+)`)

func tokenizerPieceLen(text string) int {
	if match := tokenizerO200k.FindStringIndex(text); match != nil {
		return match[1]
	}
	end, last := 0, 0
	for pos, r := range text {
		if !unicode.IsSpace(r) {
			break
		}
		last, end = pos, pos+utf8.RuneLen(r)
	}
	if end > 0 {
		if end < len(text) && last > 0 {
			return last
		}
		return end
	}
	_, size := utf8.DecodeRuneInString(text)
	return size
}

type tokenizerSymbol struct {
	id, prev, next int
}

type tokenizerCandidate struct {
	left, right int
	pair        tokenizerPair
	merge       tokenizerMerge
}

type tokenizerQueue []tokenizerCandidate

func (q tokenizerQueue) Len() int { return len(q) }
func (q tokenizerQueue) Less(i, j int) bool {
	if q[i].merge.rank != q[j].merge.rank {
		return q[i].merge.rank < q[j].merge.rank
	}
	return q[i].left < q[j].left
}
func (q tokenizerQueue) Swap(i, j int) { q[i], q[j] = q[j], q[i] }
func (q *tokenizerQueue) Push(v any)   { *q = append(*q, v.(tokenizerCandidate)) }
func (q *tokenizerQueue) Pop() any {
	last := len(*q) - 1
	v := (*q)[last]
	*q = (*q)[:last]
	return v
}

func (t *Tokenizer) mergePiece(text string, ids []int) []int {
	symbols := make([]tokenizerSymbol, len(text))
	for i := range len(text) {
		symbols[i] = tokenizerSymbol{t.bpe.bytes[text[i]], i - 1, i + 1}
	}
	symbols[len(symbols)-1].next = -1
	queue := make(tokenizerQueue, 0, len(text))
	add := func(left int) {
		if left < 0 {
			return
		}
		right := symbols[left].next
		if right < 0 {
			return
		}
		pair := tokenizerPair{symbols[left].id, symbols[right].id}
		if merge, ok := t.bpe.merges[pair]; ok {
			heap.Push(&queue, tokenizerCandidate{left, right, pair, merge})
		}
	}
	for i := range symbols {
		add(i)
	}
	for queue.Len() > 0 {
		candidate := heap.Pop(&queue).(tokenizerCandidate)
		left, right := candidate.left, candidate.right
		if symbols[left].next != right || symbols[left].id != candidate.pair.left || symbols[right].id != candidate.pair.right {
			continue
		}
		symbols[left].id = candidate.merge.id
		symbols[left].next = symbols[right].next
		if next := symbols[right].next; next >= 0 {
			symbols[next].prev = left
		}
		symbols[right].id, symbols[right].next = -1, -1
		add(symbols[left].prev)
		add(left)
	}
	for i := 0; i >= 0; i = symbols[i].next {
		if symbols[i].id >= 0 {
			ids = append(ids, symbols[i].id)
		}
	}
	return ids
}

func (t *Tokenizer) encodeBPE(text string) []int {
	var ids []int
	for len(text) > 0 {
		end := len(text)
		var special []int
		if t.bpe.special != nil {
			special = t.bpe.special.FindStringIndex(text)
			if special != nil {
				end = special[0]
			}
		}
		ordinary := text[:end]
		for len(ordinary) > 0 {
			n := tokenizerPieceLen(ordinary)
			ids = t.mergePiece(ordinary[:n], ids)
			ordinary = ordinary[n:]
		}
		if special == nil {
			break
		}
		ids = append(ids, t.TokenMap[text[special[0]:special[1]]])
		text = text[special[1]:]
	}
	return ids
}
