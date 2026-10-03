package tests

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"Stream-PT/ggufindex"
)

type indexOptimizationEntry struct {
	key   string
	typ   uint32
	value []byte
}

func indexOptimizationString(value string) []byte {
	return append(binary.LittleEndian.AppendUint64(nil, uint64(len(value))), value...)
}

func indexOptimizationArray(typ uint32, count uint64, payload []byte) []byte {
	data := binary.LittleEndian.AppendUint32(nil, typ)
	data = binary.LittleEndian.AppendUint64(data, count)
	return append(data, payload...)
}

func indexOptimizationFixture(t testing.TB, entries ...indexOptimizationEntry) (string, []byte) {
	t.Helper()
	data := append([]byte("GGUF"), binary.LittleEndian.AppendUint32(nil, 3)...)
	data = binary.LittleEndian.AppendUint64(data, 0)
	data = binary.LittleEndian.AppendUint64(data, uint64(len(entries)+1))
	entries = append([]indexOptimizationEntry{{
		key: "general.alignment", typ: 4, value: binary.LittleEndian.AppendUint32(nil, 1),
	}}, entries...)
	for _, entry := range entries {
		data = append(data, indexOptimizationString(entry.key)...)
		data = binary.LittleEndian.AppendUint32(data, entry.typ)
		data = append(data, entry.value...)
	}
	path := filepath.Join(t.TempDir(), "index.gguf")
	if err := os.WriteFile(path, data, 0600); err != nil {
		t.Fatal(err)
	}
	return path, data
}

func TestIndexOptimizationFixedArrays(t *testing.T) {
	widths := map[uint32]int{0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
	for typ, width := range widths {
		for _, count := range []uint64{0, 3} {
			for _, trailing := range []bool{false, true} {
				t.Run(fmt.Sprintf("type%d/count%d/trailing%t", typ, count, trailing), func(t *testing.T) {
					value := indexOptimizationArray(typ, count, bytes.Repeat([]byte{1}, int(count)*width))
					entries := []indexOptimizationEntry{{key: "array", typ: 9, value: value}}
					if trailing {
						entries = append(entries, indexOptimizationEntry{key: "sentinel", typ: 8, value: indexOptimizationString("after")})
					}
					path, data := indexOptimizationFixture(t, entries...)
					model, err := ggufindex.Open([]string{path})
					if err != nil {
						t.Fatal(err)
					}
					entry := model.Metadata["array"]
					if entry.Type != 9 || entry.Range.File != path || entry.Range.End-entry.Range.Start != uint64(len(value)) ||
						!bytes.Equal(data[entry.Range.Start:entry.Range.End], value) {
						t.Fatalf("incorrect array range: %+v", entry)
					}
					if !trailing && entry.Range.End != uint64(len(data)) {
						t.Fatal("array does not end at EOF")
					}
					if trailing {
						var decoded bytes.Buffer
						if err := model.Metadata["sentinel"].WriteJSON(&decoded); err != nil || decoded.String() != `"after"` {
							t.Fatalf("sentinel: %s, %v", decoded.String(), err)
						}
					}
				})
			}
		}
	}
}

func TestIndexOptimizationVariableArrays(t *testing.T) {
	stringsValue := indexOptimizationArray(8, 3, append(append(indexOptimizationString(""), indexOptimizationString("hello")...), indexOptimizationString("ä")...))
	fixedValue := indexOptimizationArray(2, 2, []byte{1, 0, 2, 0})
	nestedValue := indexOptimizationArray(9, 3, append(append(append([]byte(nil), stringsValue...), fixedValue...), indexOptimizationArray(9, 0, nil)...))
	for _, tc := range []struct {
		name  string
		value []byte
		json  string
	}{
		{"strings", stringsValue, `["","hello","ä"]`},
		{"nested", nestedValue, `[["","hello","ä"],[1,2],[]]`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			path, data := indexOptimizationFixture(t, indexOptimizationEntry{key: "array", typ: 9, value: tc.value})
			model, err := ggufindex.Open([]string{path})
			if err != nil {
				t.Fatal(err)
			}
			entry := model.Metadata["array"]
			if entry.Range.End != uint64(len(data)) || entry.Range.End-entry.Range.Start != uint64(len(tc.value)) {
				t.Fatalf("incorrect range: %+v", entry.Range)
			}
			var decoded bytes.Buffer
			if err := entry.WriteJSON(&decoded); err != nil || decoded.String() != tc.json {
				t.Fatalf("decoded: %s, %v; want %s", decoded.String(), err, tc.json)
			}
		})
	}
}

func TestIndexOptimizationRejectsMalformedArrays(t *testing.T) {
	cases := []struct {
		name  string
		value []byte
		error string
	}{
		{"unknown-empty", indexOptimizationArray(13, 0, nil), "unknown metadata type"},
		{"unknown", indexOptimizationArray(math.MaxUint32, 1, nil), "unknown metadata type"},
		{"count-limit", indexOptimizationArray(0, 100_000_001, nil), "array is too large"},
		{"uint64-overflow", indexOptimizationArray(10, math.MaxUint64, nil), "array is too large"},
		{"int64-overflow", indexOptimizationArray(10, uint64(math.MaxInt64)/8+1, nil), "array is too large"},
		{"missing-type", nil, "EOF"},
		{"missing-count", []byte{0, 0, 0, 0, 1}, "EOF"},
		{"string-truncated", indexOptimizationArray(8, 1, append(binary.LittleEndian.AppendUint64(nil, 2), 'x')), "EOF"},
		{"string-too-long", indexOptimizationArray(8, 1, binary.LittleEndian.AppendUint64(nil, 1<<20+1)), "too long"},
		{"nested-truncated", indexOptimizationArray(9, 1, indexOptimizationArray(10, 1, []byte{1})), "EOF"},
		{"nested-unknown-empty", indexOptimizationArray(9, 1, indexOptimizationArray(13, 0, nil)), "unknown metadata type"},
	}
	for typ, width := range map[uint32]int{0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8} {
		cases = append(cases, struct {
			name  string
			value []byte
			error string
		}{fmt.Sprintf("type%d-short-by-one", typ), indexOptimizationArray(typ, 2, make([]byte, 2*width-1)), "EOF"})
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			path, _ := indexOptimizationFixture(t, indexOptimizationEntry{key: "array", typ: 9, value: tc.value})
			if _, err := ggufindex.Open([]string{path}); err == nil || !strings.Contains(err.Error(), tc.error) {
				t.Fatalf("got %v; want error containing %q", err, tc.error)
			}
		})
	}
}

func TestIndexOptimizationNestingLimit(t *testing.T) {
	for _, empty := range []bool{false, true} {
		for _, wrappers := range []int{7, 8, 9} {
			t.Run(fmt.Sprintf("empty%t/wrappers%d", empty, wrappers), func(t *testing.T) {
				value := indexOptimizationArray(0, 1, []byte{1})
				if empty {
					value = indexOptimizationArray(0, 0, nil)
				}
				for range wrappers {
					value = indexOptimizationArray(9, 1, value)
				}
				path, _ := indexOptimizationFixture(t, indexOptimizationEntry{key: "array", typ: 9, value: value})
				_, err := ggufindex.Open([]string{path})
				valid := wrappers < 8 || empty && wrappers == 8
				if valid && err != nil {
					t.Fatal(err)
				}
				if !valid && (err == nil || !strings.Contains(err.Error(), "nesting is too deep")) {
					t.Fatalf("got %v; want nesting error", err)
				}
			})
		}
	}
}

func BenchmarkIndexOptimizationFixedArray(b *testing.B) {
	for _, count := range []int{1_024, 1_048_576} {
		b.Run(fmt.Sprintf("uint64/%d", count), func(b *testing.B) {
			value := indexOptimizationArray(10, uint64(count), make([]byte, count*8))
			path, _ := indexOptimizationFixture(b, indexOptimizationEntry{key: "array", typ: 9, value: value})
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, err := ggufindex.Open([]string{path}); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
