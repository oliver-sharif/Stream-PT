//go:build goexperiment.simd

package tests

import (
	"bytes"
	"context"
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"

	"Stream-PT/forward"
	"Stream-PT/ggufindex"
	ggufmmap "Stream-PT/ggufmap"
)

const (
	q40Elements     = 32
	q40BlockBytes   = 18
	mxfp4BlockBytes = 17
)

func tensorFixture(t *testing.T, typ uint32, shape []uint64, data []byte) (*ggufmmap.Reader, ggufindex.Tensor) {
	t.Helper()
	path := filepath.Join(t.TempDir(), "weights.bin")
	if err := os.WriteFile(path, data, 0600); err != nil {
		t.Fatal(err)
	}
	reader, err := ggufmmap.Open(&ggufindex.Model{Paths: []string{path}})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := reader.Close(); err != nil {
			t.Error(err)
		}
	})
	return reader, ggufindex.Tensor{Type: typ, Shape: shape,
		Range: ggufindex.Range{File: path, End: uint64(len(data))}}
}

func multiplyQ40Row(t *testing.T, row []byte, x []float32) float32 {
	t.Helper()
	reader, tensor := tensorFixture(t, 2, []uint64{uint64(len(x)), 1}, row)
	y, err := forward.MulQ40(context.Background(), reader, tensor, x, 1)
	if err != nil {
		t.Fatal(err)
	}
	return y[0]
}

func tokenizerFixture(t *testing.T, tokens []string) *forward.Tokenizer {
	t.Helper()
	var data bytes.Buffer
	write := func(value any) {
		if err := binary.Write(&data, binary.LittleEndian, value); err != nil {
			t.Fatal(err)
		}
	}
	write(uint32(8))
	write(uint64(len(tokens)))
	for _, token := range tokens {
		write(uint64(len(token)))
		data.WriteString(token)
	}
	path := filepath.Join(t.TempDir(), "tokens.bin")
	if err := os.WriteFile(path, data.Bytes(), 0600); err != nil {
		t.Fatal(err)
	}
	tok, err := forward.LoadTokenizer(&ggufindex.Model{Metadata: map[string]ggufindex.MetadataValue{
		"tokenizer.ggml.tokens": {Type: 9, Range: ggufindex.Range{File: path, End: uint64(data.Len())}},
	}})
	if err != nil {
		t.Fatal(err)
	}
	return tok
}

func referenceFloat16(bits uint16) float32 {
	sign := float64(1)
	if bits&0x8000 != 0 {
		sign = -1
	}
	exponent, fraction := int(bits>>10&31), int(bits&1023)
	switch exponent {
	case 0:
		return float32(sign * math.Ldexp(float64(fraction), -24))
	case 31:
		if fraction != 0 {
			return float32(math.NaN())
		}
		return float32(math.Inf(int(sign)))
	default:
		return float32(sign * math.Ldexp(float64(1024+fraction), exponent-25))
	}
}
