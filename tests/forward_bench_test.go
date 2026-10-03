//go:build goexperiment.simd

package tests

import (
	"context"
	"encoding/binary"
	"fmt"
	"os"
	"path/filepath"
	"simd"
	"testing"

	"Stream-PT/forward"
	"Stream-PT/ggufindex"
	ggufmmap "Stream-PT/ggufmap"
)

func BenchmarkForwardQuantized(b *testing.B) {
	b.Logf("SIMD: %d bits, emulated=%v", simd.VectorBitSize(), simd.Emulated())
	const input, output = 2880, 256
	for _, typ := range []uint32{2, 8, 39} {
		b.Run(fmt.Sprintf("type%d", typ), func(b *testing.B) {
			blockBytes := map[uint32]int{2: 18, 8: 34, 39: 17}[typ]
			data := make([]byte, input/32*output*blockBytes)
			for block := 0; block < len(data)/blockBytes; block++ {
				encoded := data[block*blockBytes : (block+1)*blockBytes]
				start := 2
				if typ == 39 {
					encoded[0] = byte(124 + block%5)
					start = 1
				} else {
					binary.LittleEndian.PutUint16(encoded, 0x3800)
				}
				for i := start; i < blockBytes; i++ {
					encoded[i] = byte(block*13 + i*7)
				}
			}
			path := filepath.Join(b.TempDir(), "weights.bin")
			if err := os.WriteFile(path, data, 0600); err != nil {
				b.Fatal(err)
			}
			reader, err := ggufmmap.Open(&ggufindex.Model{Paths: []string{path}})
			if err != nil {
				b.Fatal(err)
			}
			defer reader.Close()
			shape := []uint64{input, output}
			if typ == 39 {
				shape = append(shape, 1)
			}
			tensor := ggufindex.Tensor{Type: typ, Shape: shape,
				Range: ggufindex.Range{File: path, End: uint64(len(data))}}
			x, y := make([]float32, input), make([]float32, output)
			for i := range x {
				x[i] = float32(i%23-11) / 32
			}
			ctx := context.Background()
			b.ReportAllocs()
			b.SetBytes(int64(len(data)))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				switch typ {
				case 2:
					err = forward.MulQ40Into(ctx, reader, tensor, x, y, forward.Q40Options{Workers: 1})
				case 8:
					err = forward.MulQ80Into(ctx, reader, tensor, x, y, forward.Q40Options{Workers: 1})
				case 39:
					err = forward.MulMXFP4Expert(ctx, reader, tensor, 0, x, nil, y, forward.Q40Options{Workers: 1})
				}
				if err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func BenchmarkForwardAttention(b *testing.B) {
	for _, window := range []int{0, 128} {
		b.Run(fmt.Sprintf("window%d", window), func(b *testing.B) {
			const heads, kvHeads, dim, positions = 64, 8, 64, 512
			cache := forward.NewKVCache(1, positions, kvHeads, dim)
			q, k, v := make([]float32, heads*dim), make([]float32, kvHeads*dim), make([]float32, kvHeads*dim)
			out := make([]float32, len(q))
			for i := range q {
				q[i] = float32(i%17-8) / 16
			}
			for p := range positions {
				for i := range k {
					cache.Keys[0][p][i] = float32((i+p)%19-9) / 16
					cache.Values[0][p][i] = float32((i+p)%23-11) / 16
				}
			}
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				forward.ForwardAttention(q, k, v, cache, 0, positions-1, heads, kvHeads, dim, out,
					forward.AttentionOptions{SlidingWindow: window})
			}
		})
	}
}
