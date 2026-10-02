//go:build goexperiment.simd

package tests

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"simd"
	"sync/atomic"
	"testing"

	. "Stream-PT/forward"
	"Stream-PT/ggufindex"
	ggufmmap "Stream-PT/ggufmap"
)

func q40Fixture(tb testing.TB, input, output int) (*ggufmmap.Reader, ggufindex.Tensor, []byte) {
	tb.Helper()
	rowBytes := input / q40Elements * q40BlockBytes
	data := make([]byte, rowBytes*output)
	for block := 0; block < len(data)/q40BlockBytes; block++ {
		b := data[block*q40BlockBytes : (block+1)*q40BlockBytes]
		scales := [...]uint16{0x3c00, 0xb800, 0x3400, 0x0001}
		binary.LittleEndian.PutUint16(b, scales[block%len(scales)])
		for i := 2; i < len(b); i++ {
			b[i] = byte(block*13 + i*7)
		}
	}
	path := filepath.Join(tb.TempDir(), "weights.bin")
	// Exercise a tensor offset that is not page aligned.
	const prefix = 37
	if err := os.WriteFile(path, append(make([]byte, prefix), data...), 0600); err != nil {
		tb.Fatal(err)
	}
	reader, err := ggufmmap.Open(&ggufindex.Model{Paths: []string{path}})
	if err != nil {
		tb.Fatal(err)
	}
	tb.Cleanup(func() {
		if err := reader.Close(); err != nil {
			tb.Error(err)
		}
	})
	return reader, ggufindex.Tensor{
		Name: "test.weight", Type: 2, Shape: []uint64{uint64(input), uint64(output)},
		Range: ggufindex.Range{File: path, Start: prefix, End: prefix + uint64(len(data))},
	}, data
}

func BenchmarkMulQ40(b *testing.B) {
	reader, tensor, _ := q40Fixture(b, 512, 256)
	x := make([]float32, 512)
	for i := range x {
		x[i] = float32(i%17-8) / 16
	}
	b.ReportAllocs()
	b.SetBytes(int64(tensor.Range.End - tensor.Range.Start))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := MulQ40(context.Background(), reader, tensor, x, 1); err != nil {
			b.Fatal(err)
		}
	}
}

func TestMulQ40StreamingBatch(t *testing.T) {
	const input, output = 96, 11
	reader, tensor, data := q40Fixture(t, input, output)
	rowBytes := input / q40Elements * q40BlockBytes
	var vector simd.Float32s
	for _, batch := range []int{1, 3, vector.Len() + 1, 19} {
		for _, workers := range []int{0, 1, 3, 100} {
			for _, windowRows := range []int{1, 4, output} {
				t.Run(fmt.Sprintf("batch%d/workers%d/rows%d", batch, workers, windowRows), func(t *testing.T) {
					x := make([]float32, batch*input)
					y := make([]float32, batch*output)
					want := make([]float32, len(y))
					for i := range x {
						x[i] = float32(i%23-11) / 32
					}
					original := append([]float32(nil), x...)
					for item := 0; item < batch; item++ {
						for row := 0; row < output; row++ {
							want[item*output+row] = referenceQ40Dot(data[row*rowBytes:(row+1)*rowBytes], x[item*input:(item+1)*input])
						}
					}
					options := Q40Options{Workers: workers, WindowBytes: uint64(windowRows*rowBytes + 7)}
					if err := MulQ40BatchInto(context.Background(), reader, tensor, x, y, batch, options); err != nil {
						t.Fatal(err)
					}
					compareVectors(t, y, want)
					compareVectors(t, x, original)
					clear(y)
					if err := MulQ40BatchInto(context.Background(), reader, tensor, x, y, batch, options); err != nil {
						t.Fatal(err)
					}
					compareVectors(t, y, want)
				})
			}
		}
	}
}

func TestMulQ40Validation(t *testing.T) {
	reader, tensor, _ := q40Fixture(t, 64, 5)
	x, y := make([]float32, 64), make([]float32, 5)
	ctx := context.Background()
	tests := []struct {
		name string
		run  func() error
	}{
		{"nil context", func() error { return MulQ40Into(nil, reader, tensor, x, y, Q40Options{}) }},
		{"nil reader", func() error { return MulQ40Into(ctx, nil, tensor, x, y, Q40Options{}) }},
		{"input length", func() error { return MulQ40Into(ctx, reader, tensor, x[1:], y, Q40Options{}) }},
		{"output length", func() error { return MulQ40Into(ctx, reader, tensor, x, y[1:], Q40Options{}) }},
		{"overlap", func() error { return MulQ40Into(ctx, reader, tensor, x, x[3:8], Q40Options{}) }},
		{"zero batch", func() error { return MulQ40BatchInto(ctx, reader, tensor, x, y, 0, Q40Options{}) }},
		{"overflow batch", func() error { return MulQ40BatchInto(ctx, reader, tensor, x, y, int(^uint(0)>>1), Q40Options{}) }},
		{"small window", func() error { return MulQ40Into(ctx, reader, tensor, x, y, Q40Options{WindowBytes: 35}) }},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := tt.run(); err == nil {
				t.Fatal("expected an error")
			}
		})
	}
	for _, mutate := range []func(*ggufindex.Tensor){
		func(t *ggufindex.Tensor) { t.Type = 1 },
		func(t *ggufindex.Tensor) { t.Shape = []uint64{64} },
		func(t *ggufindex.Tensor) { t.Shape = []uint64{63, 5} },
		func(t *ggufindex.Tensor) { t.Shape = []uint64{64, 0} },
		func(t *ggufindex.Tensor) { t.Shape = []uint64{^uint64(0), 5} },
		func(t *ggufindex.Tensor) { t.Range.End-- },
		func(t *ggufindex.Tensor) { t.Range.Start = t.Range.End },
	} {
		bad := tensor
		mutate(&bad)
		if err := MulQ40Into(ctx, reader, bad, x, y, Q40Options{}); err == nil {
			t.Fatalf("expected an error for %+v", bad)
		}
	}
	cancelled, cancel := context.WithCancel(ctx)
	cancel()
	if err := MulQ40Into(cancelled, reader, tensor, x, y, Q40Options{}); !errors.Is(err, context.Canceled) {
		t.Fatalf("expected cancellation, got %v", err)
	}
}

type cancelAfterChecks struct {
	context.Context
	cancel context.CancelFunc
	checks atomic.Int32
	limit  int32
}

func (c *cancelAfterChecks) Err() error {
	if c.checks.Add(1) >= c.limit {
		c.cancel()
	}
	return c.Context.Err()
}

func TestMulQ40CancellationDuringStreaming(t *testing.T) {
	reader, tensor, _ := q40Fixture(t, 64, 11)
	x, y := make([]float32, 64), make([]float32, 11)
	for i := range x {
		x[i] = 1
	}
	for i := range y {
		y[i] = 123456
	}
	base, cancel := context.WithCancel(context.Background())
	defer cancel()
	ctx := &cancelAfterChecks{Context: base, cancel: cancel, limit: 5}
	options := Q40Options{Workers: 1, WindowBytes: 3 * 2 * q40BlockBytes}
	if err := MulQ40Into(ctx, reader, tensor, x, y, options); !errors.Is(err, context.Canceled) {
		t.Fatalf("expected cancellation during streaming, got %v", err)
	}
	if y[0] == 123456 || y[len(y)-1] != 123456 {
		t.Fatalf("expected partially completed output, got %v", y)
	}
	// The reader remains usable after a cancelled window has been released.
	if err := MulQ40Into(context.Background(), reader, tensor, x, y, options); err != nil {
		t.Fatal(err)
	}
}

func BenchmarkMulQ40BatchInto(b *testing.B) {
	reader, tensor, _ := q40Fixture(b, 512, 256)
	for _, batch := range []int{1, 4, 8} {
		b.Run(fmt.Sprintf("batch%d", batch), func(b *testing.B) {
			x, y := make([]float32, batch*512), make([]float32, batch*256)
			for i := range x {
				x[i] = float32(i%17-8) / 16
			}
			b.ReportAllocs()
			b.SetBytes(int64(tensor.Range.End - tensor.Range.Start))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if err := MulQ40BatchInto(context.Background(), reader, tensor, x, y, batch, Q40Options{}); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
