//go:build goexperiment.simd

package tests

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"simd"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"Stream-PT/forward"
	"Stream-PT/ggufindex"
	ggufmmap "Stream-PT/ggufmap"
)

func quantOptimizationFixture(tb testing.TB, typ uint32, input, output int) (*ggufmmap.Reader, ggufindex.Tensor, []byte) {
	tb.Helper()
	blockBytes := 18
	if typ == 8 {
		blockBytes = 34
	}
	data := make([]byte, input/32*blockBytes*output)
	scales := [...]uint16{0x3c00, 0xb800, 0x3400, 0x0001, 0x03ff, 0x0400, 0x7bff, 0x8000}
	for block := 0; block < len(data)/blockBytes; block++ {
		b := data[block*blockBytes : (block+1)*blockBytes]
		binary.LittleEndian.PutUint16(b, scales[block%len(scales)])
		for i := 2; i < blockBytes; i++ {
			b[i] = byte(block*13 + i*7)
		}
	}
	path := filepath.Join(tb.TempDir(), "quant.bin")
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
	return reader, ggufindex.Tensor{Type: typ, Shape: []uint64{uint64(input), uint64(output)},
		Range: ggufindex.Range{File: path, Start: prefix, End: prefix + uint64(len(data))}}, data
}

func quantOptimizationReplaceData(t *testing.T, tensor ggufindex.Tensor, data []byte) {
	t.Helper()
	if err := os.WriteFile(tensor.Range.File, append(make([]byte, int(tensor.Range.Start)), data...), 0600); err != nil {
		t.Fatal(err)
	}
}

func TestQuantOptimizationDispatch(t *testing.T) {
	t.Logf("SIMD bits=%d, emulated=%v", simd.VectorBitSize(), simd.Emulated())
}

func TestQuantOptimizationFloat16Scales(t *testing.T) {
	const output = 1 << 16
	x := make([]float32, 32)
	for i := range x {
		x[i] = 1
	}
	for _, typ := range []uint32{2, 8} {
		t.Run(fmt.Sprintf("type%d", typ), func(t *testing.T) {
			reader, tensor, data := quantOptimizationFixture(t, typ, 32, output)
			rowBytes, weight := len(data)/output, byte(0x99)
			if typ == 8 {
				weight = 1
			}
			for row := range output {
				encoded := data[row*rowBytes : (row+1)*rowBytes]
				binary.LittleEndian.PutUint16(encoded, uint16(row))
				for i := 2; i < rowBytes; i++ {
					encoded[i] = weight
				}
			}
			quantOptimizationReplaceData(t, tensor, data)
			y := make([]float32, output)
			opts := forward.Q40Options{Workers: 4, WindowBytes: uint64(rowBytes * 257)}
			var err error
			if typ == 2 {
				err = forward.MulQ40Into(context.Background(), reader, tensor, x, y, opts)
			} else {
				err = forward.MulQ80Into(context.Background(), reader, tensor, x, y, opts)
			}
			if err != nil {
				t.Fatal(err)
			}
			for row, got := range y {
				want := referenceFloat16(uint16(row)) * 32
				if math.IsNaN(float64(want)) {
					if !math.IsNaN(float64(got)) {
						t.Fatalf("scale %#04x: got %g, want NaN", row, got)
					}
				} else if got != want {
					t.Fatalf("scale %#04x: got %g, want %g", row, got, want)
				}
			}
		})
	}
}

func referenceQ80Optimization(row []byte, x []float32) float32 {
	var sum float32
	for block := 0; block < len(row)/34; block++ {
		b := row[block*34 : (block+1)*34]
		scale := referenceFloat16(binary.LittleEndian.Uint16(b))
		for i, q := range b[2:] {
			sum += scale * float32(int8(q)) * x[block*32+i]
		}
	}
	return sum
}

func TestQuantOptimizationNumerics(t *testing.T) {
	for _, typ := range []uint32{2, 8} {
		for _, input := range []int{32, 96, 512} {
			t.Run(fmt.Sprintf("type%d/input%d", typ, input), func(t *testing.T) {
				const output = 17
				reader, tensor, data := quantOptimizationFixture(t, typ, input, output)
				rowBytes := len(data) / output
				x, want := make([]float32, input), make([]float32, output)
				for i := range x {
					x[i] = float32(i%23-11) / 32
				}
				for row := range want {
					encoded := data[row*rowBytes : (row+1)*rowBytes]
					if typ == 2 {
						want[row] = referenceQ40Dot(encoded, x)
					} else {
						want[row] = referenceQ80Optimization(encoded, x)
					}
				}
				for _, workers := range []int{1, 4, 100} {
					for _, rows := range []int{1, 4, 8, output} {
						y := make([]float32, output)
						opts := forward.Q40Options{Workers: workers, WindowBytes: uint64(rows*rowBytes + 1)}
						var err error
						if typ == 2 {
							err = forward.MulQ40Into(context.Background(), reader, tensor, x, y, opts)
						} else {
							err = forward.MulQ80Into(context.Background(), reader, tensor, x, y, opts)
						}
						if err != nil {
							t.Fatal(err)
						}
						compareVectors(t, y, want)
						if typ == 8 {
							token, logit, err := forward.MulQ80Argmax(context.Background(), reader, tensor, x, opts)
							best := 0
							for i := range y {
								if y[i] > y[best] {
									best = i
								}
							}
							if err != nil || token != best || logit != y[best] {
								t.Fatalf("argmax = %d/%g/%v, want %d/%g", token, logit, err, best, y[best])
							}
						}
					}
				}
			})
		}
	}
}

func TestQuantOptimizationArgmaxWindowAndTies(t *testing.T) {
	reader, tensor, data := quantOptimizationFixture(t, 8, 32, 17)
	x := make([]float32, 32)
	for _, size := range []uint64{1, 33} {
		if _, _, err := forward.MulQ80Argmax(context.Background(), reader, tensor, x,
			forward.Q40Options{Workers: 4, WindowBytes: size}); err == nil || !strings.Contains(err.Error(), "cannot accommodate") {
			t.Fatalf("%d-byte window: expected row-size validation, got %v", size, err)
		}
	}
	for _, best := range []int{0, 3} {
		if best == 3 {
			clear(data)
			for row := range 17 {
				binary.LittleEndian.PutUint16(data[row*34:], 0x3c00)
			}
			for _, row := range []int{3, 4, 8, 16} {
				data[row*34+2] = 1
			}
			quantOptimizationReplaceData(t, tensor, data)
			x[0] = 7
		}
		wantLogit := x[0]
		for _, workers := range []int{1, 4, 100} {
			for _, rows := range []int{1, 4, 8, 17} {
				for range 5 {
					token, logit, err := forward.MulQ80Argmax(context.Background(), reader, tensor, x,
						forward.Q40Options{Workers: workers, WindowBytes: uint64(rows * 34)})
					if err != nil || token != best || logit != wantLogit {
						t.Fatalf("ties: workers=%d rows=%d got %d/%g/%v, want %d/%g", workers, rows, token, logit, err, best, wantLogit)
					}
				}
			}
		}
	}
}

type quantRecordingContext struct {
	context.Context
	mu  sync.Mutex
	ids map[string]bool
}

func (c *quantRecordingContext) Err() error {
	var stack [64]byte
	n := runtime.Stack(stack[:], false)
	id := strings.Fields(string(stack[:n]))[1]
	c.mu.Lock()
	c.ids[id] = true
	c.mu.Unlock()
	return c.Context.Err()
}

func quantOptimizationRun(ctx context.Context, reader *ggufmmap.Reader, tensor ggufindex.Tensor,
	x, y []float32, mode string, opts forward.Q40Options) error {
	switch mode {
	case "q40":
		return forward.MulQ40Into(ctx, reader, tensor, x, y, opts)
	case "q40batch":
		return forward.MulQ40BatchInto(ctx, reader, tensor, x, y, 4, opts)
	case "q80":
		return forward.MulQ80Into(ctx, reader, tensor, x, y, opts)
	default:
		_, _, err := forward.MulQ80Argmax(ctx, reader, tensor, x, opts)
		return err
	}
}

func TestQuantOptimizationPersistentWorkers(t *testing.T) {
	previous := runtime.GOMAXPROCS(4)
	defer runtime.GOMAXPROCS(previous)
	for _, mode := range []string{"q40", "q40batch", "q80", "argmax"} {
		t.Run(mode, func(t *testing.T) {
			typ, batch, blockBytes := uint32(2), 1, 18
			if mode == "q80" || mode == "argmax" {
				typ, blockBytes = 8, 34
			}
			if mode == "q40batch" {
				batch = 4
			}
			reader, tensor, _ := quantOptimizationFixture(t, typ, 32, 33)
			ctx := &quantRecordingContext{Context: context.Background(), ids: make(map[string]bool)}
			if err := quantOptimizationRun(ctx, reader, tensor, make([]float32, batch*32), make([]float32, batch*33), mode,
				forward.Q40Options{Workers: 4, WindowBytes: uint64(4 * blockBytes)}); err != nil {
				t.Fatal(err)
			}
			if len(ctx.ids) != 5 {
				t.Fatalf("used %d goroutines, want caller plus 4 persistent workers", len(ctx.ids))
			}
		})
	}
}

type quantBlockingContext struct {
	context.Context
	checks  atomic.Int32
	entered chan struct{}
	release chan struct{}
}

func (c *quantBlockingContext) Err() error {
	if c.checks.Add(1) > 1 {
		c.entered <- struct{}{}
		<-c.release
	}
	return c.Context.Err()
}

func TestQuantOptimizationCancellationJoins(t *testing.T) {
	previous := runtime.GOMAXPROCS(4)
	defer runtime.GOMAXPROCS(previous)
	for _, mode := range []string{"q40", "q40batch", "q80", "argmax"} {
		t.Run(mode, func(t *testing.T) {
			typ, batch, blockBytes := uint32(2), 1, 18
			if mode == "q80" || mode == "argmax" {
				typ, blockBytes = 8, 34
			}
			if mode == "q40batch" {
				batch = 4
			}
			reader, tensor, _ := quantOptimizationFixture(t, typ, 32, 17)
			base, cancel := context.WithCancel(context.Background())
			defer cancel()
			ctx := &quantBlockingContext{Context: base, entered: make(chan struct{}, 64), release: make(chan struct{})}
			x, y := make([]float32, batch*32), make([]float32, batch*17)
			opts := forward.Q40Options{Workers: 4, WindowBytes: uint64(4 * blockBytes)}
			result := make(chan error, 1)
			go func() { result <- quantOptimizationRun(ctx, reader, tensor, x, y, mode, opts) }()
			for range 4 {
				select {
				case <-ctx.entered:
				case <-time.After(5 * time.Second):
					close(ctx.release)
					t.Fatal("workers did not reach the mapped window")
				}
			}
			cancel()
			select {
			case err := <-result:
				close(ctx.release)
				t.Fatalf("returned before worker join: %v", err)
			default:
			}
			close(ctx.release)
			if err := <-result; !errors.Is(err, context.Canceled) {
				t.Fatalf("want cancellation, got %v", err)
			}
			if err := quantOptimizationRun(context.Background(), reader, tensor, x, y, mode, opts); err != nil {
				t.Fatalf("reader reuse after cancellation: %v", err)
			}
			bad := tensor
			bad.Range.File += ".missing"
			if err := quantOptimizationRun(context.Background(), reader, bad, x, y, mode, opts); err == nil {
				t.Fatal("expected mapping error")
			}
		})
	}
}

func BenchmarkQuantOptimization(b *testing.B) {
	const input, output = 512, 1024
	for _, mode := range []string{"q40", "q40batch", "q80", "argmax"} {
		typ, batch, blockBytes := uint32(2), 1, 18
		if mode == "q80" || mode == "argmax" {
			typ, blockBytes = 8, 34
		}
		if mode == "q40batch" {
			batch = 4
		}
		reader, tensor, _ := quantOptimizationFixture(b, typ, input, output)
		x, y := make([]float32, batch*input), make([]float32, batch*output)
		for i := range x {
			x[i] = float32(i%17-8) / 16
		}
		for _, workers := range []int{1, 4} {
			for _, rows := range []int{4, 32, output} {
				b.Run(fmt.Sprintf("%s/workers%d/rows%d", mode, workers, rows), func(b *testing.B) {
					opts := forward.Q40Options{Workers: workers, WindowBytes: uint64(rows * input / 32 * blockBytes)}
					b.ReportAllocs()
					b.SetBytes(int64(tensor.Range.End - tensor.Range.Start))
					b.ResetTimer()
					for i := 0; i < b.N; i++ {
						if err := quantOptimizationRun(context.Background(), reader, tensor, x, y, mode, opts); err != nil {
							b.Fatal(err)
						}
					}
				})
			}
		}
	}
}

func TestQuantOptimizationQ80ShapeOverflow(t *testing.T) {
	reader, tensor, _ := quantOptimizationFixture(t, 8, 32, 1)
	tensor.Shape = []uint64{32, math.MaxUint64}
	if _, _, err := forward.MulQ80Argmax(context.Background(), reader, tensor, make([]float32, 32), forward.Q40Options{}); err == nil {
		t.Fatal("accepted overflowing output shape")
	}
}
