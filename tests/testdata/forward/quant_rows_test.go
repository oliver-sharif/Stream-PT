//go:build goexperiment.simd

package forward

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"runtime"
	"testing"
	"time"
)

var quantRowsSink [2]float32

func quantRowsData(input, rows int, q40 bool) ([]byte, []float32) {
	blockBytes := q80BlockBytes
	if q40 {
		blockBytes = q40BlockBytes
	}
	data := make([]byte, input/32*blockBytes*rows)
	x := make([]float32, input)
	state := uint32(12345)
	next := func() uint32 {
		state = state*1664525 + 1013904223
		return state
	}
	for i := range x {
		x[i] = float32(int32(next()>>16)-32768) / 32768
	}
	scales := [...]uint16{0x3400, 0xb800, 0x3c00, 0xbc00}
	for block := 0; block < len(data)/blockBytes; block++ {
		encoded := data[block*blockBytes : (block+1)*blockBytes]
		binary.LittleEndian.PutUint16(encoded, scales[block%len(scales)])
		for i := 2; i < blockBytes; i++ {
			encoded[i] = byte(next() >> 24)
		}
	}
	return data, x
}

func BenchmarkQuantRows(b *testing.B) {
	for _, q40 := range []bool{false, true} {
		for _, input := range []int{32, 96, 512, 4096} {
			data, x := quantRowsData(input, 2, q40)
			rowBytes := len(data) / 2
			dot := dotQ80
			typ := "Q8"
			if q40 {
				dot, typ = dotQ40, "Q4"
			}
			b.Run(fmt.Sprintf("%s/%d/baseline", typ, input), func(b *testing.B) {
				b.ReportAllocs()
				b.SetBytes(int64(len(data)))
				for b.Loop() {
					quantRowsSink = [2]float32{dot(data[:rowBytes], x), dot(data[rowBytes:], x)}
				}
			})
			if !q40 {
				b.Run(fmt.Sprintf("%s/%d/pair", typ, input), func(b *testing.B) {
					b.ReportAllocs()
					b.SetBytes(int64(len(data)))
					for b.Loop() {
						quantRowsSink[0], quantRowsSink[1] = quantDotQ80Pair(data, x)
					}
				})
			}
		}
	}
}

func BenchmarkQuantRowsStreaming(b *testing.B) {
	const output = 1024
	for _, input := range []int{512, 4096} {
		data, x := quantRowsData(input, output, false)
		rowBytes := len(data) / output
		b.Run(fmt.Sprintf("Q8/%d/baseline", input), func(b *testing.B) {
			b.ReportAllocs()
			b.SetBytes(int64(len(data)))
			for b.Loop() {
				for row := 0; row < output; row += 2 {
					quantRowsSink[0] = dotQ80(data[row*rowBytes:(row+1)*rowBytes], x)
					quantRowsSink[1] = dotQ80(data[(row+1)*rowBytes:(row+2)*rowBytes], x)
				}
			}
		})
		b.Run(fmt.Sprintf("Q8/%d/pair", input), func(b *testing.B) {
			b.ReportAllocs()
			b.SetBytes(int64(len(data)))
			for b.Loop() {
				for row := 0; row < output; row += 2 {
					quantRowsSink[0], quantRowsSink[1] = quantDotQ80Pair(data[row*rowBytes:(row+2)*rowBytes], x)
				}
			}
		})
	}
}

func TestQuantRowsPairNumerics(t *testing.T) {
	for _, input := range []int{32, 64, 96, 512, 4096} {
		for _, extreme := range []bool{false, true} {
			data, x := quantRowsData(input, 2, false)
			if extreme {
				scales := [...]uint16{0x0001, 0x8001, 0x03ff, 0x83ff, 0x0400, 0x8400, 0x7bff, 0xfbff, 0, 0x8000}
				for block := 0; block < len(data)/34; block++ {
					binary.LittleEndian.PutUint16(data[block*34:], scales[block%len(scales)])
				}
			}
			a, b := quantDotQ80Pair(data, x)
			c, d := quantQ80PairPortable(data, x)
			for row, got := range []float32{a, b} {
				encoded := data[row*len(data)/2 : (row+1)*len(data)/2]
				baseline := []float32{c, d}[row]
				if math.Float32bits(got) != math.Float32bits(baseline) {
					t.Fatalf("input=%d extreme=%v row=%d: pair %g baseline %g", input, extreme, row, got, baseline)
				}
				want := quantDotScalar(encoded, x, false)
				var magnitude float64
				for block := 0; block < len(encoded)/34; block++ {
					scale := float16(binary.LittleEndian.Uint16(encoded[block*34:]))
					for i, q := range encoded[block*34+2 : (block+1)*34] {
						magnitude += math.Abs(float64(float32(int8(q)) * scale * x[block*32+i]))
					}
				}
				if math.Abs(float64(got)-float64(want)) > 2e-5*magnitude+1e-10 {
					t.Fatalf("input=%d extreme=%v row=%d: pair %g scalar %g", input, extreme, row, got, want)
				}
			}
		}
	}
}

func TestQuantRowsPairSpecial(t *testing.T) {
	data := make([]byte, 2*q80BlockBytes)
	x := make([]float32, 32)
	for i := range x {
		x[i] = 1
	}
	for row := range 2 {
		for i := 2; i < q80BlockBytes; i++ {
			data[row*q80BlockBytes+i] = 1
		}
	}
	for scale := range 1 << 16 {
		binary.LittleEndian.PutUint16(data, uint16(scale))
		binary.LittleEndian.PutUint16(data[q80BlockBytes:], uint16(scale)^0x8000)
		a, b := quantDotQ80Pair(data, x)
		for row, got := range []float32{a, b} {
			want := quantDotScalar(data[row*q80BlockBytes:(row+1)*q80BlockBytes], x, false)
			if got != want && !(math.IsNaN(float64(got)) && math.IsNaN(float64(want))) {
				t.Fatalf("scale=%04x row=%d: pair %g scalar %g", scale, row, got, want)
			}
		}
	}
	for _, value := range []float32{float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1)), math.Float32frombits(0x80000000)} {
		data, x := quantRowsData(96, 2, false)
		x[7] = value
		a, b := quantDotQ80Pair(data, x)
		c, d := quantQ80PairPortable(data, x)
		for row, got := range []float32{a, b} {
			want := []float32{c, d}[row]
			if got != want && !(math.IsNaN(float64(got)) && math.IsNaN(float64(want))) {
				t.Fatalf("activation=%g row=%d: pair %g baseline %g", value, row, got, want)
			}
		}
	}
}

func TestQuantRowsMatrix(t *testing.T) {
	for _, typ := range []uint32{2, 8} {
		for _, input := range []int{32, 96, 512, 4096} {
			t.Run(fmt.Sprintf("type%d/%d", typ, input), func(t *testing.T) {
				const output = 19
				reader, tensor, _ := quantOptimizationFixture(t, typ, input, output)
				data, x := quantRowsData(input, output, typ == 2)
				quantOptimizationReplaceData(t, tensor, data)
				rowBytes := len(data) / output
				want := make([]float32, output)
				for row := range want {
					encoded := data[row*rowBytes : (row+1)*rowBytes]
					if typ == 2 {
						want[row] = dotQ40(encoded, x)
					} else {
						want[row] = dotQ80(encoded, x)
					}
				}
				for _, workers := range []int{1, 4, 32} {
					for _, rows := range []int{1, 2, 3, 4, 7, output} {
						opts := Q40Options{Workers: workers, WindowBytes: uint64(rows*rowBytes + 1)}
						y := make([]float32, output)
						var err error
						if typ == 2 {
							err = MulQ40Into(context.Background(), reader, tensor, x, y, opts)
						} else {
							err = MulQ80Into(context.Background(), reader, tensor, x, y, opts)
						}
						if err != nil {
							t.Fatal(err)
						}
						for row := range y {
							if y[row] != want[row] {
								t.Fatalf("workers=%d rows=%d row=%d: got %g want %g", workers, rows, row, y[row], want[row])
							}
						}
						if typ == 8 {
							best := 0
							for row := range want {
								if want[row] > want[best] {
									best = row
								}
							}
							token, logit, err := MulQ80Argmax(context.Background(), reader, tensor, x, opts)
							if err != nil || token != best || logit != want[best] {
								t.Fatalf("argmax got %d/%g/%v want %d/%g", token, logit, err, best, want[best])
							}
						}
					}
				}
			})
		}
	}
}

func TestQuantRowsArgmaxSpecial(t *testing.T) {
	for _, scales := range [][]uint16{
		{0x7e00, 0xbc00, 0x3c00, 0x3c00, 0x7e00},
		{0x7e00, 0x7e00, 0x7e00, 0x7e00, 0x7e00},
		{0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00},
		{0x7e00, 0xbc00, 0x7c00, 0x7c00, 0x7e00},
	} {
		reader, tensor, data := quantOptimizationFixture(t, 8, 32, len(scales))
		x := make([]float32, 32)
		for i := range x {
			x[i] = 1
		}
		for row, scale := range scales {
			encoded := data[row*34 : (row+1)*34]
			binary.LittleEndian.PutUint16(encoded, scale)
			for i := 2; i < len(encoded); i++ {
				encoded[i] = 1
			}
		}
		quantOptimizationReplaceData(t, tensor, data)
		best, bestLogit := 0, float32(math.Inf(-1))
		for row := range scales {
			value := dotQ80(data[row*34:(row+1)*34], x)
			if value > bestLogit {
				best, bestLogit = row, value
			}
		}
		for _, workers := range []int{1, 4, 32} {
			for _, rows := range []int{1, 2, 3, 5} {
				token, logit, err := MulQ80Argmax(context.Background(), reader, tensor, x,
					Q40Options{Workers: workers, WindowBytes: uint64(rows * 34)})
				if err != nil || token != best || logit != bestLogit {
					t.Fatalf("scales=%x workers=%d rows=%d: got %d/%g/%v want %d/%g", scales, workers, rows, token, logit, err, best, bestLogit)
				}
			}
		}
	}
}

func TestQuantRowsIntoCancellationJoins(t *testing.T) {
	previous := runtime.GOMAXPROCS(4)
	defer runtime.GOMAXPROCS(previous)
	reader, tensor, _ := quantOptimizationFixture(t, 8, 32, 17)
	base, cancel := context.WithCancel(context.Background())
	defer cancel()
	ctx := &quantBlockingContext{Context: base, entered: make(chan struct{}, 64), release: make(chan struct{})}
	x, y := make([]float32, 32), make([]float32, 17)
	opts := Q40Options{Workers: 4, WindowBytes: 17 * q80BlockBytes}
	result := make(chan error, 1)
	go func() { result <- MulQ80Into(ctx, reader, tensor, x, y, opts) }()
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
	if err := MulQ80Into(context.Background(), reader, tensor, x, y, opts); err != nil {
		t.Fatalf("reader reuse after cancellation: %v", err)
	}
}
