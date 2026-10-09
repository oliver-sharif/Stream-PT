//go:build goexperiment.simd

package forward

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"testing"
	"time"

	"Stream-PT/ggufindex"
	ggufmmap "Stream-PT/ggufmap"
)

func workerPoolGoroutine() string {
	var stack [128]byte
	n := runtime.Stack(stack[:], false)
	return string(bytes.Fields(stack[:n])[1])
}

func TestWorkerPoolReuseAndLimits(t *testing.T) {
	previous := runtime.GOMAXPROCS(2)
	defer runtime.GOMAXPROCS(previous)
	for _, count := range []int{1, 2, 4} {
		ids := make([]string, count)
		values := make([]int, 7)
		caller := workerPoolGoroutine()
		operation := newQuantWorkers(count, func(worker int, job quantRowJob) {
			id := workerPoolGoroutine()
			if ids[worker] != "" && ids[worker] != id {
				t.Errorf("worker %d changed goroutine: %s -> %s", worker, ids[worker], id)
			}
			ids[worker] = id
			if count == 1 && id != caller {
				t.Error("single worker must execute on the caller")
			}
			if count > 1 && id == caller {
				t.Error("parallel worker executed on the caller")
			}
			if job.firstRow != 13 {
				t.Errorf("first row = %d, want 13", job.firstRow)
			}
			for row := job.begin; row < job.end; row++ {
				values[row]++
			}
		})
		want := count
		if count == 1 {
			want = 0
		}
		if len(operation.jobs) != want {
			t.Fatalf("count %d: got %d goroutines, want %d", count, len(operation.jobs), want)
		}
		for iteration := 0; iteration < 20; iteration++ {
			operation.run(nil, 13, 7)
			operation.run(nil, 13, 0)
			operation.run(nil, 13, 1)
		}
		operation.close()
		for row, value := range values {
			want := 20
			if row == 0 {
				want = 40
			}
			if value != want {
				t.Fatalf("row %d visited %d times, want %d", row, value, want)
			}
		}
		for _, jobs := range operation.jobs {
			if _, open := <-jobs; open {
				t.Fatal("worker channel still open after close")
			}
		}
		operation.joined.Wait()
	}
	// Public kernels normalize counts before creating per-multiplication workers.
	for _, mode := range []string{"q40", "q40batch", "argmax"} {
		for _, count := range []int{-2, 0, 1, 100} {
			for _, rows := range []int{1, 4} {
				typ, blockBytes, batch := uint32(2), 18, 1
				if mode == "argmax" {
					typ, blockBytes = 8, 34
				}
				if mode == "q40batch" {
					batch = 4
				}
				reader, tensor, _ := quantOptimizationFixture(t, typ, 32, 7)
				ctx := &quantRecordingContext{Context: context.Background(), ids: make(map[string]bool)}
				if err := quantOptimizationRun(ctx, reader, tensor, make([]float32, batch*32), make([]float32, batch*7), mode,
					Q40Options{Workers: count, WindowBytes: uint64(rows * blockBytes)}); err != nil {
					t.Fatal(err)
				}
				want := 1
				workers := min(max(count, 1), 2, rows)
				if workers > 1 {
					want += workers
				}
				if len(ctx.ids) != want {
					t.Fatalf("%s/count%d/rows%d: used %d goroutines, want %d", mode, count, rows, len(ctx.ids), want)
				}
			}
		}
	}
}

func workerPoolReader(t *testing.T, data []byte, kind uint32, input, output int) (*ggufmmap.Reader, ggufindex.Tensor) {
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
	return reader, ggufindex.Tensor{Name: "weights", Type: kind,
		Shape: []uint64{uint64(input), uint64(output)},
		Range: ggufindex.Range{File: path, End: uint64(len(data))}}
}

func workerPoolEqual(t *testing.T, want, got []float32) {
	t.Helper()
	for i := range want {
		if math.Float32bits(want[i]) != math.Float32bits(got[i]) {
			t.Fatalf("element %d: got %g, want %g (not bit-identical)", i, got[i], want[i])
		}
	}
}

func TestWorkerPoolQuantOutputs(t *testing.T) {
	previous := runtime.GOMAXPROCS(4)
	defer runtime.GOMAXPROCS(previous)
	const input, output = 64, 11
	for _, kind := range []uint32{2, 8} {
		blockBytes := q40BlockBytes
		if kind == 8 {
			blockBytes = q80BlockBytes
		}
		rowBytes := input / 32 * blockBytes
		data := make([]byte, output*rowBytes)
		for block := 0; block < len(data)/blockBytes; block++ {
			encoded := data[block*blockBytes : (block+1)*blockBytes]
			binary.LittleEndian.PutUint16(encoded, 0x3800+uint16(block%4)*0x400)
			for i := 2; i < len(encoded); i++ {
				encoded[i] = byte(block*17 + i*13)
			}
		}
		reader, tensor := workerPoolReader(t, data, kind, input, output)
		for _, batch := range []int{1, 3, 17} {
			if kind == 8 && batch != 1 {
				continue
			}
			x := make([]float32, input*batch)
			for i := range x {
				x[i] = float32(i%19-9) / 8
			}
			want := make([]float32, output*batch)
			if kind == 8 {
				if err := q80KernelRowsForTest(reader, tensor, x, want); err != nil {
					t.Fatal(err)
				}
			} else if err := MulQ40BatchInto(context.Background(), reader, tensor, x, want, batch,
				Q40Options{Workers: 1, WindowBytes: uint64(rowBytes * 4)}); err != nil {
				t.Fatal(err)
			}
			for _, workers := range []int{1, 2, 4, 100} {
				options := Q40Options{Workers: workers, WindowBytes: uint64(rowBytes * 4)}
				for repeat := 0; repeat < 3; repeat++ {
					if kind == 8 {
						token, logit, err := MulQ80Argmax(context.Background(), reader, tensor, x, options)
						best := 0
						for i := range want {
							if want[i] > want[best] {
								best = i
							}
						}
						if err != nil || token != best || math.Float32bits(logit) != math.Float32bits(want[best]) {
							t.Fatalf("argmax: got (%d, %g, %v), want (%d, %g)", token, logit, err, best, want[best])
						}
					} else {
						got := make([]float32, len(want))
						if err := MulQ40BatchInto(context.Background(), reader, tensor, x, got, batch, options); err != nil {
							t.Fatal(err)
						}
						workerPoolEqual(t, want, got)
					}
				}
			}
		}
	}
}

func TestWorkerPoolMXFP4Outputs(t *testing.T) {
	previous := runtime.GOMAXPROCS(4)
	defer runtime.GOMAXPROCS(previous)
	const input, output = 64, 11
	data := make([]byte, output*input/32*mxfp4BlockBytes)
	for block := 0; block < len(data)/mxfp4BlockBytes; block++ {
		encoded := data[block*mxfp4BlockBytes : (block+1)*mxfp4BlockBytes]
		encoded[0] = byte(125 + block%5)
		for i := 1; i < len(encoded); i++ {
			encoded[i] = byte(block*17 + i*13)
		}
	}
	for _, batch := range []int{1, 2, 5, 17} {
		x := make([]float32, batch*input)
		bias := make([]float32, output)
		xOffsets, yOffsets := make([]int, batch), make([]int, batch)
		for i := range x {
			x[i] = float32(i%19-9) / 8
		}
		for i := range bias {
			bias[i] = float32(i) / 4
		}
		for i := range xOffsets {
			xOffsets[i], yOffsets[i] = (batch-1-i)*input, i*(output+3)
		}
		want := make([]float32, batch*(output+3))
		if err := mulMoEBatchExpert(context.Background(), data, input, output, x, xOffsets, bias, want, yOffsets, Q40Options{Workers: 1}); err != nil {
			t.Fatal(err)
		}
		for _, workers := range []int{2, 4, 100} {
			for repeat := 0; repeat < 3; repeat++ {
				got := make([]float32, len(want))
				if err := mulMoEBatchExpert(context.Background(), data, input, output, x, xOffsets, bias, got, yOffsets, Q40Options{Workers: workers}); err != nil {
					t.Fatal(err)
				}
				workerPoolEqual(t, want, got)
			}
		}
	}
}

func TestWorkerPoolCancellationMmapAndClose(t *testing.T) {
	previous := runtime.GOMAXPROCS(2)
	defer runtime.GOMAXPROCS(previous)
	reader, tensor := workerPoolReader(t, []byte{7, 9}, 0, 1, 2)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	started := make(chan struct{}, 2)
	release := make(chan struct{})
	defer func() {
		select {
		case <-release:
		default:
			close(release)
		}
	}()
	completed := make(chan error, 1)
	closed := make(chan struct{})
	values := make([]byte, 2)
	operation := newQuantWorkers(2, func(worker int, job quantRowJob) {
		started <- struct{}{}
		<-release
		// Deliberately read the live mapping even after cancellation.
		values[worker] = job.data[job.begin]
	})
	go func() {
		err := reader.WithTensorChunks(tensor, 2, func(_ uint64, data []byte) error {
			operation.run(data, 0, len(data))
			return ctx.Err()
		})
		operation.close()
		close(closed)
		completed <- err
	}()
	for i := 0; i < 2; i++ {
		select {
		case <-started:
		case <-time.After(5 * time.Second):
			t.Fatal("workers did not start")
		}
	}
	cancel()
	select {
	case err := <-completed:
		t.Fatalf("mapping returned before worker barrier: %v", err)
	case <-closed:
		t.Fatal("workers closed while mapping was leased")
	case <-time.After(20 * time.Millisecond):
	}
	close(release)
	select {
	case err := <-completed:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("got %v, want cancellation", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("canceled multiplication did not join")
	}
	select {
	case <-closed:
	default:
		t.Fatal("close did not join workers")
	}
	if !bytes.Equal(values, []byte{7, 9}) {
		t.Fatalf("mapping was not valid until workers joined: %v", values)
	}
	for _, jobs := range operation.jobs {
		if _, open := <-jobs; open {
			t.Fatal("worker channel still open after cancellation")
		}
	}
}

func TestWorkerPoolConcurrentLeases(t *testing.T) {
	var joined sync.WaitGroup
	for caller := 0; caller < 8; caller++ {
		joined.Add(1)
		go func() {
			defer joined.Done()
			values := make([]int, 13)
			operation := newQuantWorkers(4, func(_ int, job quantRowJob) {
				for row := job.begin; row < job.end; row++ {
					values[row]++
				}
			})
			for repeat := 0; repeat < 10; repeat++ {
				operation.run(nil, 0, len(values))
			}
			operation.close()
			for _, value := range values {
				if value != 10 {
					t.Errorf("concurrent operation lost rows: %v", values)
					break
				}
			}
		}()
	}
	joined.Wait()
}

func BenchmarkWorkerPoolExpert(b *testing.B) {
	previous := runtime.GOMAXPROCS(4)
	defer runtime.GOMAXPROCS(previous)
	const input, output = 2880, 2880
	data := make([]byte, input/32*mxfp4BlockBytes*output)
	for block := 0; block < len(data)/mxfp4BlockBytes; block++ {
		data[block*mxfp4BlockBytes] = 125
		for i := 1; i < mxfp4BlockBytes; i++ {
			data[block*mxfp4BlockBytes+i] = byte(block + i)
		}
	}
	for _, batch := range []int{1, 2, 4, 16} {
		for _, workers := range []int{1, 4} {
			b.Run(fmt.Sprintf("batch=%d/workers=%d", batch, workers), func(b *testing.B) {
				x, y, bias := make([]float32, input*batch), make([]float32, output*batch), make([]float32, output)
				in, out := make([]int, batch), make([]int, batch)
				for i := range x {
					x[i] = float32(i%17-8) / 8
				}
				for i := range in {
					in[i], out[i] = i*input, i*output
				}
				opts := Q40Options{Workers: workers}
				b.ResetTimer()
				for b.Loop() {
					if err := mulMoEBatchExpert(context.Background(), data, input, output, x, in, bias, y, out, opts); err != nil {
						b.Fatal(err)
					}
				}
			})
		}
	}
}
