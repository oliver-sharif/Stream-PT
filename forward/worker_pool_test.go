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
	for _, count := range []int{-2, 0, 1, 100} {
		pool := newQuantWorkerPool(count)
		want := 0
		if count > 1 {
			want = 2
		}
		if len(pool.jobs) != want {
			t.Fatalf("count %d: got %d goroutines, want %d", count, len(pool.jobs), want)
		}
		ids := make([]string, max(want, 1))
		caller := workerPoolGoroutine()
		for iteration := 0; iteration < 20; iteration++ {
			operation := newQuantWorkers(count, func(worker int, job quantRowJob) {
				id := workerPoolGoroutine()
				if ids[worker] != "" && ids[worker] != id {
					t.Errorf("worker %d changed goroutine: %s -> %s", worker, ids[worker], id)
				}
				ids[worker] = id
				if want == 0 && id != caller {
					t.Error("single worker must execute on the caller")
				}
			}, pool)
			operation.run(nil, 0, 7)
			operation.run(nil, 0, 0)
			operation.close()
			operation.close()
		}
		pool.Close()
		pool.Close()
		for _, jobs := range pool.jobs {
			if _, open := <-jobs; open {
				t.Fatal("worker channel still open after Close")
			}
		}
		pool.joined.Wait()
		operation := newQuantWorkers(count, func(_ int, job quantRowJob) {
			if workerPoolGoroutine() != caller {
				t.Error("closed pool must fall back to synchronous execution")
			}
		}, pool)
		operation.run(nil, 0, 3)
		operation.close()
	}
	var pool *quantWorkerPool
	pool.Close()
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
	pool := newQuantWorkerPool(2)
	defer pool.Close()
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
			mul := func(y []float32, options Q40Options) error {
				if kind == 8 {
					return MulQ80Into(context.Background(), reader, tensor, x, y, options)
				}
				return MulQ40BatchInto(context.Background(), reader, tensor, x, y, batch, options)
			}
			if err := mul(want, Q40Options{Workers: 1, WindowBytes: uint64(rowBytes * 4)}); err != nil {
				t.Fatal(err)
			}
			for _, shared := range []*quantWorkerPool{nil, pool} {
				options := Q40Options{Workers: 4, WindowBytes: uint64(rowBytes * 4), workerPool: shared}
				for repeat := 0; repeat < 3; repeat++ {
					got := make([]float32, len(want))
					if err := mul(got, options); err != nil {
						t.Fatal(err)
					}
					workerPoolEqual(t, want, got)
					if kind == 8 {
						token, logit, err := MulQ80Argmax(context.Background(), reader, tensor, x, options)
						best := 0
						for i := range want {
							if want[i] > want[best] {
								best = i
							}
						}
						if err != nil || token != best || logit != want[best] {
							t.Fatalf("argmax: got (%d, %g, %v), want (%d, %g)", token, logit, err, best, want[best])
						}
					}
				}
			}
		}
	}
}

func TestWorkerPoolMXFP4Outputs(t *testing.T) {
	previous := runtime.GOMAXPROCS(4)
	defer runtime.GOMAXPROCS(previous)
	pool := newQuantWorkerPool(2)
	defer pool.Close()
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
		for _, shared := range []*quantWorkerPool{nil, pool} {
			for repeat := 0; repeat < 3; repeat++ {
				got := make([]float32, len(want))
				if err := mulMoEBatchExpert(context.Background(), data, input, output, x, xOffsets, bias, got, yOffsets, Q40Options{Workers: 4, workerPool: shared}); err != nil {
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
	pool := newQuantWorkerPool(2)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	started := make(chan struct{}, 2)
	release := make(chan struct{})
	completed := make(chan error, 1)
	closed := make(chan struct{})
	values := make([]byte, 2)
	go func() {
		operation := newQuantWorkers(2, func(worker int, job quantRowJob) {
			started <- struct{}{}
			<-release
			// Deliberately read the live mapping even after cancellation.
			values[worker] = job.data[job.begin]
		}, pool)
		err := reader.WithTensorChunks(tensor, 2, func(_ uint64, data []byte) error {
			operation.run(data, 0, len(data))
			return ctx.Err()
		})
		operation.close()
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
	go func() {
		pool.Close()
		close(closed)
	}()
	select {
	case err := <-completed:
		t.Fatalf("mapping returned before worker barrier: %v", err)
	case <-closed:
		t.Fatal("Close returned while mapping was leased")
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
	case <-time.After(5 * time.Second):
		t.Fatal("Close did not join workers")
	}
	if !bytes.Equal(values, []byte{7, 9}) {
		t.Fatalf("mapping was not valid until workers joined: %v", values)
	}
	pool.Close()
}

func TestWorkerPoolConcurrentLeases(t *testing.T) {
	pool := newQuantWorkerPool(4)
	defer pool.Close()
	var joined sync.WaitGroup
	for caller := 0; caller < 8; caller++ {
		joined.Add(1)
		go func() {
			defer joined.Done()
			values := make([]int, 13)
			for repeat := 0; repeat < 10; repeat++ {
				operation := newQuantWorkers(4, func(_ int, job quantRowJob) {
					for row := job.begin; row < job.end; row++ {
						values[row]++
					}
				}, pool)
				operation.run(nil, 0, len(values))
				operation.close()
			}
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
		for _, shared := range []bool{false, true} {
			b.Run(fmt.Sprintf("batch=%d/pool=%t", batch, shared), func(b *testing.B) {
				pool := newQuantWorkerPool(4)
				defer pool.Close()
				x, y, bias := make([]float32, input*batch), make([]float32, output*batch), make([]float32, output)
				in, out := make([]int, batch), make([]int, batch)
				for i := range x {
					x[i] = float32(i%17-8) / 8
				}
				for i := range in {
					in[i], out[i] = i*input, i*output
				}
				opts := Q40Options{Workers: 4}
				if shared {
					opts.workerPool = pool
				}
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
