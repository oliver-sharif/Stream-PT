//go:build goexperiment.simd

package tests

import (
	"context"
	"math"
	"os"
	"path/filepath"
	"testing"

	. "Stream-PT/forward"
	"Stream-PT/ggufindex"
	ggufmmap "Stream-PT/ggufmap"
)

func TestRMSNormInto(t *testing.T) {
	for _, typ := range []uint32{0, 1, 28} {
		data := encodeFloat32([]float32{1, 2, 0.5, -1, 1})
		if typ == 1 {
			data = encodeUint16([]uint16{0x3c00, 0x4000, 0x3800, 0xbc00, 0x3c00})
		} else if typ == 28 {
			data = encodeUint16([]uint16{0x3f80, 0x4000, 0x3f00, 0xbf80, 0x3f80})
		}
		path := filepath.Join(t.TempDir(), "weights.bin")
		if err := os.WriteFile(path, append([]byte{0}, data...), 0600); err != nil {
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
		weight := ggufindex.Tensor{Type: typ, Shape: []uint64{5},
			Range: ggufindex.Range{File: path, Start: 1, End: uint64(1 + len(data))}}
		x := []float32{2, -4, 1, 3, -2}
		want := referenceRMSNorm(x, []float32{1, 2, 0.5, -1, 1}, 1e-5)
		y := make([]float32, len(x))
		if err := RMSNormInto(context.Background(), reader, weight, x, y, 1e-5); err != nil {
			t.Fatal(err)
		}
		compareVectors(t, y, want)
		if err := RMSNormInto(context.Background(), reader, weight, x, x, 1e-5); err != nil {
			t.Fatal(err)
		}
		compareVectors(t, x, want)
		for _, epsilon := range []float32{-1, float32(math.NaN()), float32(math.Inf(1))} {
			if err := RMSNormInto(context.Background(), reader, weight, x, y, epsilon); err == nil {
				t.Fatalf("expected error for epsilon %v", epsilon)
			}
		}
		shared := make([]float32, 6)
		if err := RMSNormInto(context.Background(), reader, weight, shared[:5], shared[1:], 1e-5); err == nil {
			t.Fatal("expected partial-overlap error")
		}
		if err := RMSNormInto(context.Background(), reader, weight, x, y[:4], 1e-5); err == nil {
			t.Fatal("expected output-length error")
		}
	}
}

func TestRMSNormIntoWindowBoundary(t *testing.T) {
	// Cross the default weight window and exercise a partial final SIMD vector.
	count := DefaultQ40WindowBytes/4 + 3
	weights, x, y := make([]float32, count), make([]float32, count), make([]float32, count)
	for i := range x {
		weights[i] = float32(i%5+1) / 2
		x[i] = float32(i%3 - 1)
	}
	path := filepath.Join(t.TempDir(), "weights.bin")
	if err := os.WriteFile(path, encodeFloat32(weights), 0600); err != nil {
		t.Fatal(err)
	}
	reader, err := ggufmmap.Open(&ggufindex.Model{Paths: []string{path}})
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := reader.Close(); err != nil {
			t.Error(err)
		}
	}()
	weight := ggufindex.Tensor{Type: 0, Shape: []uint64{uint64(count)},
		Range: ggufindex.Range{File: path, End: uint64(count * 4)}}
	if err := RMSNormInto(context.Background(), reader, weight, x, y, 1e-5); err != nil {
		t.Fatal(err)
	}
	compareVectors(t, y, referenceRMSNorm(x, weights, 1e-5))
}
