//go:build goexperiment.simd

package forward

import (
	"context"
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"

	"Stream-PT/ggufindex"
	ggufmmap "Stream-PT/ggufmap"
)

func TestMulQ40RealGGUFRow(t *testing.T) {
	paths := []string{
		filepath.Join("..", "model", "gpt-oss-120b-Q4_0-00001-of-00002.gguf"),
		filepath.Join("..", "model", "gpt-oss-120b-Q4_0-00002-of-00002.gguf"),
	}
	for _, path := range paths {
		if _, err := os.Stat(path); err != nil {
			t.Skipf("Model shard unavailable: %s: %v", path, err)
		}
	}

	model, err := ggufindex.Open(paths)
	if err != nil {
		t.Fatalf("Open GGUF index: %v", err)
	}

	var chosen ggufindex.Tensor
	found := false

	for _, layer := range model.Layers {
		for _, tensor := range layer.Tensors {
			if tensor.Type != 2 || len(tensor.Shape) != 2 {
				continue
			}
			input := tensor.Shape[0]
			if input == 0 || input%q40Elements != 0 || input > 16384 {
				continue
			}
			chosen = tensor
			found = true
			break
		}
		if found {
			break
		}
	}
	if !found {
		t.Skip("No suitable two-dimensional Q4_0 layer matrix found")
	}

	reader, err := ggufmmap.Open(model)
	if err != nil {
		t.Fatalf("Open GGUF reader: %v", err)
	}
	defer func() {
		if err := reader.Close(); err != nil {
			t.Errorf("Close GGUF reader: %v", err)
		}
	}()

	input := int(chosen.Shape[0])
	rowBytes := uint64(input/q40Elements) * q40BlockBytes

	// A valid tensor subrange containing only row 0.
	firstRow := chosen
	firstRow.Shape = []uint64{uint64(input), 1}
	firstRow.Range.End = firstRow.Range.Start + rowBytes

	x := make([]float32, input)
	for i := range x {
		// Deterministic input without a random source.
		x[i] = float32((i%17)-8) / 16
	}

	got, err := mulQ40ForTest(context.Background(), reader, firstRow, x, 1)
	if err != nil {
		t.Fatalf("SIMD multiplication of %q: %v", chosen.Name, err)
	}

	var want float32
	err = reader.WithTensor(firstRow, func(data []byte) error {
		want = referenceQ40Dot(data, x)
		return nil
	})
	if err != nil {
		t.Fatalf("Read reference data for %q: %v", chosen.Name, err)
	}

	if len(got) != 1 {
		t.Fatalf("Expected one output, got %d", len(got))
	}

	// SIMD FMA and a different summation order may introduce small rounding
	// differences compared with scalar computation.
	tolerance := float32(0.001) * (1 + float32(math.Abs(float64(want))))
	if math.IsNaN(float64(got[0])) ||
		float32(math.Abs(float64(got[0]-want))) > tolerance {
		t.Fatalf("%q: SIMD=%g, scalar=%g, tolerance=%g",
			chosen.Name, got[0], want, tolerance)
	}

	t.Logf("Real GGUF tensor %q: input=%d, SIMD=%g, scalar=%g",
		chosen.Name, input, got[0], want)
}

func referenceQ40Dot(row []byte, x []float32) float32 {
	var sum float32

	for block := 0; block < len(row)/q40BlockBytes; block++ {
		b := row[block*q40BlockBytes : (block+1)*q40BlockBytes]
		scale := referenceFloat16(binary.LittleEndian.Uint16(b[:2]))

		for i := range 16 {
			packed := b[2+i]
			low := int(packed&0x0f) - 8
			high := int(packed>>4) - 8

			sum += scale * float32(low) * x[block*32+i]
			sum += scale * float32(high) * x[block*32+i+16]
		}
	}
	return sum
}
