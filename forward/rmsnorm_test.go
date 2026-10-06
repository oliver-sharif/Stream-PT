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

func TestRMSNormFormats(t *testing.T) {
	tests := []struct {
		name string
		typ  uint32
		data []byte
	}{
		{
			name: "F32",
			typ:  0,
			data: encodeFloat32([]float32{1, 2, 0.5, -1, 1}),
		},
		{
			name: "F16",
			typ:  1,
			data: encodeUint16([]uint16{
				0x3c00, //  1
				0x4000, //  2
				0x3800, //  0.5
				0xbc00, // -1
				0x3c00, //  1
			}),
		},
		{
			name: "BF16",
			typ:  28,
			data: encodeUint16([]uint16{
				0x3f80, //  1
				0x4000, //  2
				0x3f00, //  0.5
				0xbf80, // -1
				0x3f80, //  1
			}),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "weights.bin")
			if err := os.WriteFile(path, tt.data, 0600); err != nil {
				t.Fatal(err)
			}

			tensor := ggufindex.Tensor{
				Name:  "norm.weight",
				Type:  tt.typ,
				Shape: []uint64{5},
				Range: ggufindex.Range{
					File:  path,
					Start: 0,
					End:   uint64(len(tt.data)),
				},
			}
			reader, err := ggufmmap.Open(&ggufindex.Model{
				Paths: []string{path},
			})
			if err != nil {
				t.Fatal(err)
			}
			defer func() {
				if err := reader.Close(); err != nil {
					t.Error(err)
				}
			}()

			x := []float32{2, -4, 1, 3, -2}
			original := append([]float32(nil), x...)
			epsilon := float32(1e-5)

			got, err := rmsNormForTest(
				context.Background(), reader, tensor, x, epsilon,
			)
			if err != nil {
				t.Fatal(err)
			}
			if len(got) != len(x) {
				t.Fatalf("%d outputs, want %d", len(got), len(x))
			}

			weights := []float32{1, 2, 0.5, -1, 1}
			want := referenceRMSNorm(x, weights, epsilon)
			compareVectors(t, got, want)

			for i := range x {
				if x[i] != original[i] {
					t.Fatalf("Input x[%d] was modified", i)
				}
			}
		})
	}
}

func TestRMSNormRealGGUF(t *testing.T) {
	paths := []string{
		filepath.Join("..", "model",
			"gpt-oss-120b-Q4_0-00001-of-00002.gguf"),
		filepath.Join("..", "model",
			"gpt-oss-120b-Q4_0-00002-of-00002.gguf"),
	}
	for _, path := range paths {
		if _, err := os.Stat(path); err != nil {
			t.Skipf("Model shard missing: %s: %v", path, err)
		}
	}

	model, err := ggufindex.Open(paths)
	if err != nil {
		t.Fatal(err)
	}

	weight, ok := model.TensorByName("blk.0.attn_norm.weight")
	if !ok {
		t.Fatal("blk.0.attn_norm.weight is missing from GGUF")
	}
	if len(weight.Shape) != 1 {
		t.Fatalf("Unexpected normalization shape: %v", weight.Shape)
	}

	reader, err := ggufmmap.Open(model)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := reader.Close(); err != nil {
			t.Error(err)
		}
	}()

	x := make([]float32, int(weight.Shape[0]))
	for i := range x {
		x[i] = float32((i%23)-11) / 8
	}

	// This is a fixed TEST VALUE, not a claim that epsilon was read
	// from the GGUF metadata.
	epsilon := float32(1e-5)
	got, err := rmsNormForTest(
		context.Background(), reader, weight, x, epsilon,
	)
	if err != nil {
		t.Fatal(err)
	}

	weights := make([]float32, len(x))
	err = reader.WithTensor(weight, func(data []byte) error {
		for i := range weights {
			switch weight.Type {
			case 0:
				weights[i] = math.Float32frombits(
					binary.LittleEndian.Uint32(data[i*4:]),
				)
			case 1:
				weights[i] = referenceFloat16(
					binary.LittleEndian.Uint16(data[i*2:]),
				)
			case 28:
				weights[i] = math.Float32frombits(
					uint32(binary.LittleEndian.Uint16(data[i*2:])) << 16,
				)
			default:
				t.Fatalf("Normalization tensor type %d is unsupported", weight.Type)
			}
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}

	want := referenceRMSNorm(x, weights, epsilon)
	compareVectors(t, got, want)
	t.Logf("Real GGUF normalization tensor %q: type=%d, dimension=%d",
		weight.Name, weight.Type, len(x))
}

func referenceRMSNorm(
	x, weights []float32,
	epsilon float32,
) []float32 {
	var sum float64
	for _, value := range x {
		sum += float64(value) * float64(value)
	}

	factor := 1 / math.Sqrt(sum/float64(len(x))+float64(epsilon))
	y := make([]float32, len(x))
	for i := range y {
		y[i] = float32(float64(x[i]) *
			float64(weights[i]) * factor)
	}
	return y
}

func compareVectors(t *testing.T, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("Lengths: got %d, want %d",
			len(got), len(want))
	}
	for i := range got {
		tolerance := 0.0001 * (1 + math.Abs(float64(want[i])))
		if math.IsNaN(float64(got[i])) ||
			math.Abs(float64(got[i]-want[i])) > tolerance {
			t.Fatalf("Element %d: got %g, want %g",
				i, got[i], want[i])
		}
	}
}

func encodeFloat32(values []float32) []byte {
	data := make([]byte, len(values)*4)
	for i, value := range values {
		binary.LittleEndian.PutUint32(
			data[i*4:], math.Float32bits(value),
		)
	}
	return data
}

func encodeUint16(values []uint16) []byte {
	data := make([]byte, len(values)*2)
	for i, value := range values {
		binary.LittleEndian.PutUint16(data[i*2:], value)
	}
	return data
}
