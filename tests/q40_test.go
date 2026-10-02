//go:build goexperiment.simd

package tests

import (
	"context"
	"encoding/binary"
	"math"
	"testing"

	. "Stream-PT/forward"
)

func TestDotQ40(t *testing.T) {
	// One block with scale 1.0: first weight 7 (nibble 15),
	// seventeenth weight -8 (nibble 0).
	block := make([]byte, q40BlockBytes)
	binary.LittleEndian.PutUint16(block[:2], 0x3c00)
	block[2] = 0x0f

	x := make([]float32, q40Elements)
	x[0] = 2
	x[16] = 3

	got := multiplyQ40Row(t, block, x)
	want := float32(7*2 - 8*3)
	if got != want {
		t.Fatalf("Q4_0 dot product = %v, want %v", got, want)
	}
}

func TestDotQ40TwoBlocks(t *testing.T) {
	row := make([]byte, 2*q40BlockBytes)

	// Both blocks have scale 1 and only nibbles 8 (zero weights).
	for block := 0; block < 2; block++ {
		begin := block * q40BlockBytes
		binary.LittleEndian.PutUint16(row[begin:begin+2], 0x3c00)
		for i := begin + 2; i < begin+q40BlockBytes; i++ {
			row[i] = 0x88
		}
	}

	// Set the first weight in the second block to +1.
	row[q40BlockBytes+2] = 0x89

	x := make([]float32, 2*q40Elements)
	x[q40Elements] = 5

	if got := multiplyQ40Row(t, row, x); got != 5 {
		t.Fatalf("Q4_0 dot product = %v, want 5", got)
	}
}

func TestFloat16(t *testing.T) {
	tests := []struct {
		bits uint16
		want float32
	}{
		{0x0000, 0},
		{0x3c00, 1},
		{0xbc00, -1},
		{0x0001, 1.0 / 16777216},
		{0x7c00, float32(math.Inf(1))},
	}

	for _, tt := range tests {
		reader, weight := tensorFixture(t, 1, []uint64{1}, encodeUint16([]uint16{tt.bits}))
		y := make([]float32, 1)
		if err := RMSNormInto(context.Background(), reader, weight, []float32{1}, y, 0); err != nil {
			t.Fatal(err)
		}
		got := y[0]
		if got != tt.want {
			t.Errorf("F16 decoding(%#04x) = %v, want %v",
				tt.bits, got, tt.want)
		}
	}
}
