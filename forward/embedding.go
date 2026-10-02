//go:build goexperiment.simd

package forward

import (
	"context"
	"encoding/binary"
	"fmt"

	"Stream-PT/ggufindex"
	ggufmmap "Stream-PT/ggufmap"
)

// ReadEmbedding extracts the embedding vector for tokenID from token_embd.weight (Q4_0)
// directly into out without mapping or copying the rest of the embedding matrix.
func ReadEmbedding(
	ctx context.Context,
	reader *ggufmmap.Reader,
	tensor ggufindex.Tensor,
	tokenID int,
	out []float32,
) error {
	if ctx == nil {
		return fmt.Errorf("nil context")
	}
	if reader == nil {
		return fmt.Errorf("nil reader")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if len(tensor.Shape) != 2 {
		return fmt.Errorf("%q: expected 2D embedding tensor", tensor.Name)
	}
	dim, vocab := int(tensor.Shape[0]), int(tensor.Shape[1])
	if tokenID < 0 || tokenID >= vocab {
		return fmt.Errorf("token %d out of vocab range [0, %d)", tokenID, vocab)
	}
	if len(out) != dim {
		return fmt.Errorf("output buffer length %d != embedding dim %d", len(out), dim)
	}

	if tensor.Type != 2 {
		return fmt.Errorf("%q: unsupported embedding tensor type %d (expected Q4_0)", tensor.Name, tensor.Type)
	}

	rowBlocks := dim / q40Elements
	rowBytes := uint64(rowBlocks * q40BlockBytes)
	offset := uint64(tokenID) * rowBytes

	span := ggufindex.Range{
		File:  tensor.Range.File,
		Start: tensor.Range.Start + offset,
		End:   tensor.Range.Start + offset + rowBytes,
	}

	rowTensor := ggufindex.Tensor{
		Name:  fmt.Sprintf("%s[%d]", tensor.Name, tokenID),
		Type:  tensor.Type,
		Shape: []uint64{uint64(dim), 1},
		Range: span,
	}

	return reader.WithTensor(rowTensor, func(data []byte) error {
		if len(data) < int(rowBytes) {
			return fmt.Errorf("unexpected short read: %d < %d", len(data), rowBytes)
		}
		for block := 0; block < rowBlocks; block++ {
			b := data[block*q40BlockBytes : (block+1)*q40BlockBytes]
			scale := float16(binary.LittleEndian.Uint16(b[:2]))
			for i := 0; i < 16; i++ {
				packed := b[2+i]
				low := int(packed&0x0f) - 8
				high := int(packed>>4) - 8
				out[block*32+i] = scale * float32(low)
				out[block*32+i+16] = scale * float32(high)
			}
		}
		return nil
	})
}
