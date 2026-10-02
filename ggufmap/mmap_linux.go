package ggufmmap

import (
	"errors"
	"fmt"
	"os"
	"syscall"

	"Stream-PT/ggufindex"
)

// Reader keeps file descriptors open without loading model weights into the Go heap.
// Files must remain immutable while the Reader is in use: truncating a mapped
// file can cause a fatal fault. Mapped pages still consume physical RAM through
// the OS page cache, even though they are not allocated on the Go heap.
type Reader struct {
	files map[string]*os.File
}

// Open opens the files referenced by an existing GGUF index.
func Open(model *ggufindex.Model) (*Reader, error) {
	if model == nil {
		return nil, fmt.Errorf("nil model")
	}

	r := &Reader{files: make(map[string]*os.File, len(model.Paths))}
	for _, path := range model.Paths {
		f, err := os.Open(path)
		if err != nil {
			r.Close()
			return nil, fmt.Errorf("open %q: %w", path, err)
		}
		r.files[path] = f
	}
	return r, nil
}

// Close closes the files. Do not call it concurrently with WithTensor,
// WithTensorChunks, or WithLayer (including from their callbacks).
func (r *Reader) Close() error {
	var first error
	for path, f := range r.files {
		if err := f.Close(); err != nil && first == nil {
			first = fmt.Errorf("close %q: %w", path, err)
		}
		delete(r.files, path)
	}
	return first
}

// WithTensor provides the tensor's bytes for the duration of fn.
// data is read-only: fn must never modify it or retain it after returning.
func (r *Reader) WithTensor(t ggufindex.Tensor, fn func(data []byte) error) (err error) {
	if fn == nil {
		return fmt.Errorf("nil callback")
	}

	data, unmap, err := r.mapRange(t.Range)
	if err != nil {
		return fmt.Errorf("tensor %q: %w", t.Name, err)
	}
	defer func() { err = errors.Join(err, unmap()) }()

	return fn(data)
}

// WithTensorChunks provides successive read-only, zero-copy chunks of a tensor.
// Each callback receives an offset relative to the tensor and at most chunkBytes
// bytes, with capacity equal to length. It must never modify or retain data.
// The entire range is validated before any callback, and each mapping is released
// before the next chunk. A callback or cleanup error stops iteration.
// The requested mapping length is bounded by chunkBytes plus at most one page
// minus one byte of leading alignment; the OS also rounds the end to a page.
// This bounds the active mapping, not physical RAM: mmap uses the file page cache,
// and readahead or cached pages may outlive a mapping. No global cache is dropped.
// Files must remain immutable, and Close must not run concurrently with this call.
func (r *Reader) WithTensorChunks(t ggufindex.Tensor, chunkBytes uint64, fn func(offset uint64, data []byte) error) error {
	if fn == nil {
		return fmt.Errorf("nil callback")
	}
	if chunkBytes == 0 {
		return fmt.Errorf("zero chunk size")
	}
	f, err := r.validateRange(t.Range)
	if err != nil {
		return fmt.Errorf("tensor %q: %w", t.Name, err)
	}
	for start := t.Range.Start; start < t.Range.End; {
		length := t.Range.End - start
		if length > chunkBytes {
			length = chunkBytes
		}
		span := ggufindex.Range{File: t.Range.File, Start: start, End: start + length}
		data, unmap, err := mapFileRange(f, span, true)
		if err != nil {
			return fmt.Errorf("tensor %q, offset %d: %w", t.Name, start-t.Range.Start, err)
		}
		err = func() (err error) {
			defer func() { err = errors.Join(err, unmap()) }()
			return fn(start-t.Range.Start, data)
		}()
		if err != nil {
			return err
		}
		start += length
	}
	return nil
}

// WithLayer maps all tensors in a layer for the duration of fn.
// The slices are read-only and become invalid when fn returns.
func (r *Reader) WithLayer(layer ggufindex.Layer, fn func(tensors []MappedTensor) error) (err error) {
	if fn == nil {
		return fmt.Errorf("nil callback")
	}

	tensors := make([]MappedTensor, 0, len(layer.Tensors))
	var unmaps []func() error
	defer func() {
		for i := len(unmaps) - 1; i >= 0; i-- {
			err = errors.Join(err, unmaps[i]())
		}
	}()

	for _, t := range layer.Tensors {
		data, unmap, err := r.mapRange(t.Range)
		if err != nil {
			return fmt.Errorf("layer %d, tensor %q: %w", layer.Number, t.Name, err)
		}
		unmaps = append(unmaps, unmap)
		tensors = append(tensors, MappedTensor{Tensor: t, Data: data})
	}
	return fn(tensors)
}

// MappedTensor pairs tensor metadata with its temporarily mapped bytes.
type MappedTensor struct {
	Tensor ggufindex.Tensor
	Data   []byte // Valid only during the callback; do not modify.
}

func (r *Reader) mapRange(span ggufindex.Range) ([]byte, func() error, error) {
	f, err := r.validateRange(span)
	if err != nil {
		return nil, nil, err
	}
	return mapFileRange(f, span, false)
}

func (r *Reader) validateRange(span ggufindex.Range) (*os.File, error) {
	f := r.files[span.File]
	if f == nil {
		return nil, fmt.Errorf("file %q is not open", span.File)
	}
	if span.End <= span.Start {
		return nil, fmt.Errorf("empty or invalid range")
	}

	info, err := f.Stat()
	if err != nil {
		return nil, err
	}
	if info.Size() < 0 || span.End > uint64(info.Size()) {
		return nil, fmt.Errorf("range exceeds file size")
	}
	return f, nil
}

func mapFileRange(f *os.File, span ggufindex.Range, sequential bool) ([]byte, func() error, error) {
	pageSize := uint64(os.Getpagesize())
	alignedStart := span.Start / pageSize * pageSize
	prefix := span.Start - alignedStart
	length := span.End - alignedStart
	maxInt := uint64(^uint(0) >> 1)
	if alignedStart > uint64(^uint64(0)>>1) || length > maxInt {
		return nil, nil, fmt.Errorf("mapping exceeds platform limits")
	}

	mapped, err := syscall.Mmap(
		int(f.Fd()), int64(alignedStart), int(length),
		syscall.PROT_READ, syscall.MAP_PRIVATE,
	)
	if err != nil {
		return nil, nil, err
	}
	if sequential {
		// Readahead is only a hint; failure does not prevent reading the mapping.
		_ = syscall.Madvise(mapped, syscall.MADV_SEQUENTIAL)
	}

	data := mapped[int(prefix):int(length):int(length)]
	return data, func() error { return syscall.Munmap(mapped) }, nil
}
