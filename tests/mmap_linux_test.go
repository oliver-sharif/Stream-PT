package tests

import (
	"bytes"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"syscall"
	"testing"

	"Stream-PT/ggufindex"
	. "Stream-PT/ggufmap"
)

func syntheticReader(t *testing.T) (*Reader, string, []byte) {
	t.Helper()
	data := make([]byte, 3*os.Getpagesize()+137)
	for i := range data {
		data[i] = byte(i % 251)
	}
	path := filepath.Join(t.TempDir(), "weights.bin")
	if err := os.WriteFile(path, data, 0600); err != nil {
		t.Fatal(err)
	}
	r, err := Open(&ggufindex.Model{Paths: []string{path}})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := r.Close(); err != nil {
			t.Error(err)
		}
	})
	return r, path, data
}

func mappingLines(t *testing.T, path string) []string {
	t.Helper()
	data, err := os.ReadFile("/proc/self/maps")
	if err != nil {
		t.Fatal(err)
	}
	var lines []string
	for _, line := range strings.Split(string(data), "\n") {
		if strings.HasSuffix(line, " "+path) {
			lines = append(lines, line)
		}
	}
	return lines
}

func assertUnmapped(t *testing.T, path string) {
	t.Helper()
	if lines := mappingLines(t, path); len(lines) != 0 {
		t.Fatalf("mappings remain: %v", lines)
	}
}

func TestWithTensorChunks(t *testing.T) {
	r, path, contents := syntheticReader(t)
	for _, chunkBytes := range []uint64{1, 127, uint64(os.Getpagesize()), uint64(len(contents)), ^uint64(0)} {
		t.Run(fmt.Sprint(chunkBytes), func(t *testing.T) {
			span := ggufindex.Range{File: path, Start: 17, End: uint64(len(contents))}
			var offset uint64
			var calls int
			err := r.WithTensorChunks(ggufindex.Tensor{Range: span}, chunkBytes, func(gotOffset uint64, data []byte) error {
				calls++
				if gotOffset != offset || len(data) == 0 || uint64(len(data)) > chunkBytes || cap(data) != len(data) {
					t.Fatalf("offset %d want %d, len %d cap %d, bound %d", gotOffset, offset, len(data), cap(data), chunkBytes)
				}
				if !bytes.Equal(data, contents[span.Start+offset:span.Start+offset+uint64(len(data))]) {
					t.Fatal("incorrect chunk contents")
				}
				lines := mappingLines(t, path)
				if len(lines) != 1 {
					t.Fatalf("want exactly one active mapping, got %v", lines)
				}
				fields := strings.Fields(lines[0])
				bounds := strings.Split(fields[0], "-")
				start, err := strconv.ParseUint(bounds[0], 16, 64)
				if err != nil {
					t.Fatal(err)
				}
				end, err := strconv.ParseUint(bounds[1], 16, 64)
				if err != nil {
					t.Fatal(err)
				}
				pointer := uint64(reflect.ValueOf(data).Pointer())
				if pointer < start || pointer+uint64(len(data)) > end || fields[1] != "r--p" {
					t.Fatalf("slice is not directly backed by a read-only file mapping: %s", lines[0])
				}
				offset += uint64(len(data))
				return nil
			})
			if err != nil {
				t.Fatal(err)
			}
			wantCalls := (span.End-span.Start-1)/chunkBytes + 1
			if offset != span.End-span.Start || uint64(calls) != wantCalls {
				t.Fatalf("read %d bytes in %d calls, want %d calls", offset, calls, wantCalls)
			}
			assertUnmapped(t, path)
		})
	}
}

func TestWithTensorChunksInvalid(t *testing.T) {
	r, path, contents := syntheticReader(t)
	valid := ggufindex.Range{File: path, Start: 1, End: 10}
	for _, tc := range []struct {
		name  string
		span  ggufindex.Range
		chunk uint64
	}{
		{"zero chunk", valid, 0},
		{"empty", ggufindex.Range{File: path, Start: 1, End: 1}, 2},
		{"reversed", ggufindex.Range{File: path, Start: 3, End: 2}, 2},
		{"past EOF", ggufindex.Range{File: path, Start: 1, End: uint64(len(contents) + 1)}, 2},
		{"overflow", ggufindex.Range{File: path, Start: 1, End: ^uint64(0)}, 2},
		{"missing file", ggufindex.Range{File: path + ".missing", End: 10}, 2},
	} {
		t.Run(tc.name, func(t *testing.T) {
			called := false
			err := r.WithTensorChunks(ggufindex.Tensor{Range: tc.span}, tc.chunk, func(uint64, []byte) error {
				called = true
				return nil
			})
			if err == nil || called {
				t.Fatalf("err=%v callback=%v", err, called)
			}
			assertUnmapped(t, path)
		})
	}
	if err := r.WithTensorChunks(ggufindex.Tensor{Range: valid}, 2, nil); err == nil {
		t.Fatal("nil callback accepted")
	}
	if err := r.Close(); err != nil {
		t.Fatal(err)
	}
	if err := r.Close(); err != nil {
		t.Fatal(err)
	}
	if err := r.WithTensorChunks(ggufindex.Tensor{Range: valid}, 2, func(uint64, []byte) error {
		t.Fatal("callback after Close")
		return nil
	}); err == nil {
		t.Fatal("closed reader accepted")
	}
}

func TestMappingCleanup(t *testing.T) {
	r, path, _ := syntheticReader(t)
	tensor := ggufindex.Tensor{Range: ggufindex.Range{File: path, End: uint64(os.Getpagesize())}}
	callbackErr := errors.New("callback failed")
	calls := 0
	if err := r.WithTensorChunks(tensor, 7, func(uint64, []byte) error {
		calls++
		return callbackErr
	}); !errors.Is(err, callbackErr) || calls != 1 {
		t.Fatalf("err=%v calls=%d", err, calls)
	}
	assertUnmapped(t, path)
	if err := r.WithTensor(tensor, func(data []byte) error {
		if cap(data) != len(data) {
			t.Fatal("tensor capacity exceeds length")
		}
		return callbackErr
	}); !errors.Is(err, callbackErr) {
		t.Fatal(err)
	}
	assertUnmapped(t, path)
	ranges := []ggufindex.Range{tensor.Range, {File: path, Start: uint64(2 * os.Getpagesize()), End: uint64(3 * os.Getpagesize())}}
	if err := r.WithExpertRanges(ranges, func(_ int, data []byte) error {
		if cap(data) != len(data) {
			t.Fatal("expert capacity exceeds length")
		}
		return callbackErr
	}); !errors.Is(err, callbackErr) {
		t.Fatal(err)
	}
	assertUnmapped(t, path)
	ranges[1].End = ^uint64(0)
	if err := r.WithExpertRanges(ranges, func(int, []byte) error {
		t.Fatal("callback with invalid ranges")
		return nil
	}); err == nil {
		t.Fatal("invalid ranges accepted")
	}
	assertUnmapped(t, path)
}

func TestMunmapErrors(t *testing.T) {
	r, path, _ := syntheticReader(t)
	tensor := ggufindex.Tensor{Range: ggufindex.Range{File: path, End: uint64(os.Getpagesize())}}
	callbackErr := errors.New("callback failed")
	// Deliberately unmap an aligned, full mapping to provoke cleanup failure.
	callback := func(data []byte) error {
		if err := syscall.Munmap(data); err != nil {
			t.Fatal(err)
		}
		return callbackErr
	}
	for _, run := range []func() error{
		func() error { return r.WithTensor(tensor, callback) },
		func() error {
			return r.WithTensorChunks(tensor, tensor.Range.End, func(_ uint64, data []byte) error { return callback(data) })
		},
		func() error {
			return r.WithExpertRanges([]ggufindex.Range{tensor.Range,
				{File: path, Start: uint64(2 * os.Getpagesize()), End: uint64(3 * os.Getpagesize())}}, func(index int, data []byte) error {
				if index == 0 {
					return nil
				}
				return callback(data)
			})
		},
	} {
		if err := run(); !errors.Is(err, callbackErr) || !errors.Is(err, syscall.EINVAL) {
			t.Fatalf("missing callback or cleanup error: %v", err)
		}
		assertUnmapped(t, path)
	}
}
