package ggufmmap

import (
	"errors"
	"fmt"
	"os"
	"sort"
	"sync"
	"syscall"

	"Stream-PT/ggufindex"
)

// Reader keeps file descriptors open without loading model weights into the Go heap.
// Files must remain immutable while the Reader is in use: truncating a mapped
// file can cause a fatal fault. Mapped pages still consume physical RAM through
// the OS page cache, even though they are not allocated on the Go heap.
type Reader struct {
	files           map[string]*os.File
	mu              sync.Mutex
	used            bool
	closed          bool
	expertOptions   ExpertCacheOptions
	expertStats     ExpertCacheStats
	expertCache     map[ggufindex.Range][]byte
	expertLookahead bool
}

// DefaultExpertCoalesceBytes bounds selected bytes in a coalesced mapping.
// A single expert larger than this limit is mapped on its own.
const DefaultExpertCoalesceBytes uint64 = 32 << 20

// DefaultExpertPrefetchBytes bounds each explicit selected-run readahead hint.
// Linux can truncate a larger WILLNEED request to its device readahead window.
const DefaultExpertPrefetchBytes = 128 << 10

// ExpertCacheOptions enables optional hot-run retention. MaxBytes is charged
// for the full page-rounded mapping, including leading alignment. MinUses is
// the admission threshold for every expert in a run (zero means one).
// Admission is first come, with no eviction. Failed mlock is nonfatal and the
// mapping is not retained. Only exactly adjacent selected ranges are retained;
// page alignment may necessarily include bytes outside the requested ranges.
// With both MaxBytes and TrackStats zero, no per-range history is kept.
type ExpertCacheOptions struct {
	MaxBytes   uint64
	MinUses    uint64
	TrackStats bool
}

// ExpertCacheStats is an independent snapshot. MappedBytes counts selected
// bytes in newly created mappings; MappedPageBytes counts their page footprint.
// CacheHits counts reused runs. RangeUses counts validated selected ranges and
// is populated only with TrackStats or an enabled cache. Counters are cumulative,
// while CachedRuns and CachedBytes describe current retention.
// SelectedRanges and SelectedBytes count complete validated selections, including
// cache hits and ranges not visited after a callback error. Invalid calls add none.
type ExpertCacheStats struct {
	RangeUses        map[ggufindex.Range]uint64
	SelectedRanges   uint64
	SelectedBytes    uint64
	MappedRuns       uint64
	MappedBytes      uint64
	MappedPageBytes  uint64
	CacheHits        uint64
	CachedRuns       uint64
	CachedBytes      uint64
	LockFailures     uint64
	PrefetchCalls    uint64
	PrefetchFailures uint64
}

// ConfigureExpertCache must be called before any range is used on this Reader.
func (r *Reader) ConfigureExpertCache(options ExpertCacheOptions) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.used || r.closed {
		return fmt.Errorf("expert cache must be configured before use")
	}
	r.expertOptions = options
	r.expertStats = ExpertCacheStats{}
	if options.MaxBytes != 0 || options.TrackStats {
		r.expertStats.RangeUses = make(map[ggufindex.Range]uint64)
	}
	r.expertCache = nil
	return nil
}

// ConfigureExpertLookahead controls one selected run of prefetch lookahead (on by default).
// Configure before use. At most two transient expert runs are mapped per call.
func (r *Reader) ConfigureExpertLookahead(enabled bool) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.used || r.closed {
		return fmt.Errorf("expert lookahead must be configured before use")
	}
	r.expertLookahead = enabled
	return nil
}

// ExpertCacheStats returns a snapshot safe to inspect or modify independently.
func (r *Reader) ExpertCacheStats() ExpertCacheStats {
	r.mu.Lock()
	defer r.mu.Unlock()
	stats := r.expertStats
	if stats.RangeUses != nil {
		stats.RangeUses = make(map[ggufindex.Range]uint64, len(r.expertStats.RangeUses))
		for span, uses := range r.expertStats.RangeUses {
			stats.RangeUses[span] = uses
		}
	}
	return stats
}

// Open opens the files referenced by an existing GGUF index.
func Open(model *ggufindex.Model) (*Reader, error) {
	if model == nil {
		return nil, fmt.Errorf("nil model")
	}

	r := &Reader{files: make(map[string]*os.File, len(model.Paths)), expertLookahead: true}
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
// WithTensorChunks, or WithExpertRanges (including from callbacks).
func (r *Reader) Close() error {
	r.mu.Lock()
	defer r.mu.Unlock()
	var first error
	for span, mapped := range r.expertCache {
		if err := errors.Join(syscall.Munlock(mapped), syscall.Munmap(mapped)); err != nil && first == nil {
			first = fmt.Errorf("release cached range %v: %w", span, err)
		}
		delete(r.expertCache, span)
	}
	r.expertStats.CachedRuns = 0
	r.expertStats.CachedBytes = 0
	r.closed = true
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

func (r *Reader) mapRange(span ggufindex.Range) ([]byte, func() error, error) {
	f, err := r.validateRange(span)
	if err != nil {
		return nil, nil, err
	}
	return mapFileRange(f, span, false)
}

func (r *Reader) validateRange(span ggufindex.Range) (*os.File, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.used = true
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
	if _, _, err := mappingBounds(span); err != nil {
		return nil, err
	}
	return f, nil
}

func mapFileRange(f *os.File, span ggufindex.Range, sequential bool) ([]byte, func() error, error) {
	mapped, prefix, err := mmapFileRange(f, span)
	if err != nil {
		return nil, nil, err
	}
	if sequential {
		// Readahead is only a hint; failure does not prevent reading the mapping.
		_ = syscall.Madvise(mapped, syscall.MADV_SEQUENTIAL)
	}
	data := mapped[int(prefix):len(mapped):len(mapped)]
	return data, func() error { return syscall.Munmap(mapped) }, nil
}

func mappingBounds(span ggufindex.Range) (uint64, uint64, error) {
	pageSize := uint64(os.Getpagesize())
	alignedStart := span.Start / pageSize * pageSize
	length := span.End - alignedStart
	maxInt := uint64(^uint(0) >> 1)
	if alignedStart > uint64(^uint64(0)>>1) || length > maxInt {
		return 0, 0, fmt.Errorf("mapping exceeds platform limits")
	}
	return alignedStart, length, nil
}

func mmapFileRange(f *os.File, span ggufindex.Range) ([]byte, uint64, error) {
	alignedStart, length, err := mappingBounds(span)
	if err != nil {
		return nil, 0, err
	}
	mapped, err := syscall.Mmap(
		int(f.Fd()), int64(alignedStart), int(length),
		syscall.PROT_READ, syscall.MAP_PRIVATE,
	)
	if err != nil {
		return nil, 0, err
	}
	return mapped, span.Start - alignedStart, nil
}

// WithExpertRanges validates all ranges before invoking fn, rejects duplicates
// and overlaps, and visits ranges in lexical file / ascending offset order.
// index is the original input index. Slices are read-only, capacity-capped, and
// valid only during their callback, even when caching is enabled. Adjacent
// selected ranges share a mapping up to DefaultExpertCoalesceBytes; gaps are
// never bridged. MADV_RANDOM avoids speculative readahead. No whole tensor is
// read. Calls and statistics may run concurrently, but Close must not.
// By default the next selected run is prefetched before the current callbacks,
// overlapping its kernel I/O with current computation. Both mappings are released
// on callback error or panic unless admitted to the optional hot cache.
// An empty selection succeeds without invoking fn, which may then be nil.
func (r *Reader) WithExpertRanges(ranges []ggufindex.Range, fn func(index int, data []byte) error) error {
	if len(ranges) == 0 {
		return nil
	}
	if fn == nil {
		return fmt.Errorf("nil callback")
	}
	type selectedRange struct {
		span  ggufindex.Range
		index int
		file  *os.File
	}
	r.mu.Lock()
	r.used = true
	closed := r.closed
	lookahead := r.expertLookahead
	r.mu.Unlock()
	if closed {
		return fmt.Errorf("reader is closed")
	}
	selected := make([]selectedRange, len(ranges))
	for index, span := range ranges {
		f, err := r.validateRange(span)
		if err != nil {
			return fmt.Errorf("expert range %d: %w", index, err)
		}
		selected[index] = selectedRange{span: span, index: index, file: f}
	}
	sort.Slice(selected, func(i, j int) bool {
		a, b := selected[i].span, selected[j].span
		if a.File != b.File {
			return a.File < b.File
		}
		return a.Start < b.Start
	})
	for i := 1; i < len(selected); i++ {
		a, b := selected[i-1].span, selected[i].span
		if a.File == b.File && b.Start < a.End {
			return fmt.Errorf("overlapping expert ranges %d and %d", selected[i-1].index, selected[i].index)
		}
	}
	r.mu.Lock()
	for _, item := range selected {
		r.expertStats.SelectedRanges++
		r.expertStats.SelectedBytes += item.span.End - item.span.Start
		if r.expertStats.RangeUses != nil {
			r.expertStats.RangeUses[item.span]++
		}
	}
	r.mu.Unlock()
	type expertRun struct {
		span       ggufindex.Range
		first, end int
		data       []byte
		release    func() error
	}
	prepare := func(first int) (expertRun, error) {
		end := first + 1
		run := selected[first].span
		for end < len(selected) {
			next := selected[end].span
			if next.File != run.File || next.Start != run.End || next.End-run.Start > DefaultExpertCoalesceBytes {
				break
			}
			run.End = next.End
			end++
		}
		hot := false
		r.mu.Lock()
		if r.expertOptions.MaxBytes != 0 {
			hot = true
			for _, item := range selected[first:end] {
				if r.expertStats.RangeUses[item.span] < r.expertOptions.MinUses {
					hot = false
				}
			}
		}
		r.mu.Unlock()
		data, release, err := r.mapExpertRun(selected[first].file, run, hot)
		if err != nil {
			return expertRun{}, fmt.Errorf("expert run %v: %w", run, err)
		}
		return expertRun{span: run, first: first, end: end, data: data, release: release}, nil
	}
	return func() (err error) {
		var current, next expertRun
		defer func() {
			if current.release != nil {
				err = errors.Join(err, current.release())
			}
			if next.release != nil {
				err = errors.Join(err, next.release())
			}
		}()
		for first := 0; first < len(selected); {
			if next.release != nil {
				current, next = next, expertRun{}
			} else {
				current, err = prepare(first)
				if err != nil {
					return err
				}
			}
			if lookahead && current.end < len(selected) {
				next, err = prepare(current.end)
				if err != nil {
					return err
				}
			}
			for _, item := range selected[current.first:current.end] {
				start, stop := int(item.span.Start-current.span.Start), int(item.span.End-current.span.Start)
				if err := fn(item.index, current.data[start:stop:stop]); err != nil {
					return err
				}
			}
			err = current.release()
			current.release = nil
			if err != nil {
				return err
			}
			first = current.end
		}
		return nil
	}()
}

func (r *Reader) mapExpertRun(f *os.File, span ggufindex.Range, hot bool) ([]byte, func() error, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	pageSize := uint64(os.Getpagesize())
	prefix := span.Start % pageSize
	if mapped := r.expertCache[span]; mapped != nil {
		r.expertStats.CacheHits++
		return mapped[int(prefix):len(mapped):len(mapped)], func() error { return nil }, nil
	}
	mapped, prefix, err := mmapFileRange(f, span)
	if err != nil {
		return nil, nil, err
	}
	_ = syscall.Madvise(mapped, syscall.MADV_RANDOM)
	// A single large WILLNEED can cover only the beginning of an expert.
	// Keep every hint page-aligned and inside this selected run's mapping.
	step := max(DefaultExpertPrefetchBytes, os.Getpagesize())
	for offset := 0; offset < len(mapped); {
		end := offset + min(step, len(mapped)-offset)
		r.expertStats.PrefetchCalls++
		if err := syscall.Madvise(mapped[offset:end], syscall.MADV_WILLNEED); err != nil {
			r.expertStats.PrefetchFailures++
		}
		offset = end
	}
	pageBytes := (uint64(len(mapped)) + pageSize - 1) / pageSize * pageSize
	r.expertStats.MappedRuns++
	r.expertStats.MappedBytes += span.End - span.Start
	r.expertStats.MappedPageBytes += pageBytes
	release := func() error { return syscall.Munmap(mapped) }
	if hot && pageBytes <= r.expertOptions.MaxBytes-r.expertStats.CachedBytes {
		if err := syscall.Mlock(mapped); err != nil {
			r.expertStats.LockFailures++
		} else {
			if r.expertCache == nil {
				r.expertCache = make(map[ggufindex.Range][]byte)
			}
			r.expertCache[span] = mapped
			r.expertStats.CachedRuns++
			r.expertStats.CachedBytes += pageBytes
			release = func() error { return nil }
		}
	}
	return mapped[int(prefix):len(mapped):len(mapped)], release, nil
}
