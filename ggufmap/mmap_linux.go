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
	files            map[string]*os.File
	fileSizes        map[string]uint64
	mu               sync.Mutex
	used             bool
	closed           bool
	activeCalls      uint64
	expertOptions    ExpertCacheOptions
	expertStats      ExpertCacheStats
	expertCache      map[ggufindex.Range]*cacheEntry
	frequencies      map[ggufindex.Range]uint64
	cacheClock       uint64
	lockPages        func([]byte) error
	residentRanges   map[ggufindex.Range]uint64
	residentByFile   map[string][]ggufindex.Range
	residentReserved uint64
	expertLookahead  bool
}

// DefaultExpertCoalesceBytes bounds selected bytes in a coalesced mapping.
// A single expert larger than this limit is mapped on its own.
const DefaultExpertCoalesceBytes uint64 = 32 << 20

// DefaultExpertPrefetchBytes bounds each explicit selected-run readahead hint.
// Linux can truncate a larger WILLNEED request to its device readahead window.
const DefaultExpertPrefetchBytes = 128 << 10

// DefaultExpertCacheBytes is the recommended shared retention budget for callers.
// Open leaves retention disabled until ConfigureExpertCache is called.
const DefaultExpertCacheBytes uint64 = 2 << 30

const DefaultExpertLookaheadBytes uint64 = 32 << 20

// Cover the complete selected run by default. Limiting advice while using
// MADV_RANDOM leaves the remaining pages to small synchronous storage reads.
const DefaultExpertPrefetchLimitBytes uint64 = ^uint64(0)
const DefaultExpertAgingInterval uint64 = 1024

// ExpertCacheOptions enables LFU retention with periodic frequency halving and
// LRU tie-breaking. MaxBytes includes resident tensors and full page-rounded
// mappings. Active leases and resident tensors cannot be evicted. MinUses is the
// admission threshold for each selected range (zero means one). Pinning is best
// effort: mlock failures never prevent retention. Zero tuning values use the
// defaults above. LookaheadRuns is the number of runs prepared ahead of the
// current run (zero means one; values outside 0..8 are rejected). LookaheadBytes
// bounds their combined page footprint, not the current run or concurrent calls.
// PrefetchBytes bounds advice per new mapping.
// The budget limits retained mappings, not transient mappings or OS page cache.
type ExpertCacheOptions struct {
	MaxBytes        uint64
	MinUses         uint64
	TrackStats      bool
	AgingInterval   uint64
	LookaheadBytes  uint64
	LookaheadRuns   int
	PrefetchBytes   uint64
	DisablePrefetch bool
}

// ExpertCacheStats is an independent snapshot. MappedBytes counts selected
// bytes in newly created mappings; MappedPageBytes counts their page footprint.
// CacheHits counts reused runs. RangeUses counts validated selected ranges and
// is populated only with TrackStats or an enabled cache. Counters are cumulative,
// while CachedRuns and CachedBytes describe current retention.
// SelectedRanges and SelectedBytes count complete validated selections, including
// cache hits and ranges not visited after a callback error. Invalid calls add none.
type ExpertCacheStats struct {
	RangeUses              map[ggufindex.Range]uint64
	SelectedRanges         uint64
	SelectedBytes          uint64
	MappedRuns             uint64
	MappedBytes            uint64
	MappedPageBytes        uint64
	CacheHits              uint64
	CachedRuns             uint64
	CachedBytes            uint64
	LockFailures           uint64
	PrefetchCalls          uint64
	PrefetchFailures       uint64
	BudgetBytes            uint64
	RetainedBytes          uint64
	PinnedBytes            uint64
	Evictions              uint64
	EvictedBytes           uint64
	AdmissionFailures      uint64
	AgingPasses            uint64
	ActiveLeases           uint64
	ResidentRuns           uint64
	ResidentBytes          uint64
	ResidentHits           uint64
	ResidentBudgetBytes    uint64
	ResidentConfiguredRuns uint64
}

// ConfigureExpertCache must be called before any range is used on this Reader.
func (r *Reader) ConfigureExpertCache(options ExpertCacheOptions) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.used || r.closed || r.residentRanges != nil {
		return fmt.Errorf("expert cache must be configured before use")
	}
	if options.LookaheadRuns < 0 || options.LookaheadRuns > 8 {
		return fmt.Errorf("expert lookahead runs must be between 0 and 8")
	}
	r.expertOptions = options
	r.expertStats = ExpertCacheStats{BudgetBytes: options.MaxBytes}
	if options.MaxBytes != 0 || options.TrackStats {
		r.expertStats.RangeUses = make(map[ggufindex.Range]uint64)
	}
	if options.MaxBytes != 0 {
		r.frequencies = make(map[ggufindex.Range]uint64)
	} else {
		r.frequencies = nil
	}
	r.expertCache = nil
	return nil
}

// ConfigureExpertLookahead controls selected-run prefetch lookahead (on by default).
// Configure before use. At most LookaheadRuns plus one runs are mapped per call.
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

	r := &Reader{files: make(map[string]*os.File, len(model.Paths)), fileSizes: make(map[string]uint64, len(model.Paths)), expertLookahead: true, lockPages: syscall.Mlock}
	for _, path := range model.Paths {
		if r.files[path] != nil {
			continue
		}
		f, err := os.Open(path)
		if err != nil {
			r.Close()
			return nil, fmt.Errorf("open %q: %w", path, err)
		}
		r.files[path] = f
		info, err := f.Stat()
		if err != nil || info.Size() < 0 {
			r.Close()
			return nil, fmt.Errorf("stat %q: %w", path, errors.Join(err, fmt.Errorf("invalid file size")))
		}
		r.fileSizes[path] = uint64(info.Size())
	}
	return r, nil
}

// Close rejects new calls immediately. Existing callbacks keep their mappings
// and files alive until their leases/calls finish, including Close from a callback.
// It does not wait for callbacks; deferred cleanup errors are returned by those calls.
func (r *Reader) Close() error {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.closed = true
	var err error
	for _, entry := range r.expertCache {
		if entry.refs == 0 {
			err = errors.Join(err, r.releaseEntryLocked(entry))
		}
	}
	if r.activeCalls == 0 {
		err = errors.Join(err, r.closeFilesLocked())
	}
	return err
}

// WithTensor provides the tensor's bytes for the duration of fn.
// data is read-only: fn must never modify it or retain it after returning.
func (r *Reader) WithTensor(t ggufindex.Tensor, fn func(data []byte) error) (err error) {
	if fn == nil {
		return fmt.Errorf("nil callback")
	}
	if err := r.beginCall(); err != nil {
		return err
	}
	defer func() { err = errors.Join(err, r.endCall()) }()

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
// Registered resident tensors reuse one leased mapping for all chunks.
// Files must remain immutable.
func (r *Reader) WithTensorChunks(t ggufindex.Tensor, chunkBytes uint64, fn func(offset uint64, data []byte) error) (err error) {
	if fn == nil {
		return fmt.Errorf("nil callback")
	}
	if chunkBytes == 0 {
		return fmt.Errorf("zero chunk size")
	}
	if err := r.beginCall(); err != nil {
		return err
	}
	defer func() { err = errors.Join(err, r.endCall()) }()
	_, err = r.validateRange(t.Range)
	if err != nil {
		return fmt.Errorf("tensor %q: %w", t.Name, err)
	}
	r.mu.Lock()
	data, release, resident, residentErr := r.residentLeaseLocked(t.Range)
	r.mu.Unlock()
	if residentErr != nil {
		return fmt.Errorf("tensor %q: %w", t.Name, residentErr)
	}
	if resident {
		defer func() { err = errors.Join(err, release()) }()
		for offset := uint64(0); offset < uint64(len(data)); {
			end := offset + min(chunkBytes, uint64(len(data))-offset)
			if err := fn(offset, data[int(offset):int(end):int(end)]); err != nil {
				return err
			}
			offset = end
		}
		return nil
	}
	for start := t.Range.Start; start < t.Range.End; {
		length := t.Range.End - start
		if length > chunkBytes {
			length = chunkBytes
		}
		span := ggufindex.Range{File: t.Range.File, Start: start, End: start + length}
		data, unmap, err := r.mapRange(span)
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
	r.mu.Lock()
	defer r.mu.Unlock()
	f, err := r.validateRangeLocked(span)
	if err != nil {
		return nil, nil, err
	}
	if data, release, ok, err := r.residentLeaseLocked(span); ok || err != nil {
		return data, release, err
	}
	entry, err := r.newEntryLocked(f, span, false)
	if err != nil {
		return nil, nil, err
	}
	data, release := r.leaseLocked(entry, span)
	return data, release, nil
}

func (r *Reader) validateRange(span ggufindex.Range) (*os.File, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.used = true
	return r.validateRangeLocked(span)
}

func (r *Reader) validateRangeLocked(span ggufindex.Range) (*os.File, error) {
	f := r.files[span.File]
	if f == nil {
		return nil, fmt.Errorf("file %q is not open", span.File)
	}
	if span.End <= span.Start {
		return nil, fmt.Errorf("empty or invalid range")
	}

	if span.End > r.fileSizes[span.File] {
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
	if span.End <= span.Start {
		return 0, 0, fmt.Errorf("empty or invalid range")
	}
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
// read. Calls, statistics and Close may run concurrently.
// By default the next selected run is prefetched before the current callbacks,
// overlapping its kernel I/O with current computation. LookaheadRuns controls
// queue depth; LookaheadBytes bounds its combined page footprint. An oversized
// next run stops lookahead without skipping it. All leases are released on
// callback error or panic; admitted mappings remain in the optional hot cache.
// An empty selection succeeds without invoking fn, which may then be nil.
func (r *Reader) WithExpertRanges(ranges []ggufindex.Range, fn func(index int, data []byte) error) (err error) {
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
	if err := r.beginCall(); err != nil {
		return err
	}
	defer func() { err = errors.Join(err, r.endCall()) }()
	r.mu.Lock()
	lookahead := r.expertLookahead
	lookaheadRuns := r.expertOptions.LookaheadRuns
	if lookaheadRuns == 0 {
		lookaheadRuns = 1
	}
	lookaheadBytes := r.expertOptions.LookaheadBytes
	if lookaheadBytes == 0 {
		lookaheadBytes = DefaultExpertLookaheadBytes
	}
	r.mu.Unlock()
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
		r.recordUseLocked(item.span)
	}
	r.mu.Unlock()
	type expertRun struct {
		span       ggufindex.Range
		first, end int
		pageBytes  uint64
		data       []byte
		release    func() error
	}
	runBounds := func(first int) (ggufindex.Range, int) {
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
		return run, end
	}
	prepare := func(first int) (expertRun, error) {
		run, end := runBounds(first)
		pageBytes, err := mappingPageBytes(run)
		if err != nil {
			return expertRun{}, err
		}
		hot := false
		frequency := ^uint64(0)
		r.mu.Lock()
		if r.expertOptions.MaxBytes != 0 {
			hot = true
			for _, item := range selected[first:end] {
				frequency = min(frequency, r.frequencies[item.span])
				if r.expertStats.RangeUses[item.span] < r.expertOptions.MinUses {
					hot = false
				}
			}
		}
		r.mu.Unlock()
		data, release, err := r.mapExpertRun(selected[first].file, run, hot, frequency)
		if err != nil {
			return expertRun{}, fmt.Errorf("expert run %v: %w", run, err)
		}
		return expertRun{span: run, first: first, end: end, pageBytes: pageBytes, data: data, release: release}, nil
	}
	return func() (err error) {
		var current expertRun
		queue := make([]expertRun, 0, lookaheadRuns)
		var queuedBytes uint64
		defer func() {
			if current.release != nil {
				err = errors.Join(err, current.release())
			}
			for _, run := range queue {
				err = errors.Join(err, run.release())
			}
		}()
		for first := 0; first < len(selected); {
			if len(queue) != 0 {
				current = queue[0]
				queuedBytes -= current.pageBytes
				copy(queue, queue[1:])
				queue[len(queue)-1] = expertRun{}
				queue = queue[:len(queue)-1]
			} else {
				current, err = prepare(first)
				if err != nil {
					return err
				}
			}
			if lookahead {
				nextFirst := current.end
				if len(queue) != 0 {
					nextFirst = queue[len(queue)-1].end
				}
				for len(queue) < lookaheadRuns && nextFirst < len(selected) {
					span, _ := runBounds(nextFirst)
					pageBytes, boundsErr := mappingPageBytes(span)
					if boundsErr != nil || pageBytes > lookaheadBytes-queuedBytes {
						break
					}
					next, prepareErr := prepare(nextFirst)
					err = prepareErr
					if err != nil {
						return err
					}
					queue = append(queue, next)
					queuedBytes += pageBytes
					nextFirst = next.end
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

func (r *Reader) mapExpertRun(f *os.File, span ggufindex.Range, hot bool, frequency uint64) ([]byte, func() error, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if data, release, ok, err := r.residentLeaseLocked(span); ok || err != nil {
		return data, release, err
	}
	if entry := r.expertCache[span]; entry != nil {
		r.expertStats.CacheHits++
		entry.frequency = increment(entry.frequency)
		entry.lastUse = r.cacheClock
		data, release := r.leaseLocked(entry, span)
		return data, release, nil
	}
	entry, err := r.newEntryLocked(f, span, false)
	if err != nil {
		return nil, nil, err
	}
	r.prefetchLocked(entry.mapped)
	r.expertStats.MappedRuns++
	r.expertStats.MappedBytes += span.End - span.Start
	r.expertStats.MappedPageBytes += entry.pageBytes
	entry.frequency = max(1, frequency)
	entry.lastUse = r.cacheClock
	if hot && !r.closed {
		room, roomErr := r.makeRoomLocked(entry.pageBytes, entry.frequency)
		if roomErr != nil {
			return nil, nil, errors.Join(roomErr, r.releaseEntryLocked(entry))
		}
		if room {
			r.retainLocked(entry)
		} else {
			r.expertStats.AdmissionFailures++
		}
	}
	data, release := r.leaseLocked(entry, span)
	return data, release, nil
}
