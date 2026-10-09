package ggufmmap

import (
	"errors"
	"fmt"
	"os"
	"sort"
	"syscall"

	"Stream-PT/ggufindex"
)

type cacheEntry struct {
	span      ggufindex.Range
	mapped    []byte
	pageBytes uint64
	refs      uint64
	resident  bool
	retained  bool
	pinned    bool
	frequency uint64
	lastUse   uint64
}

// ConfigureResidentTensors reserves part of MaxBytes for base weights before use.
// Tensor order defines priority. Valid tensors that do not fit maxResidentBytes
// are skipped; exact duplicates count once. Overlapping accepted ranges are rejected.
// The cap is clamped to MaxBytes; zero reserves nothing. Full page-rounded mmap
// footprints count, including leading alignment. No pages are mapped or pinned
// here: accepted tensors are retained lazily on first access, then never evicted.
// The reserved portion is unavailable to experts even before those accesses.
// WithTensor, WithTensorChunks and WithExpertRanges reuse resident mappings,
// including contained subranges. mlock failure does not prevent retention.
func (r *Reader) ConfigureResidentTensors(tensors []ggufindex.Tensor, maxResidentBytes uint64) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.used || r.closed || r.residentRanges != nil {
		return fmt.Errorf("resident tensors must be configured once before use")
	}
	limit := min(maxResidentBytes, r.expertOptions.MaxBytes)
	selected := make(map[ggufindex.Range]uint64)
	byFile := make(map[string][]ggufindex.Range)
	var reserved uint64
	for _, tensor := range tensors {
		span := tensor.Range
		if _, err := r.validateRangeLocked(span); err != nil {
			return fmt.Errorf("resident tensor %q: %w", tensor.Name, err)
		}
		pageBytes, err := mappingPageBytes(span)
		if err != nil {
			return fmt.Errorf("resident tensor %q: %w", tensor.Name, err)
		}
		if _, exists := selected[span]; exists || pageBytes > limit-reserved {
			continue
		}
		selected[span] = pageBytes
		byFile[span.File] = append(byFile[span.File], span)
		reserved += pageBytes
	}
	for _, spans := range byFile {
		sort.Slice(spans, func(i, j int) bool { return spans[i].Start < spans[j].Start })
		for i := 1; i < len(spans); i++ {
			if spans[i].Start < spans[i-1].End {
				return fmt.Errorf("overlapping resident ranges %v and %v", spans[i-1], spans[i])
			}
		}
	}
	r.residentRanges = selected
	r.residentByFile = byFile
	r.residentReserved = reserved
	r.expertStats.ResidentBudgetBytes = reserved
	r.expertStats.ResidentConfiguredRuns = uint64(len(selected))
	return nil
}

func mappingPageBytes(span ggufindex.Range) (uint64, error) {
	_, length, err := mappingBounds(span)
	if err != nil {
		return 0, err
	}
	page := uint64(os.Getpagesize())
	if length > ^uint64(0)-(page-1) {
		return 0, fmt.Errorf("page footprint overflow")
	}
	return (length + page - 1) / page * page, nil
}

func (r *Reader) beginCall() error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return fmt.Errorf("reader is closed")
	}
	r.used = true
	r.activeCalls++
	return nil
}

func (r *Reader) endCall() error {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.activeCalls--
	if r.closed && r.activeCalls == 0 {
		return r.closeFilesLocked()
	}
	return nil
}

func (r *Reader) closeFilesLocked() error {
	var err error
	for path, f := range r.files {
		if closeErr := f.Close(); closeErr != nil {
			err = errors.Join(err, fmt.Errorf("close %q: %w", path, closeErr))
		}
		delete(r.files, path)
	}
	return err
}

func (r *Reader) newEntryLocked(f *os.File, span ggufindex.Range, resident bool) (*cacheEntry, error) {
	pageBytes, err := mappingPageBytes(span)
	if err != nil {
		return nil, err
	}
	mapped, _, err := mmapFileRange(f, span)
	if err != nil {
		return nil, err
	}
	return &cacheEntry{span: span, mapped: mapped, pageBytes: pageBytes, resident: resident}, nil
}

func (r *Reader) residentLeaseLocked(span ggufindex.Range) ([]byte, func() error, bool, error) {
	resident := span
	if _, exact := r.residentRanges[span]; !exact {
		spans := r.residentByFile[span.File]
		index := sort.Search(len(spans), func(i int) bool { return spans[i].Start > span.Start }) - 1
		if index < 0 || spans[index].End < span.End {
			return nil, nil, false, nil
		}
		resident = spans[index]
	}
	entry := r.expertCache[resident]
	if entry == nil {
		var err error
		entry, err = r.newEntryLocked(r.files[resident.File], resident, true)
		if err != nil {
			return nil, nil, false, err
		}
		if !r.closed {
			r.retainLocked(entry)
		}
	} else {
		r.expertStats.ResidentHits++
	}
	data, release := r.leaseLocked(entry, span)
	return data, release, true, nil
}

func (r *Reader) leaseLocked(entry *cacheEntry, span ggufindex.Range) ([]byte, func() error) {
	entry.refs++
	r.expertStats.ActiveLeases++
	start := span.Start - entry.span.Start + entry.span.Start%uint64(os.Getpagesize())
	end := start + span.End - span.Start
	data := entry.mapped[int(start):int(end):int(end)]
	released := false
	return data, func() error {
		r.mu.Lock()
		defer r.mu.Unlock()
		if released {
			return nil
		}
		released = true
		entry.refs--
		r.expertStats.ActiveLeases--
		if entry.refs == 0 && (!entry.retained || r.closed) {
			return r.releaseEntryLocked(entry)
		}
		return nil
	}
}

func (r *Reader) retainLocked(entry *cacheEntry) {
	if r.expertCache == nil {
		r.expertCache = make(map[ggufindex.Range]*cacheEntry)
	}
	entry.retained = true
	r.expertCache[entry.span] = entry
	r.expertStats.RetainedBytes += entry.pageBytes
	if entry.resident {
		r.expertStats.ResidentRuns++
		r.expertStats.ResidentBytes += entry.pageBytes
	} else {
		r.expertStats.CachedRuns++
		r.expertStats.CachedBytes += entry.pageBytes
	}
	if err := r.lockPages(entry.mapped); err != nil {
		r.expertStats.LockFailures++
	} else {
		entry.pinned = true
		r.expertStats.PinnedBytes += entry.pageBytes
	}
}

func (r *Reader) releaseEntryLocked(entry *cacheEntry) error {
	var err error
	if entry.pinned {
		err = syscall.Munlock(entry.mapped)
		r.expertStats.PinnedBytes -= entry.pageBytes
		entry.pinned = false
	}
	err = errors.Join(err, syscall.Munmap(entry.mapped))
	entry.mapped = nil
	if entry.retained {
		delete(r.expertCache, entry.span)
		r.expertStats.RetainedBytes -= entry.pageBytes
		if entry.resident {
			r.expertStats.ResidentRuns--
			r.expertStats.ResidentBytes -= entry.pageBytes
		} else {
			r.expertStats.CachedRuns--
			r.expertStats.CachedBytes -= entry.pageBytes
		}
		entry.retained = false
	}
	return err
}

func increment(value uint64) uint64 {
	if value == ^uint64(0) {
		return value
	}
	return value + 1
}

func (r *Reader) recordUseLocked(span ggufindex.Range) {
	if r.frequencies == nil {
		return
	}
	r.cacheClock = increment(r.cacheClock)
	interval := r.expertOptions.AgingInterval
	if interval == 0 {
		interval = DefaultExpertAgingInterval
	}
	if r.cacheClock%interval == 0 {
		for key, frequency := range r.frequencies {
			if frequency <= 1 {
				delete(r.frequencies, key)
			} else {
				r.frequencies[key] = frequency / 2
			}
		}
		for _, entry := range r.expertCache {
			entry.frequency /= 2
		}
		r.expertStats.AgingPasses++
	}
	r.frequencies[span] = increment(r.frequencies[span])
}

func (r *Reader) makeRoomLocked(pageBytes, frequency uint64) (bool, error) {
	limit := r.expertOptions.MaxBytes - r.residentReserved
	if pageBytes > limit {
		return false, nil
	}
	if r.expertStats.CachedBytes <= limit-pageBytes {
		return true, nil
	}
	needed := r.expertStats.CachedBytes - (limit - pageBytes)
	var reclaimable uint64
	for _, entry := range r.expertCache {
		if !entry.resident && entry.refs == 0 && entry.frequency <= frequency {
			reclaimable += entry.pageBytes
		}
	}
	if reclaimable < needed {
		return false, nil
	}
	for r.expertStats.CachedBytes > limit-pageBytes {
		var victim *cacheEntry
		for _, entry := range r.expertCache {
			if entry.resident || entry.refs != 0 || entry.frequency > frequency {
				continue
			}
			if victim == nil || entry.frequency < victim.frequency ||
				(entry.frequency == victim.frequency && entry.lastUse < victim.lastUse) {
				victim = entry
			}
		}
		if err := r.releaseEntryLocked(victim); err != nil {
			return false, err
		}
		r.expertStats.Evictions++
		r.expertStats.EvictedBytes += victim.pageBytes
	}
	return true, nil
}

func (r *Reader) prefetchLocked(mapped []byte) {
	_ = syscall.Madvise(mapped, syscall.MADV_RANDOM)
	if r.expertOptions.DisablePrefetch {
		return
	}
	limit := r.expertOptions.PrefetchBytes
	if limit == 0 {
		limit = DefaultExpertPrefetchLimitBytes
	}
	page := uint64(os.Getpagesize())
	limit = limit / page * page
	if limit == 0 {
		return
	}
	step := max(DefaultExpertPrefetchBytes, os.Getpagesize())
	end := int(min(uint64(len(mapped)), limit))
	for offset := 0; offset < end; {
		stop := offset + min(step, end-offset)
		r.expertStats.PrefetchCalls++
		if err := syscall.Madvise(mapped[offset:stop], syscall.MADV_WILLNEED); err != nil {
			r.expertStats.PrefetchFailures++
		}
		offset = stop
	}
}
