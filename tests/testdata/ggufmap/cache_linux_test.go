package ggufmmap

import (
	"bytes"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"syscall"
	"testing"

	"Stream-PT/ggufindex"
)

func cacheReader(t *testing.T, pages int) (*Reader, []ggufindex.Range) {
	t.Helper()
	page := os.Getpagesize()
	path := filepath.Join(t.TempDir(), "weights.bin")
	data := bytes.Repeat([]byte{37}, page*pages)
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
	ranges := make([]ggufindex.Range, pages)
	for i := range ranges {
		ranges[i] = ggufindex.Range{File: path, Start: uint64(i * page), End: uint64((i + 1) * page)}
	}
	return r, ranges
}

func readExpert(t *testing.T, r *Reader, span ggufindex.Range) {
	t.Helper()
	if err := r.WithExpertRanges([]ggufindex.Range{span}, func(_ int, data []byte) error {
		if len(data) != int(span.End-span.Start) || cap(data) != len(data) || data[0] != 37 || data[len(data)-1] != 37 {
			t.Fatal("incorrect mapping")
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
}

func TestCacheEvictsInsteadOfFirstCome(t *testing.T) {
	r, spans := cacheReader(t, 2)
	if err := r.ConfigureExpertCache(ExpertCacheOptions{MaxBytes: uint64(os.Getpagesize())}); err != nil {
		t.Fatal(err)
	}
	readExpert(t, r, spans[0])
	// Repeated use must replace the first run even if pinning is unavailable.
	for i := 0; i < 4; i++ {
		readExpert(t, r, spans[1])
	}
	before := r.ExpertCacheStats()
	readExpert(t, r, spans[1])
	after := r.ExpertCacheStats()
	if before.CachedBytes != uint64(os.Getpagesize()) || after.MappedRuns != before.MappedRuns || after.CacheHits != before.CacheHits+1 {
		t.Fatalf("hot replacement was not retained: before=%+v after=%+v", before, after)
	}
	if after.Evictions != 1 || after.EvictedBytes != uint64(os.Getpagesize()) {
		t.Fatalf("incorrect eviction accounting: %+v", after)
	}
}

func TestCacheRetainsWithoutMlock(t *testing.T) {
	r, spans := cacheReader(t, 2)
	r.lockPages = func([]byte) error { return syscall.EPERM }
	page := uint64(os.Getpagesize())
	if err := r.ConfigureExpertCache(ExpertCacheOptions{MaxBytes: page}); err != nil {
		t.Fatal(err)
	}
	readExpert(t, r, spans[0])
	readExpert(t, r, spans[0])
	stats := r.ExpertCacheStats()
	if stats.LockFailures != 1 || stats.CacheHits != 1 || stats.CachedBytes != page || stats.PinnedBytes != 0 {
		t.Fatalf("lock failure prevented retention: %+v", stats)
	}
}

func TestCacheActiveLeaseBlocksEvictionAndClose(t *testing.T) {
	r, spans := cacheReader(t, 2)
	page := uint64(os.Getpagesize())
	if err := r.ConfigureExpertCache(ExpertCacheOptions{MaxBytes: page}); err != nil {
		t.Fatal(err)
	}
	entered, resume := make(chan struct{}), make(chan struct{})
	done := make(chan error, 1)
	go func() {
		done <- r.WithExpertRanges(spans[:1], func(_ int, data []byte) error {
			close(entered)
			<-resume
			if data[0] != 37 || data[len(data)-1] != 37 {
				return errors.New("active data changed")
			}
			return nil
		})
	}()
	<-entered
	readExpert(t, r, spans[1])
	if stats := r.ExpertCacheStats(); stats.Evictions != 0 || stats.AdmissionFailures != 1 || stats.ActiveLeases != 1 {
		t.Errorf("active mapping was evicted: %+v", stats)
	}
	if err := r.Close(); err != nil {
		t.Error(err)
	}
	if stats := r.ExpertCacheStats(); stats.RetainedBytes != page || stats.ActiveLeases != 1 {
		t.Errorf("Close unmapped an active lease: %+v", stats)
	}
	if err := r.WithTensor(ggufindex.Tensor{Range: spans[0]}, func([]byte) error { return nil }); err == nil {
		t.Error("closed reader accepted a new call")
	}
	close(resume)
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	if stats := r.ExpertCacheStats(); stats.ActiveLeases != 0 || stats.RetainedBytes != 0 || stats.PinnedBytes != 0 {
		t.Fatalf("lease leaked after Close: %+v", stats)
	}
	if len(r.files) != 0 {
		t.Fatal("deferred file close was not performed")
	}
}

func TestCacheAgingReplacesFormerlyHotEntry(t *testing.T) {
	r, spans := cacheReader(t, 2)
	if err := r.ConfigureExpertCache(ExpertCacheOptions{MaxBytes: uint64(os.Getpagesize()), AgingInterval: 4}); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 12; i++ {
		readExpert(t, r, spans[0])
	}
	for i := 0; i < 12; i++ {
		readExpert(t, r, spans[1])
	}
	before := r.ExpertCacheStats()
	readExpert(t, r, spans[1])
	after := r.ExpertCacheStats()
	if before.AgingPasses != 6 || before.Evictions != 1 || after.CacheHits != before.CacheHits+1 {
		t.Fatalf("aging failed: before=%+v after=%+v", before, after)
	}
}

func TestCacheSharedReferencesSurviveCloseUntilLastRelease(t *testing.T) {
	r, spans := cacheReader(t, 1)
	page := uint64(os.Getpagesize())
	if err := r.ConfigureExpertCache(ExpertCacheOptions{MaxBytes: page}); err != nil {
		t.Fatal(err)
	}
	entered := make(chan struct{}, 2)
	resume := []chan struct{}{make(chan struct{}), make(chan struct{})}
	done := make(chan error, 2)
	for worker := range resume {
		go func(worker int) {
			done <- r.WithExpertRanges(spans, func(_ int, data []byte) error {
				entered <- struct{}{}
				<-resume[worker]
				if data[0] != 37 {
					return errors.New("shared lease unmapped early")
				}
				return nil
			})
		}(worker)
	}
	<-entered
	<-entered
	if err := r.Close(); err != nil {
		t.Error(err)
	}
	if stats := r.ExpertCacheStats(); stats.ActiveLeases != 2 || stats.RetainedBytes != page || stats.CacheHits != 1 {
		t.Errorf("shared entry not leased twice: %+v", stats)
	}
	close(resume[0])
	if err := <-done; err != nil {
		t.Error(err)
	}
	if stats := r.ExpertCacheStats(); stats.ActiveLeases != 1 || stats.RetainedBytes != page {
		t.Errorf("first release unmapped shared entry: %+v", stats)
	}
	close(resume[1])
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	if stats := r.ExpertCacheStats(); stats.ActiveLeases != 0 || stats.RetainedBytes != 0 || len(r.files) != 0 {
		t.Fatalf("last release failed cleanup: %+v", stats)
	}
}

func TestCacheLFUEvictsCoolerEntry(t *testing.T) {
	r, spans := cacheReader(t, 3)
	if err := r.ConfigureExpertCache(ExpertCacheOptions{MaxBytes: uint64(2 * os.Getpagesize())}); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 5; i++ {
		readExpert(t, r, spans[0])
	}
	readExpert(t, r, spans[1])
	readExpert(t, r, spans[2])
	before := r.ExpertCacheStats()
	readExpert(t, r, spans[0])
	readExpert(t, r, spans[2])
	after := r.ExpertCacheStats()
	if after.Evictions != 1 || after.MappedRuns != before.MappedRuns || after.CacheHits != before.CacheHits+2 {
		t.Fatalf("LFU did not protect hot entry: before=%+v after=%+v", before, after)
	}
}

func TestResidentPriorityLazyRetentionAndSharedBudget(t *testing.T) {
	r, spans := cacheReader(t, 5)
	r.lockPages = func([]byte) error { return syscall.EPERM }
	page := uint64(os.Getpagesize())
	if err := r.ConfigureExpertCache(ExpertCacheOptions{MaxBytes: 2 * page}); err != nil {
		t.Fatal(err)
	}
	tooLarge := ggufindex.Tensor{Range: ggufindex.Range{File: spans[0].File, Start: page - 1, End: page + 1}}
	base := ggufindex.Tensor{Name: "output.weight", Range: ggufindex.Range{File: spans[0].File, Start: 17, End: page}}
	if err := r.ConfigureResidentTensors([]ggufindex.Tensor{tooLarge, base, base, {Range: spans[2]}}, page); err != nil {
		t.Fatal(err)
	}
	stats := r.ExpertCacheStats()
	if stats.ResidentConfiguredRuns != 1 || stats.ResidentBudgetBytes != page || stats.ResidentBytes != 0 || stats.LockFailures != 0 {
		t.Fatalf("resident configuration not lazy/page-budgeted: %+v", stats)
	}
	readExpert(t, r, spans[3])
	readExpert(t, r, spans[4])
	if err := r.WithTensor(base, func(data []byte) error {
		if len(data) != int(page-17) || cap(data) != len(data) || data[0] != 37 {
			return errors.New("incorrect resident tensor")
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	var total uint64
	if err := r.WithTensorChunks(base, 31, func(offset uint64, data []byte) error {
		if offset != total || cap(data) != len(data) || data[0] != 37 {
			return errors.New("incorrect resident chunks")
		}
		total += uint64(len(data))
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	readExpert(t, r, ggufindex.Range{File: base.Range.File, Start: 31, End: 43})
	stats = r.ExpertCacheStats()
	if total != page-17 || stats.ResidentHits != 2 || stats.ResidentBytes != page || stats.CachedBytes != page || stats.RetainedBytes != 2*page || stats.PinnedBytes != 0 || stats.LockFailures != 3 || stats.Evictions != 1 {
		t.Fatalf("incorrect shared budget/reuse: %+v", stats)
	}
}

func TestResidentCloseFromCallbackAndPanicCleanup(t *testing.T) {
	for _, resident := range []bool{false, true} {
		t.Run(map[bool]string{false: "expert", true: "resident"}[resident], func(t *testing.T) {
			r, spans := cacheReader(t, 2)
			page := uint64(os.Getpagesize())
			if err := r.ConfigureExpertCache(ExpertCacheOptions{MaxBytes: 2 * page}); err != nil {
				t.Fatal(err)
			}
			tensor := ggufindex.Tensor{Range: spans[0]}
			if resident {
				if err := r.ConfigureResidentTensors([]ggufindex.Tensor{tensor}, page); err != nil {
					t.Fatal(err)
				}
			}
			func() {
				defer func() {
					if recover() != "callback panic" {
						t.Error("callback panic lost")
					}
				}()
				callback := func(data []byte) error {
					if err := r.Close(); err != nil {
						t.Error(err)
					}
					if data[0] != 37 {
						t.Error("Close invalidated callback bytes")
					}
					panic("callback panic")
				}
				if resident {
					_ = r.WithTensor(tensor, callback)
				} else {
					_ = r.WithExpertRanges(spans, func(_ int, data []byte) error { return callback(data) })
				}
			}()
			if stats := r.ExpertCacheStats(); stats.ActiveLeases != 0 || stats.RetainedBytes != 0 || len(r.files) != 0 {
				t.Fatalf("panic/Close cleanup leaked: %+v", stats)
			}
		})
	}
}

func TestCacheParallelResidentAndExpertLeases(t *testing.T) {
	r, spans := cacheReader(t, 4)
	r.lockPages = func([]byte) error { return syscall.EPERM }
	page := uint64(os.Getpagesize())
	if err := r.ConfigureExpertCache(ExpertCacheOptions{MaxBytes: 2 * page, AgingInterval: 8}); err != nil {
		t.Fatal(err)
	}
	base := ggufindex.Tensor{Range: spans[0]}
	if err := r.ConfigureResidentTensors([]ggufindex.Tensor{base}, page); err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	for worker := 0; worker < 8; worker++ {
		wg.Add(1)
		go func(worker int) {
			defer wg.Done()
			for i := 0; i < 40; i++ {
				err := r.WithTensor(base, func(data []byte) error {
					if data[0] != 37 {
						return errors.New("incorrect parallel resident")
					}
					return r.WithExpertRanges([]ggufindex.Range{spans[1+(i+worker)%3]}, func(_ int, data []byte) error {
						stats := r.ExpertCacheStats()
						if data[0] != 37 || stats.RetainedBytes > 2*page {
							return errors.New("parallel cache exceeded budget or changed bytes")
						}
						delete(stats.RangeUses, spans[0])
						return nil
					})
				})
				if err != nil {
					t.Error(err)
					return
				}
			}
		}(worker)
	}
	wg.Wait()
	if stats := r.ExpertCacheStats(); stats.ResidentHits != 319 || stats.ActiveLeases != 0 || stats.RetainedBytes > 2*page {
		t.Fatalf("incorrect parallel accounting: %+v", stats)
	}
}

func TestResidentValidationAndOverflowBudget(t *testing.T) {
	r, spans := cacheReader(t, 2)
	page := uint64(os.Getpagesize())
	if err := r.ConfigureExpertCache(ExpertCacheOptions{MaxBytes: page}); err != nil {
		t.Fatal(err)
	}
	invalid := ggufindex.Tensor{Range: ggufindex.Range{File: spans[0].File, Start: 1, End: ^uint64(0)}}
	if err := r.ConfigureResidentTensors([]ggufindex.Tensor{{Range: spans[0]}, invalid}, ^uint64(0)); err == nil {
		t.Fatal("invalid resident range accepted")
	}
	if r.ExpertCacheStats().ResidentBudgetBytes != 0 {
		t.Fatal("invalid configuration reserved bytes")
	}
	if err := r.ConfigureResidentTensors([]ggufindex.Tensor{{Range: spans[0]}, {Range: spans[1]}}, ^uint64(0)); err != nil {
		t.Fatal(err)
	}
	if stats := r.ExpertCacheStats(); stats.ResidentBudgetBytes != page || stats.ResidentConfiguredRuns != 1 {
		t.Fatalf("resident cap overflowed total budget: %+v", stats)
	}
	readExpert(t, r, spans[1])
	if stats := r.ExpertCacheStats(); stats.CachedBytes != 0 || stats.AdmissionFailures != 1 {
		t.Fatalf("expert used reserved budget: %+v", stats)
	}
	if err := r.ConfigureResidentTensors(nil, 0); err == nil {
		t.Fatal("configuration after use accepted")
	}
}

func TestExpertBoundedLookaheadAndPrefetch(t *testing.T) {
	r, spans := cacheReader(t, 4)
	page := uint64(os.Getpagesize())
	if err := r.ConfigureExpertCache(ExpertCacheOptions{LookaheadBytes: page, PrefetchBytes: page}); err != nil {
		t.Fatal(err)
	}
	ranges := []ggufindex.Range{spans[0], {File: spans[0].File, Start: 2 * page, End: 4 * page}}
	if err := r.WithExpertRanges(ranges, func(index int, data []byte) error {
		stats := r.ExpertCacheStats()
		if stats.ActiveLeases != 1 || stats.MappedRuns != uint64(index+1) || data[0] != 37 {
			return fmt.Errorf("oversized lookahead was mapped: %+v", stats)
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	if stats := r.ExpertCacheStats(); stats.PrefetchCalls != 2 || stats.ActiveLeases != 0 {
		t.Fatalf("prefetch cap not applied: %+v", stats)
	}
}
