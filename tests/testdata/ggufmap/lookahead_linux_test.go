package ggufmmap

import (
	"errors"
	"fmt"
	"os"
	"strings"
	"syscall"
	"testing"

	"Stream-PT/ggufindex"
)

func TestExpertLookaheadQueueDepthAndBytes(t *testing.T) {
	page := uint64(os.Getpagesize())
	for _, tc := range []struct {
		name     string
		depth    int
		bytes    uint64
		disabled bool
		want     int
	}{
		{name: "default", want: 1},
		{name: "one", depth: 1, want: 1},
		{name: "three", depth: 3, want: 3},
		{name: "maximum", depth: 8, want: 8},
		{name: "combined byte cap", depth: 3, bytes: 2 * page, want: 2},
		{name: "disabled", depth: 3, disabled: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			r, spans := cacheReader(t, 20)
			if err := r.ConfigureExpertCache(ExpertCacheOptions{LookaheadRuns: tc.depth, LookaheadBytes: tc.bytes}); err != nil {
				t.Fatal(err)
			}
			if err := r.ConfigureExpertLookahead(!tc.disabled); err != nil {
				t.Fatal(err)
			}
			ranges := make([]ggufindex.Range, 10)
			for i := range ranges {
				ranges[i] = spans[2*i]
			}
			calls := 0
			if err := r.WithExpertRanges(ranges, func(index int, data []byte) error {
				stats := r.ExpertCacheStats()
				wantMapped := min(len(ranges), index+1+tc.want)
				wantActive := min(len(ranges)-index, 1+tc.want)
				if index != calls || stats.MappedRuns != uint64(wantMapped) || stats.ActiveLeases != uint64(wantActive) || data[0] != 37 {
					return fmt.Errorf("incorrect pipeline at %d: %+v", index, stats)
				}
				calls++
				return nil
			}); err != nil {
				t.Fatal(err)
			}
			if stats := r.ExpertCacheStats(); calls != len(ranges) || stats.ActiveLeases != 0 {
				t.Fatalf("pipeline did not drain: calls=%d stats=%+v", calls, stats)
			}
		})
	}
}

func TestExpertLookaheadOversizedNextStopsQueue(t *testing.T) {
	r, spans := cacheReader(t, 9)
	page := uint64(os.Getpagesize())
	if err := r.ConfigureExpertCache(ExpertCacheOptions{LookaheadRuns: 3, LookaheadBytes: 2 * page}); err != nil {
		t.Fatal(err)
	}
	ranges := []ggufindex.Range{spans[0], {File: spans[0].File, Start: 2 * page, End: 5 * page}, spans[6], spans[8]}
	wantMapped := []uint64{1, 4, 4, 4}
	wantActive := []uint64{1, 3, 2, 1}
	if err := r.WithExpertRanges(ranges, func(index int, data []byte) error {
		stats := r.ExpertCacheStats()
		if stats.MappedRuns != wantMapped[index] || stats.ActiveLeases != wantActive[index] || len(data) != int(ranges[index].End-ranges[index].Start) {
			return fmt.Errorf("oversized run skipped or prefetched at %d: %+v", index, stats)
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
}

func TestExpertLookaheadPageRoundedCap(t *testing.T) {
	r, spans := cacheReader(t, 6)
	page := uint64(os.Getpagesize())
	if err := r.ConfigureExpertCache(ExpertCacheOptions{LookaheadRuns: 3, LookaheadBytes: 2 * page}); err != nil {
		t.Fatal(err)
	}
	ranges := []ggufindex.Range{spans[0], {File: spans[0].File, Start: 2*page + 1, End: 3*page + 1}, spans[5]}
	if err := r.WithExpertRanges(ranges, func(index int, _ []byte) error {
		stats := r.ExpertCacheStats()
		if index == 0 && (stats.MappedRuns != 2 || stats.MappedPageBytes != 3*page || stats.ActiveLeases != 2) {
			return fmt.Errorf("queue ignored page rounding: %+v", stats)
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
}

func TestExpertLookaheadHotLeasesSurviveEvictionAndClose(t *testing.T) {
	r, spans := cacheReader(t, 9)
	r.lockPages = func([]byte) error { return syscall.EPERM }
	page := uint64(os.Getpagesize())
	if err := r.ConfigureExpertCache(ExpertCacheOptions{MaxBytes: 4 * page, LookaheadRuns: 3, LookaheadBytes: 3 * page}); err != nil {
		t.Fatal(err)
	}
	ranges := []ggufindex.Range{spans[0], spans[2], spans[4], spans[6]}
	for _, span := range ranges {
		readExpert(t, r, span)
	}
	if err := r.WithExpertRanges(ranges, func(index int, data []byte) error {
		if index == 0 {
			for i := 0; i < 3; i++ {
				readExpert(t, r, spans[8])
			}
			stats := r.ExpertCacheStats()
			if stats.ActiveLeases != 4 || stats.Evictions != 0 || stats.AdmissionFailures != 3 || stats.CacheHits != 4 || stats.RetainedBytes != 4*page {
				return fmt.Errorf("queued hot leases were evicted: %+v", stats)
			}
			if err := r.Close(); err != nil {
				return err
			}
		}
		if data[0] != 37 || data[len(data)-1] != 37 {
			return errors.New("Close invalidated queued bytes")
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	assertLookaheadClosed(t, r, spans[0].File)
}

func TestExpertLookaheadErrorAndPanicTeardown(t *testing.T) {
	for _, retained := range []bool{false, true} {
		for _, closeInCallback := range []bool{false, true} {
			for _, panics := range []bool{false, true} {
				t.Run(fmt.Sprintf("retained=%t/close=%t/panic=%t", retained, closeInCallback, panics), func(t *testing.T) {
					r, spans := cacheReader(t, 7)
					page := uint64(os.Getpagesize())
					options := ExpertCacheOptions{LookaheadRuns: 3, LookaheadBytes: 3 * page}
					if retained {
						options.MaxBytes = 4 * page
					}
					if err := r.ConfigureExpertCache(options); err != nil {
						t.Fatal(err)
					}
					sentinel := errors.New("pipeline callback failed")
					calls := 0
					func() {
						defer func() {
							got := recover()
							if panics && got != sentinel || !panics && got != nil {
								t.Errorf("unexpected panic: %v", got)
							}
						}()
						err := r.WithExpertRanges([]ggufindex.Range{spans[0], spans[2], spans[4], spans[6]}, func(_ int, data []byte) error {
							calls++
							if stats := r.ExpertCacheStats(); stats.ActiveLeases != 4 {
								t.Errorf("pipeline not filled before callback: %+v", stats)
							}
							if closeInCallback {
								if err := r.Close(); err != nil {
									t.Error(err)
								}
							}
							if data[0] != 37 {
								t.Error("callback bytes invalidated")
							}
							if panics {
								panic(sentinel)
							}
							return sentinel
						})
						if !panics && !errors.Is(err, sentinel) {
							t.Errorf("callback error lost: %v", err)
						}
					}()
					if stats := r.ExpertCacheStats(); stats.ActiveLeases != 0 || calls != 1 || r.activeCalls != 0 {
						t.Fatalf("pipeline teardown leaked: calls=%d stats=%+v", calls, stats)
					}
					for _, entry := range r.expertCache {
						if entry.refs != 0 {
							t.Error("queued cache reference leaked")
						}
					}
					if err := r.Close(); err != nil {
						t.Fatal(err)
					}
					assertLookaheadClosed(t, r, spans[0].File)
				})
			}
		}
	}
}

func assertLookaheadClosed(t *testing.T, r *Reader, path string) {
	t.Helper()
	if stats := r.ExpertCacheStats(); stats.ActiveLeases != 0 || stats.RetainedBytes != 0 || len(r.files) != 0 {
		t.Fatalf("Close cleanup leaked: %+v", stats)
	}
	maps, err := os.ReadFile("/proc/self/maps")
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(maps), path) {
		t.Fatal("pipeline mapping leaked after Close")
	}
}

func TestExpertLookaheadRunsValidation(t *testing.T) {
	r, _ := cacheReader(t, 1)
	for _, depth := range []int{-1, 9, int(^uint(0) >> 1)} {
		if err := r.ConfigureExpertCache(ExpertCacheOptions{LookaheadRuns: depth, MaxBytes: 1}); err == nil {
			t.Fatalf("invalid depth %d accepted", depth)
		}
		if r.expertOptions.MaxBytes != 0 || r.ExpertCacheStats().BudgetBytes != 0 {
			t.Fatal("invalid depth partially configured cache")
		}
	}
	if err := r.ConfigureExpertCache(ExpertCacheOptions{LookaheadRuns: 3}); err != nil {
		t.Fatal(err)
	}
}
