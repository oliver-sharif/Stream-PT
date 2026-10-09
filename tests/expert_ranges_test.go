package tests

import (
	"bytes"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"sync"
	"testing"

	"Stream-PT/ggufindex"
	. "Stream-PT/ggufmap"
)

func expertReader(t *testing.T, size int) (*Reader, []string, []byte) {
	t.Helper()
	contents := make([]byte, size)
	for i := range contents {
		contents[i] = byte(i % 251)
	}
	dir := t.TempDir()
	paths := []string{filepath.Join(dir, "a.bin"), filepath.Join(dir, "b.bin")}
	for _, path := range paths {
		if err := os.WriteFile(path, contents, 0600); err != nil {
			t.Fatal(err)
		}
	}
	r, err := Open(&ggufindex.Model{Paths: []string{paths[1], paths[0]}})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := r.Close(); err != nil {
			t.Error(err)
		}
	})
	return r, paths, contents
}

func expertMappingLines(t *testing.T, path string) []string {
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

func expertAssertUnmapped(t *testing.T, paths []string) {
	t.Helper()
	for _, path := range paths {
		if lines := expertMappingLines(t, path); len(lines) != 0 {
			t.Fatalf("remaining mappings for %s: %v", path, lines)
		}
	}
}

func TestWithExpertRangesPhysicalOrder(t *testing.T) {
	r, paths, contents := expertReader(t, 3*os.Getpagesize()+137)
	// Keep the single-run path covered separately from TestExpertLookahead.
	if err := r.ConfigureExpertLookahead(false); err != nil {
		t.Fatal(err)
	}
	if err := r.ConfigureExpertCache(ExpertCacheOptions{TrackStats: true}); err != nil {
		t.Fatal(err)
	}
	ranges := []ggufindex.Range{
		{File: paths[1], Start: 37, End: 91},
		{File: paths[0], Start: 32, End: 64},
		{File: paths[0], Start: 128, End: 150},
		{File: paths[0], Start: 17, End: 32},
		{File: paths[1], Start: 91, End: 109},
	}
	wantRuns := []uint64{1, 1, 2, 3, 3}
	var order []int
	var firstPointer uintptr
	err := r.WithExpertRanges(ranges, func(index int, data []byte) error {
		span := ranges[index]
		if cap(data) != len(data) || !bytes.Equal(data, contents[span.Start:span.End]) {
			t.Fatalf("incorrect data or capacity at index %d", index)
		}
		stats := r.ExpertCacheStats()
		if stats.MappedRuns != wantRuns[len(order)] {
			t.Fatalf("mapped runs: %d, want %d", stats.MappedRuns, wantRuns[len(order)])
		}
		lines := expertMappingLines(t, span.File)
		if len(lines) != 1 {
			t.Fatalf("want one active mapping: %v", lines)
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
		pointer := reflect.ValueOf(data).Pointer()
		if uint64(pointer) < start || uint64(pointer)+uint64(len(data)) > end || fields[1] != "r--p" {
			t.Fatalf("not a read-only zero-copy mapping: %s", lines[0])
		}
		if index == 3 {
			firstPointer = pointer
		}
		if index == 1 && pointer != firstPointer+15 {
			t.Fatal("adjacent ranges did not share the mapping")
		}
		order = append(order, index)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(order, []int{3, 1, 2, 0, 4}) {
		t.Fatalf("callback order: %v", order)
	}
	stats := r.ExpertCacheStats()
	if stats.SelectedRanges != 5 || stats.SelectedBytes != 141 || stats.MappedBytes != 141 || stats.MappedRuns != 3 {
		t.Fatalf("incorrect selection/mapping counters: %+v", stats)
	}
	if stats.MappedPageBytes != uint64(3*os.Getpagesize()) || stats.CachedBytes != 0 {
		t.Fatalf("incorrect page/cache counters: %+v", stats)
	}
	for _, span := range ranges {
		if stats.RangeUses[span] != 1 {
			t.Fatalf("frequency for %+v: %d", span, stats.RangeUses[span])
		}
	}
	stats.RangeUses[ranges[0]] = 99
	if r.ExpertCacheStats().RangeUses[ranges[0]] != 1 {
		t.Fatal("snapshot shares frequency storage")
	}
	expertAssertUnmapped(t, paths)
}

func TestWithExpertRangesEmpty(t *testing.T) {
	r, paths, _ := expertReader(t, 32)
	for _, ranges := range [][]ggufindex.Range{nil, {}} {
		if err := r.WithExpertRanges(ranges, nil); err != nil {
			t.Fatal(err)
		}
		if err := r.WithExpertRanges(ranges, func(int, []byte) error {
			t.Fatal("empty selection invoked callback")
			return nil
		}); err != nil {
			t.Fatal(err)
		}
	}
	if !reflect.DeepEqual(r.ExpertCacheStats(), ExpertCacheStats{}) {
		t.Fatalf("empty selection changed statistics: %+v", r.ExpertCacheStats())
	}
	if err := r.ConfigureExpertCache(ExpertCacheOptions{TrackStats: true}); err != nil {
		t.Fatalf("empty selection prevented configuration: %v", err)
	}
	expertAssertUnmapped(t, paths)
}

func TestWithExpertRangesValidationBeforeCallback(t *testing.T) {
	r, paths, _ := expertReader(t, 256)
	valid := ggufindex.Range{File: paths[0], Start: 17, End: 32}
	cases := map[string][]ggufindex.Range{
		"unknown file": {valid, {File: "missing", Start: 1, End: 2}},
		"empty":        {valid, {File: paths[0], Start: 128, End: 128}},
		"reversed":     {valid, {File: paths[0], Start: 128, End: 127}},
		"past EOF":     {valid, {File: paths[0], Start: 128, End: 257}},
		"overflow":     {valid, {File: paths[0], Start: 128, End: ^uint64(0)}},
		"duplicate":    {valid, valid},
		"overlap":      {valid, {File: paths[0], Start: 31, End: 50}},
		"contained":    {valid, {File: paths[0], Start: 18, End: 22}},
	}
	for name, ranges := range cases {
		t.Run(name, func(t *testing.T) {
			err := r.WithExpertRanges(ranges, func(int, []byte) error {
				t.Fatal("invalid selection invoked callback")
				return nil
			})
			if err == nil {
				t.Fatal("invalid selection accepted")
			}
			if !reflect.DeepEqual(r.ExpertCacheStats(), ExpertCacheStats{}) {
				t.Fatal("invalid selection changed counters")
			}
			expertAssertUnmapped(t, paths)
		})
	}
	if err := r.WithExpertRanges([]ggufindex.Range{valid}, nil); err == nil {
		t.Fatal("nil callback accepted for nonempty selection")
	}
}

func TestWithExpertRangesCallbackCleanup(t *testing.T) {
	r, paths, _ := expertReader(t, 256)
	if err := r.ConfigureExpertLookahead(false); err != nil {
		t.Fatal(err)
	}
	ranges := []ggufindex.Range{
		{File: paths[0], Start: 17, End: 32},
		{File: paths[0], Start: 32, End: 64},
		{File: paths[1], Start: 128, End: 150},
	}
	sentinel := errors.New("callback error")
	calls := 0
	err := r.WithExpertRanges(ranges, func(int, []byte) error {
		calls++
		return sentinel
	})
	if !errors.Is(err, sentinel) || calls != 1 {
		t.Fatalf("callback error: %v, calls: %d", err, calls)
	}
	expertAssertUnmapped(t, paths)
	stats := r.ExpertCacheStats()
	if stats.SelectedRanges != 3 || stats.SelectedBytes != 69 || stats.MappedRuns != 1 || stats.MappedBytes != 47 || stats.RangeUses != nil {
		t.Fatalf("incorrect counters after callback error: %+v", stats)
	}
	func() {
		defer func() {
			if got := recover(); got != sentinel {
				t.Fatalf("panic not preserved: %v", got)
			}
		}()
		_ = r.WithExpertRanges(ranges, func(int, []byte) error { panic(sentinel) })
	}()
	expertAssertUnmapped(t, paths)
	if err := r.ConfigureExpertCache(ExpertCacheOptions{}); err == nil {
		t.Fatal("configuration accepted after use")
	}
	if err := r.Close(); err != nil {
		t.Fatal(err)
	}
	if err := r.WithExpertRanges(ranges, func(int, []byte) error { return nil }); err == nil {
		t.Fatal("closed reader accepted nonempty selection")
	}
	if err := r.ConfigureExpertCache(ExpertCacheOptions{}); err == nil {
		t.Fatal("closed reader accepted configuration")
	}
}

func TestWithExpertRangesCoalesceLimit(t *testing.T) {
	limit := DefaultExpertCoalesceBytes
	r, paths, contents := expertReader(t, int(limit+4096))
	for _, tc := range []struct {
		name string
		ends []uint64
	}{
		{name: "bounded adjacent", ends: []uint64{limit / 2, limit, limit + 1}},
		{name: "oversized single", ends: []uint64{limit + 1, limit + 2}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			before := r.ExpertCacheStats()
			var ranges []ggufindex.Range
			var start uint64
			for _, end := range tc.ends {
				ranges = append(ranges, ggufindex.Range{File: paths[0], Start: start, End: end})
				start = end
			}
			err := r.WithExpertRanges(ranges, func(index int, data []byte) error {
				span := ranges[index]
				if cap(data) != len(data) || !bytes.Equal(data, contents[span.Start:span.End]) {
					t.Fatal("incorrect bounded data")
				}
				return nil
			})
			if err != nil {
				t.Fatal(err)
			}
			after := r.ExpertCacheStats()
			if after.MappedRuns-before.MappedRuns != 2 || after.MappedBytes-before.MappedBytes != start {
				t.Fatalf("incorrect bounded mapping counts: before %+v, after %+v", before, after)
			}
			expertAssertUnmapped(t, paths)
		})
	}
}

func TestWithExpertRangesCache(t *testing.T) {
	page := uint64(os.Getpagesize())
	r, paths, contents := expertReader(t, int(4*page))
	options := ExpertCacheOptions{MaxBytes: page, MinUses: 2, TrackStats: true}
	if err := r.ConfigureExpertCache(options); err != nil {
		t.Fatal(err)
	}
	ranges := []ggufindex.Range{
		{File: paths[0], Start: 17, End: 32},
		{File: paths[0], Start: 32, End: 64},
	}
	read := func(ranges []ggufindex.Range) {
		t.Helper()
		if err := r.WithExpertRanges(ranges, func(index int, data []byte) error {
			span := ranges[index]
			if cap(data) != len(data) || !bytes.Equal(data, contents[span.Start:span.End]) {
				t.Fatal("incorrect cache contents")
			}
			_ = r.ExpertCacheStats()
			return nil
		}); err != nil {
			t.Fatal(err)
		}
	}
	read(ranges)
	if stats := r.ExpertCacheStats(); stats.CachedBytes != 0 || stats.LockFailures != 0 || stats.MappedRuns != 1 {
		t.Fatalf("cache admitted below threshold: %+v", stats)
	}
	expertAssertUnmapped(t, paths)
	read(ranges)
	second := r.ExpertCacheStats()
	if second.CachedBytes != page || second.CachedRuns != 1 || second.LockFailures > 1 || second.RetainedBytes != page {
		t.Fatalf("incorrect cache admission: %+v", second)
	}
	wantPinned := page
	if second.LockFailures == 1 {
		wantPinned = 0
	}
	if second.PinnedBytes != wantPinned {
		t.Fatalf("incorrect best-effort pinning: %+v", second)
	}
	read(ranges)
	third := r.ExpertCacheStats()
	if third.CacheHits != 1 || third.MappedRuns != 2 || third.CachedBytes != page || third.LockFailures != second.LockFailures {
		t.Fatalf("retained run not reused, even without pinning: %+v", third)
	}
	if third.SelectedRanges != 6 || third.SelectedBytes != 141 || third.RangeUses[ranges[0]] != 3 || third.RangeUses[ranges[1]] != 3 {
		t.Fatalf("incorrect hot frequencies: %+v", third)
	}
	tooLarge := []ggufindex.Range{{File: paths[0], Start: page - 1, End: page + 1}}
	read(tooLarge)
	read(tooLarge)
	if stats := r.ExpertCacheStats(); stats.CachedBytes != third.CachedBytes || stats.CachedRuns != third.CachedRuns || stats.LockFailures != third.LockFailures {
		t.Fatalf("page-rounded budget exceeded: %+v", stats)
	}
	other := []ggufindex.Range{{File: paths[1], Start: 17, End: 64}}
	read(other)
	read(other)
	afterOther := r.ExpertCacheStats()
	if afterOther.CachedBytes > options.MaxBytes {
		t.Fatalf("cache budget exceeded: %+v", afterOther)
	}
	if afterOther.CachedRuns != 1 || afterOther.CachedBytes != page || afterOther.LockFailures != third.LockFailures {
		t.Fatalf("colder run displaced hotter resident: %+v", afterOther)
	}
	beforeHit := r.ExpertCacheStats()
	read(ranges)
	if afterHit := r.ExpertCacheStats(); afterHit.CacheHits != beforeHit.CacheHits+1 || afterHit.MappedRuns != beforeHit.MappedRuns {
		t.Fatalf("hotter run was evicted: %+v", afterHit)
	}
	sentinel := errors.New("cached callback error")
	if err := r.WithExpertRanges(ranges, func(int, []byte) error { return sentinel }); !errors.Is(err, sentinel) {
		t.Fatalf("cache suppressed callback error: %v", err)
	}
	beforeClose := r.ExpertCacheStats()
	if err := r.Close(); err != nil {
		t.Fatal(err)
	}
	if stats := r.ExpertCacheStats(); stats.CachedBytes != 0 || stats.CachedRuns != 0 || stats.RangeUses[ranges[0]] != beforeClose.RangeUses[ranges[0]] {
		t.Fatalf("incorrect post-close snapshot: %+v", stats)
	}
	expertAssertUnmapped(t, paths)
}

func TestWithExpertRangesConcurrent(t *testing.T) {
	r, paths, contents := expertReader(t, os.Getpagesize())
	if err := r.ConfigureExpertCache(ExpertCacheOptions{TrackStats: true}); err != nil {
		t.Fatal(err)
	}
	ranges := []ggufindex.Range{{File: paths[0], Start: 17, End: 32}, {File: paths[0], Start: 32, End: 64}}
	const workers = 8
	const iterations = 20
	var wg sync.WaitGroup
	for worker := 0; worker < workers; worker++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < iterations; i++ {
				err := r.WithExpertRanges(ranges, func(index int, data []byte) error {
					span := ranges[index]
					if !bytes.Equal(data, contents[span.Start:span.End]) {
						return errors.New("incorrect concurrent data")
					}
					stats := r.ExpertCacheStats()
					delete(stats.RangeUses, span)
					return nil
				})
				if err != nil {
					t.Error(err)
					return
				}
			}
		}()
	}
	wg.Wait()
	stats := r.ExpertCacheStats()
	if stats.MappedRuns != workers*iterations || stats.MappedBytes != workers*iterations*47 || stats.SelectedRanges != workers*iterations*2 {
		t.Fatalf("incorrect concurrent counters: %+v", stats)
	}
	for _, span := range ranges {
		if stats.RangeUses[span] != workers*iterations {
			t.Fatalf("lost frequency increments: %+v", stats)
		}
	}
	expertAssertUnmapped(t, paths)
}
