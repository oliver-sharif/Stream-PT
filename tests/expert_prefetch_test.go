package tests

import (
	"bytes"
	"os"
	"testing"

	"Stream-PT/ggufindex"
	ggufmmap "Stream-PT/ggufmap"
)

func TestExpertPrefetchCoversSelectedMapping(t *testing.T) {
	step := max(ggufmmap.DefaultExpertPrefetchBytes, os.Getpagesize())
	r, paths, contents := expertReader(t, 6*step)
	ranges := []ggufindex.Range{
		{File: paths[0], Start: 17, End: uint64(2*step + 17)},
		{File: paths[0], Start: uint64(4 * step), End: uint64(5 * step)},
	}
	if err := r.WithExpertRanges(ranges, func(index int, data []byte) error {
		span := ranges[index]
		if !bytes.Equal(data, contents[span.Start:span.End]) {
			t.Fatal("incorrect selected data")
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	stats := r.ExpertCacheStats()
	if stats.PrefetchCalls != 4 || stats.PrefetchFailures != 0 || stats.MappedRuns != 2 || stats.MappedBytes != uint64(3*step) {
		t.Fatalf("prefetch must cover both selected mappings, not the gap: %+v", stats)
	}
	expertAssertUnmapped(t, paths)
}
