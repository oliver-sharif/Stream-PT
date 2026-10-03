package tests

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"os"
	"reflect"
	"sync"
	"testing"

	"Stream-PT/ggufindex"
)

func TestExpertLookahead(t *testing.T) {
	for _, mode := range []string{"complete", "cancel", "panic"} {
		t.Run(mode, func(t *testing.T) {
			page := uint64(os.Getpagesize())
			r, paths, contents := expertReader(t, int(8*page))
			ranges := []ggufindex.Range{
				{File: paths[1], Start: page + 17, End: 2 * page},
				{File: paths[0], Start: 4 * page, End: 5 * page},
				{File: paths[0], Start: 17, End: page},
				{File: paths[1], Start: 6 * page, End: 7 * page},
			}
			var order []int
			func() {
				defer func() {
					got := recover()
					if mode == "panic" && got != context.Canceled || mode != "panic" && got != nil {
						t.Fatalf("unexpected panic: %v", got)
					}
				}()
				err := r.WithExpertRanges(ranges, func(index int, data []byte) error {
					span := ranges[index]
					if cap(data) != len(data) || !bytes.Equal(data, contents[span.Start:span.End]) {
						t.Fatal("incorrect data or capacity")
					}
					stats := r.ExpertCacheStats()
					if stats.MappedRuns != uint64(min(len(order)+2, len(ranges))) {
						t.Fatalf("expected exactly one run ahead: %+v", stats)
					}
					active := len(expertMappingLines(t, paths[0])) + len(expertMappingLines(t, paths[1]))
					if active != min(2, len(ranges)-len(order)) {
						t.Fatalf("unexpected active mapping count: %d", active)
					}
					order = append(order, index)
					if mode == "cancel" {
						return context.Canceled
					}
					if mode == "panic" {
						panic(context.Canceled)
					}
					return nil
				})
				if mode == "cancel" && !errors.Is(err, context.Canceled) || mode == "complete" && err != nil {
					t.Fatalf("unexpected callback error: %v", err)
				}
			}()
			want := []int{2, 1, 0, 3}
			if mode != "complete" {
				want = want[:1]
			}
			if !reflect.DeepEqual(order, want) {
				t.Fatalf("callback order %v, want %v", order, want)
			}
			expertAssertUnmapped(t, paths)
			if err := r.ConfigureExpertLookahead(false); err == nil {
				t.Fatal("configuration accepted after use")
			}
		})
	}
}

func TestExpertLookaheadConcurrent(t *testing.T) {
	r, paths, contents := expertReader(t, 16384)
	if err := r.ConfigureExpertLookahead(true); err != nil {
		t.Fatal(err)
	}
	ranges := []ggufindex.Range{
		{File: paths[0], Start: 17, End: 1024},
		{File: paths[1], Start: 8192, End: 12000},
	}
	var wg sync.WaitGroup
	for range 4 {
		wg.Go(func() {
			for range 10 {
				if err := r.WithExpertRanges(ranges, func(index int, data []byte) error {
					span := ranges[index]
					if !bytes.Equal(data, contents[span.Start:span.End]) {
						return fmt.Errorf("incorrect concurrent data")
					}
					return nil
				}); err != nil {
					t.Error(err)
				}
			}
		})
	}
	wg.Wait()
	if stats := r.ExpertCacheStats(); stats.MappedRuns != 80 || stats.SelectedRanges != 80 {
		t.Fatalf("incorrect concurrent statistics: %+v", stats)
	}
	expertAssertUnmapped(t, paths)
}
