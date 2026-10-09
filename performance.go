//go:build goexperiment.simd

package main

import (
	"os"
	"strconv"
	"strings"
	"syscall"

	"Stream-PT/forward"
	"Stream-PT/ggufindex"
	ggufmmap "Stream-PT/ggufmap"
)

type InferencePerformance struct {
	forward.GenerationMetrics
	FirstVisibleSeconds float64          `json:"first_visible_seconds"`
	ReadBytes           uint64           `json:"read_bytes"`
	MajorFaults         int64            `json:"major_faults"`
	IOAvailable         bool             `json:"io_available"`
	Cache               CacheDiagnostics `json:"cache"`
	CacheHits           uint64           `json:"expert_hits_delta"`
	ResidentHits        uint64           `json:"resident_hits_delta"`
}

type CacheDiagnostics struct {
	BudgetBytes   uint64 `json:"budget_bytes"`
	RetainedBytes uint64 `json:"retained_bytes"`
	PinnedBytes   uint64 `json:"pinned_bytes"`
	ResidentBytes uint64 `json:"resident_bytes"`
	ExpertHits    uint64 `json:"expert_hits"`
	ResidentHits  uint64 `json:"resident_hits"`
	Evictions     uint64 `json:"evictions"`
	LockFailures  uint64 `json:"lock_failures"`
}

func cacheDiagnostics(stats ggufmmap.ExpertCacheStats) CacheDiagnostics {
	return CacheDiagnostics{BudgetBytes: stats.BudgetBytes, RetainedBytes: stats.RetainedBytes, PinnedBytes: stats.PinnedBytes, ResidentBytes: stats.ResidentBytes, ExpertHits: stats.CacheHits, ResidentHits: stats.ResidentHits, Evictions: stats.Evictions, LockFailures: stats.LockFailures}
}

func residentWeightCandidates(model *ggufindex.Model) []ggufindex.Tensor {
	var tensors []ggufindex.Tensor
	for _, name := range []string{"output.weight", "output_norm.weight"} {
		if tensor, ok := model.TensorByName(name); ok {
			tensors = append(tensors, tensor)
		}
	}
	for _, layer := range model.Layers {
		for _, tensor := range layer.Tensors {
			if !strings.Contains(tensor.Name, "_exps") {
				tensors = append(tensors, tensor)
			}
		}
	}
	return tensors
}

type processIO struct {
	readBytes   uint64
	majorFaults int64
	available   bool
}

func readProcessIO() processIO {
	data, err := os.ReadFile("/proc/self/io")
	if err != nil {
		return processIO{}
	}
	var usage syscall.Rusage
	if syscall.Getrusage(syscall.RUSAGE_SELF, &usage) != nil {
		return processIO{}
	}
	for line := range strings.SplitSeq(string(data), "\n") {
		if value, ok := strings.CutPrefix(line, "read_bytes:"); ok {
			bytes, err := strconv.ParseUint(strings.TrimSpace(value), 10, 64)
			return processIO{readBytes: bytes, majorFaults: usage.Majflt, available: err == nil}
		}
	}
	return processIO{}
}

func inferencePerformance(metrics forward.GenerationMetrics, firstVisible float64, before, after processIO) InferencePerformance {
	p := InferencePerformance{GenerationMetrics: metrics, FirstVisibleSeconds: firstVisible}
	if before.available && after.available && after.readBytes >= before.readBytes && after.majorFaults >= before.majorFaults {
		p.IOAvailable = true
		p.ReadBytes = after.readBytes - before.readBytes
		p.MajorFaults = after.majorFaults - before.majorFaults
	}
	return p
}

func cacheMiBForMemory(requested, available uint64) uint64 {
	return min(requested, available/(4<<20))
}

// Reserve at most a quarter of currently available memory by default, also
// respecting cgroup v2 headroom. Explicit budgets bypass this conservative cap.
func automaticCacheMiB(requested uint64) uint64 {
	data, err := os.ReadFile("/proc/meminfo")
	if err != nil {
		return 0
	}
	var available uint64
	for line := range strings.SplitSeq(string(data), "\n") {
		fields := strings.Fields(line)
		if len(fields) >= 2 && fields[0] == "MemAvailable:" {
			kb, err := strconv.ParseUint(fields[1], 10, 64)
			if err != nil || kb > ^uint64(0)/1024 {
				return 0
			}
			available = kb * 1024
			break
		}
	}
	limitData, limitErr := os.ReadFile("/sys/fs/cgroup/memory.max")
	usedData, usedErr := os.ReadFile("/sys/fs/cgroup/memory.current")
	if limitErr == nil && usedErr == nil {
		limit, err1 := strconv.ParseUint(strings.TrimSpace(string(limitData)), 10, 64)
		used, err2 := strconv.ParseUint(strings.TrimSpace(string(usedData)), 10, 64)
		if err1 == nil && err2 == nil {
			if used >= limit {
				available = 0
			} else {
				available = min(available, limit-used)
			}
		}
	}
	return cacheMiBForMemory(requested, available)
}
