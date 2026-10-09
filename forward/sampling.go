//go:build goexperiment.simd

package forward

import (
	"fmt"
	"math"
	"math/rand/v2"
	"sort"
)

// SamplingOptions follows llama.cpp's penalties -> top-k -> top-p -> min-p ->
// temperature -> distribution chain. Seed nil selects a fresh random seed.
type SamplingOptions struct {
	Temperature      float64 `json:"temperature"`
	TopK             int     `json:"top_k"`
	TopP             float64 `json:"top_p"`
	MinP             float64 `json:"min_p"`
	RepeatPenalty    float64 `json:"repeat_penalty"`
	RepeatLastN      int     `json:"repeat_last_n"`
	FrequencyPenalty float64 `json:"frequency_penalty"`
	PresencePenalty  float64 `json:"presence_penalty"`
	Seed             *uint64 `json:"seed,omitempty"`
}

// DefaultSamplingOptions uses llama.cpp's active sampling defaults.
func DefaultSamplingOptions() SamplingOptions {
	return SamplingOptions{Temperature: 0.8, TopK: 40, TopP: 0.95, MinP: 0.05,
		RepeatPenalty: 1, RepeatLastN: 64}
}

func (o SamplingOptions) Validate() error {
	for _, value := range []float64{o.Temperature, o.TopP, o.MinP, o.RepeatPenalty, o.FrequencyPenalty, o.PresencePenalty} {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return fmt.Errorf("sampling parameters must be finite")
		}
	}
	if o.Temperature < 0 || o.TopK < 0 || o.TopP <= 0 || o.TopP > 1 || o.MinP < 0 || o.MinP > 1 || o.RepeatPenalty <= 0 || o.RepeatLastN < -1 {
		return fmt.Errorf("invalid sampling parameters: temperature >= 0, top_k >= 0, 0 < top_p <= 1, 0 <= min_p <= 1, repeat_penalty > 0, repeat_last_n >= -1 required")
	}
	return nil
}

type tokenCandidate struct {
	id    int
	logit float64
	p     float64
}

// tokenSampler is local to one generation, including prompt and generated history.
type tokenSampler struct {
	options    SamplingOptions
	rng        *rand.Rand
	history    []int
	counts     map[int]int
	logits     []float32
	candidates []tokenCandidate
}

func newTokenSampler(options SamplingOptions, prompt []int) (*tokenSampler, error) {
	if err := options.Validate(); err != nil {
		return nil, err
	}
	seed := rand.Uint64()
	if options.Seed != nil {
		seed = *options.Seed
	}
	s := &tokenSampler{options: options, rng: rand.New(rand.NewPCG(seed, seed^0x9e3779b97f4a7c15)), counts: make(map[int]int)}
	if options.RepeatLastN > 0 && len(prompt) > options.RepeatLastN {
		prompt = prompt[len(prompt)-options.RepeatLastN:]
	}
	for _, id := range prompt {
		s.accept(id)
	}
	return s, nil
}

func (s *tokenSampler) accept(id int) {
	n := s.options.RepeatLastN
	if n == 0 {
		return
	}
	if n > 0 && len(s.history) == n {
		old := s.history[0]
		s.counts[old]--
		if s.counts[old] == 0 {
			delete(s.counts, old)
		}
		copy(s.history, s.history[1:])
		s.history = s.history[:n-1]
	}
	s.history = append(s.history, id)
	s.counts[id]++
}

func (s *tokenSampler) penalize(id int, logit float64) float64 {
	if count := s.counts[id]; count > 0 {
		if logit <= 0 {
			logit *= s.options.RepeatPenalty
		} else {
			logit /= s.options.RepeatPenalty
		}
		logit -= float64(count)*s.options.FrequencyPenalty + s.options.PresencePenalty
	}
	return logit
}

func (s *tokenSampler) sample(logits []float32) (int, error) {
	s.candidates = s.candidates[:0]
	for id, logit := range logits {
		value := s.penalize(id, float64(logit))
		if math.IsNaN(value) || math.IsInf(value, 1) {
			return 0, fmt.Errorf("invalid logit for token %d", id)
		}
		if !math.IsInf(value, -1) {
			s.candidates = append(s.candidates, tokenCandidate{id: id, logit: value})
		}
	}
	if len(s.candidates) == 0 {
		return 0, fmt.Errorf("no finite token candidates")
	}
	c := s.candidates
	sort.Slice(c, func(i, j int) bool {
		if c[i].logit == c[j].logit {
			return c[i].id < c[j].id
		}
		return c[i].logit > c[j].logit
	})
	// At zero temperature llama.cpp ultimately selects the highest remaining
	// logit; the probability filters always retain that candidate.
	if s.options.Temperature == 0 {
		return c[0].id, nil
	}
	if k := s.options.TopK; k > 0 && k < len(c) {
		c = c[:k]
	}
	if s.options.TopP < 1 {
		softmaxCandidates(c, 1)
		cumulative := 0.0
		for i := range c {
			cumulative += c[i].p
			if cumulative >= s.options.TopP {
				c = c[:i+1]
				break
			}
		}
	}
	if s.options.MinP > 0 {
		threshold := c[0].logit + math.Log(s.options.MinP)
		for i := 1; i < len(c); i++ {
			if c[i].logit < threshold {
				c = c[:i]
				break
			}
		}
	}
	softmaxCandidates(c, s.options.Temperature)
	draw := s.rng.Float64()
	for _, candidate := range c {
		draw -= candidate.p
		if draw < 0 {
			return candidate.id, nil
		}
	}
	return c[len(c)-1].id, nil
}

func softmaxCandidates(c []tokenCandidate, temperature float64) {
	total := 0.0
	for i := range c {
		c[i].p = math.Exp((c[i].logit - c[0].logit) / temperature)
		total += c[i].p
	}
	for i := range c {
		c[i].p /= total
	}
}
