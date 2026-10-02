//go:build goexperiment.simd && linux

package inference

import (
	"context"
	"errors"
	"fmt"
	"path/filepath"
	"strconv"
	"strings"

	"Stream-PT/forward"
	"Stream-PT/ggufindex"
	ggufmmap "Stream-PT/ggufmap"
)

const ContextLimit = 2048

type Request struct {
	Paths     []string
	Workers   int
	WindowMiB int
	MaxTokens int
	Prompt    string
}

type Event struct {
	Status  string
	Detail  string
	Token   string
	IsToken bool
}

type Runner func(context.Context, Request, func(Event)) error

func ParseRequest(paths, workers, window, tokens, prompt string) (Request, error) {
	r := Request{Prompt: prompt}
	seen := make(map[string]bool)
	for _, line := range strings.Split(paths, "\n") {
		if path := strings.TrimSpace(line); path != "" {
			path = filepath.Clean(path)
			if seen[path] {
				return r, fmt.Errorf("duplicate model shard: %s", path)
			}
			seen[path] = true
			r.Paths = append(r.Paths, path)
		}
	}
	for _, field := range []struct {
		name string
		text string
		out  *int
		max  int
	}{
		{"Compute workers", workers, &r.Workers, 4096},
		{"Window size (MiB)", window, &r.WindowMiB, 1048576},
		{"Maximum new tokens", tokens, &r.MaxTokens, ContextLimit},
	} {
		n, err := strconv.Atoi(strings.TrimSpace(field.text))
		if err != nil || n < 1 || n > field.max {
			return r, fmt.Errorf("%s must be a whole number between 1 and %d", field.name, field.max)
		}
		*field.out = n
	}
	return r, r.Validate()
}

func (r Request) Validate() error {
	if len(r.Paths) == 0 {
		return fmt.Errorf("select at least one GGUF model shard")
	}
	seen := make(map[string]bool)
	for _, path := range r.Paths {
		if strings.TrimSpace(path) == "" || seen[filepath.Clean(path)] {
			return fmt.Errorf("model shard paths must be non-empty and unique")
		}
		seen[filepath.Clean(path)] = true
	}
	if strings.TrimSpace(r.Prompt) == "" {
		return fmt.Errorf("enter a prompt before generating")
	}
	if r.Workers < 1 || r.Workers > 4096 || r.WindowMiB < 1 || r.WindowMiB > 1048576 {
		return fmt.Errorf("workers must be in 1–4096 and window size in 1–1048576 MiB")
	}
	if r.MaxTokens < 1 || r.MaxTokens > ContextLimit {
		return fmt.Errorf("maximum new tokens must be in 1–%d", ContextLimit)
	}
	return nil
}

func CheckContext(promptTokens, maxTokens int) error {
	if promptTokens < 1 || maxTokens < 1 || promptTokens > ContextLimit || maxTokens > ContextLimit-promptTokens+1 {
		return fmt.Errorf("prompt and generation exceed the %d-position cache; shorten the prompt or reduce the token limit", ContextLimit)
	}
	return nil
}

func Run(ctx context.Context, r Request, emit func(Event)) (err error) {
	if ctx == nil {
		return fmt.Errorf("nil context")
	}
	if err = r.Validate(); err != nil {
		return err
	}
	if err = ctx.Err(); err != nil {
		return err
	}
	if emit == nil {
		emit = func(Event) {}
	}
	emit(Event{Status: "Indexing model shards…"})
	model, err := ggufindex.Open(r.Paths)
	if err != nil {
		return fmt.Errorf("open model: %w", err)
	}
	if err = ctx.Err(); err != nil {
		return err
	}
	reader, err := ggufmmap.Open(model)
	if err != nil {
		return fmt.Errorf("open model reader: %w", err)
	}
	defer func() { err = errors.Join(err, reader.Close()) }()
	emit(Event{
		Status: "Loading tokenizer and initializing engine…",
		Detail: fmt.Sprintf("%s • %d layers • %d shards", filepath.Base(r.Paths[0]), model.LayerCount, len(model.Paths)),
	})
	engine, err := forward.NewEngineWithOptions(model, reader, forward.EngineOptions{
		Workers: r.Workers, WindowBytes: uint64(r.WindowMiB) << 20,
	})
	if err != nil {
		return fmt.Errorf("initialize engine: %w", err)
	}
	if err = ctx.Err(); err != nil {
		return err
	}
	promptTokens, err := engine.Tokenizer.EncodeChatPrompt(r.Prompt)
	if err != nil {
		return fmt.Errorf("encode prompt: %w", err)
	}
	if err = CheckContext(len(promptTokens), r.MaxTokens); err != nil {
		return err
	}
	emit(Event{Status: fmt.Sprintf("Prefilling %d prompt tokens • %d layers", len(promptTokens), model.LayerCount)})
	_, err = engine.Generate(ctx, promptTokens, r.MaxTokens, func(_ int, text string) bool {
		if ctx.Err() != nil {
			return false
		}
		emit(Event{Token: text, IsToken: true})
		return ctx.Err() == nil
	})
	return errors.Join(err, ctx.Err())
}
