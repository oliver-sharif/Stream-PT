//go:build goexperiment.simd

package forward

import (
	"context"
	"fmt"
)

// DefaultPrefillBatchSize bounds activation storage independently of prompt length.
const DefaultPrefillBatchSize = 32

// Prefill processes consecutive prompt positions layer by layer, then projects
// only the last position. Existing cache positions before startPos are preserved.
// Like ForwardToken, it is not safe to call concurrently on the same engine.
// Cache and activation buffers may be partially updated on error or cancellation.
func (e *Engine) Prefill(ctx context.Context, tokens []int, startPos int) (int, error) {
	e.resetPrefix()
	return e.prefill(ctx, tokens, startPos, nil)
}

func (e *Engine) prefill(ctx context.Context, tokens []int, startPos int, sampler *tokenSampler) (int, error) {
	if len(tokens) == 0 {
		return 0, fmt.Errorf("empty prompt tokens")
	}
	stepError := func(index int, err error) (int, error) {
		return 0, fmt.Errorf("prefill step %d (token %d): %w", startPos+index, tokens[index], err)
	}
	if ctx == nil {
		return stepError(0, fmt.Errorf("nil context"))
	}
	if err := ctx.Err(); err != nil {
		return stepError(0, err)
	}
	if startPos < 0 || startPos >= e.KVCache.MaxPos {
		return stepError(0, fmt.Errorf("position %d outside KV cache capacity %d", startPos, e.KVCache.MaxPos))
	}
	if len(tokens) > e.KVCache.MaxPos-startPos {
		index := e.KVCache.MaxPos - startPos
		return stepError(index, fmt.Errorf("position %d outside KV cache capacity %d", e.KVCache.MaxPos, e.KVCache.MaxPos))
	}
	batchSize := e.Options.PrefillBatchSize
	if batchSize < 0 {
		return 0, fmt.Errorf("invalid prefill batch size %d", batchSize)
	}
	if batchSize == 0 {
		batchSize = DefaultPrefillBatchSize
	}
	batchSize = min(batchSize, len(tokens))
	if batchSize == 1 {
		var next int
		for index, token := range tokens {
			var err error
			next, err = e.forwardToken(ctx, token, startPos+index, index == len(tokens)-1, sampler)
			if err != nil {
				return stepError(index, err)
			}
		}
		return next, nil
	}
	if e.prefillScratch == nil || e.prefillScratch.batch < batchSize {
		var err error
		e.prefillScratch, err = newPrefillScratch(e.Config, batchSize)
		if err != nil {
			return 0, err
		}
	}
	scratch := e.prefillScratch
	options := Q40Options{Workers: e.Options.Workers, WindowBytes: e.Options.WindowBytes}
	dim := e.Config.HiddenDim
	for first := 0; first < len(tokens); first += batchSize {
		batch := min(batchSize, len(tokens)-first)
		x := scratch.x[:batch*dim]
		for item := 0; item < batch; item++ {
			if err := ReadEmbedding(ctx, e.Reader, e.TokenEmbd, tokens[first+item], x[item*dim:(item+1)*dim]); err != nil {
				return stepError(first+item, fmt.Errorf("embedding lookup: %w", err))
			}
		}
		for layer := range e.Layers {
			if err := forwardLayerBatch(ctx, e.Reader, layer, &e.Layers[layer], x, e.KVCache,
				startPos+first, e.Config, scratch, batch, options); err != nil {
				return 0, fmt.Errorf("prefill batch at position %d, layer %d: %w", startPos+first, layer, err)
			}
		}
		copy(e.X, x[(batch-1)*dim:batch*dim])
	}
	next, err := e.projectSampledOutput(ctx, options, sampler)
	if err != nil {
		return stepError(len(tokens)-1, err)
	}
	return next, nil
}
