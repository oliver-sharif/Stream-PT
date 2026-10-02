//go:build goexperiment.simd

package forward

import (
	"bytes"
	"encoding/json"
	"fmt"

	"Stream-PT/ggufindex"
)

// Config holds the hyperparameters for the GPT-OSS transformer model.
type Config struct {
	HiddenDim           int     `json:"hidden_dim"`
	NumLayers           int     `json:"num_layers"`
	NumHeads            int     `json:"num_heads"`
	NumKVHeads          int     `json:"num_kv_heads"`
	HeadDim             int     `json:"head_dim"`
	NumExperts          int     `json:"num_experts"`
	NumExpertsUsed      int     `json:"num_experts_used"`
	ExpertHiddenDim     int     `json:"expert_hidden_dim"`
	VocabSize           int     `json:"vocab_size"`
	RMSNormEps          float32 `json:"rms_norm_eps"`
	RopeFreqBase        float32 `json:"rope_freq_base"`
	RopeScalingFactor   float32 `json:"rope_scaling_factor"`
	RopeOriginalContext int     `json:"rope_original_context"`
	SlidingWindow       int     `json:"sliding_window"`
	ContextLength       int     `json:"context_length"`
	BOS                 int     `json:"bos_token_id"`
	EOS                 int     `json:"eos_token_id"`
	EOSTokens           []int   `json:"eos_tokens,omitempty"`
}

// IsEOS returns true if tokenID is a designated EOS/stop token ID.
func (c *Config) IsEOS(tokenID int) bool {
	if c == nil {
		return false
	}
	if tokenID == c.EOS {
		return true
	}
	for _, id := range c.EOSTokens {
		if tokenID == id {
			return true
		}
	}
	return false
}

// NewConfigFromModel extracts model hyperparameters from GGUF metadata.
func NewConfigFromModel(model *ggufindex.Model) (*Config, error) {
	if model == nil {
		return nil, fmt.Errorf("nil model")
	}

	cfg := &Config{
		HiddenDim:           2880,
		NumLayers:           model.LayerCount,
		NumHeads:            64,
		NumKVHeads:          8,
		HeadDim:             64,
		NumExperts:          128,
		NumExpertsUsed:      4,
		ExpertHiddenDim:     2880,
		VocabSize:           201088,
		RMSNormEps:          1e-5,
		RopeFreqBase:        150000.0,
		RopeScalingFactor:   32,
		RopeOriginalContext: 4096,
		SlidingWindow:       128,
		ContextLength:       131072,
		BOS:                 199998,
		EOS:                 200002,
	}

	if v, ok := model.Metadata["gpt-oss.embedding_length"]; ok {
		if val, err := readIntMetadata(v); err == nil && val > 0 {
			cfg.HiddenDim = val
		}
	}
	if v, ok := model.Metadata["gpt-oss.block_count"]; ok {
		if val, err := readIntMetadata(v); err == nil && val > 0 {
			cfg.NumLayers = val
		}
	}
	if v, ok := model.Metadata["gpt-oss.attention.head_count"]; ok {
		if val, err := readIntMetadata(v); err == nil && val > 0 {
			cfg.NumHeads = val
		}
	}
	if v, ok := model.Metadata["gpt-oss.attention.head_count_kv"]; ok {
		if val, err := readIntMetadata(v); err == nil && val > 0 {
			cfg.NumKVHeads = val
		}
	}
	if v, ok := model.Metadata["gpt-oss.attention.key_length"]; ok {
		if val, err := readIntMetadata(v); err == nil && val > 0 {
			cfg.HeadDim = val
		}
	}
	if v, ok := model.Metadata["gpt-oss.expert_count"]; ok {
		if val, err := readIntMetadata(v); err == nil && val > 0 {
			cfg.NumExperts = val
		}
	}
	if v, ok := model.Metadata["gpt-oss.expert_used_count"]; ok {
		if val, err := readIntMetadata(v); err == nil && val > 0 {
			cfg.NumExpertsUsed = val
		}
	}
	if v, ok := model.Metadata["gpt-oss.expert_feed_forward_length"]; ok {
		if val, err := readIntMetadata(v); err == nil && val > 0 {
			cfg.ExpertHiddenDim = val
		}
	}
	if v, ok := model.Metadata["gpt-oss.attention.layer_norm_rms_epsilon"]; ok {
		if val, err := readFloatMetadata(v); err == nil && val > 0 {
			cfg.RMSNormEps = val
		}
	}
	if v, ok := model.Metadata["gpt-oss.rope.freq_base"]; ok {
		if val, err := readFloatMetadata(v); err == nil && val > 0 {
			cfg.RopeFreqBase = val
		}
	}
	if v, ok := model.Metadata["gpt-oss.context_length"]; ok {
		if val, err := readIntMetadata(v); err == nil && val > 0 {
			cfg.ContextLength = val
		}
	}
	if v, ok := model.Metadata["gpt-oss.rope.scaling.factor"]; ok {
		val, err := readFloatMetadata(v)
		if err != nil || val < 1 {
			return nil, fmt.Errorf("invalid RoPE scaling factor")
		}
		cfg.RopeScalingFactor = val
	}
	if v, ok := model.Metadata["gpt-oss.rope.scaling.original_context_length"]; ok {
		val, err := readIntMetadata(v)
		if err != nil || val <= 0 {
			return nil, fmt.Errorf("invalid original RoPE context length")
		}
		cfg.RopeOriginalContext = val
	}
	if v, ok := model.Metadata["gpt-oss.attention.sliding_window"]; ok {
		val, err := readIntMetadata(v)
		if err != nil || val < 0 {
			return nil, fmt.Errorf("invalid attention sliding window")
		}
		cfg.SlidingWindow = val
	}
	if v, ok := model.Metadata["tokenizer.ggml.bos_token_id"]; ok {
		if val, err := readIntMetadata(v); err == nil {
			cfg.BOS = val
		}
	}
	if v, ok := model.Metadata["tokenizer.ggml.eos_token_id"]; ok {
		if ids, err := readIntArrayMetadata(v); err == nil && len(ids) > 0 {
			cfg.EOS = ids[0]
			for _, id := range ids {
				if !cfg.IsEOS(id) {
					cfg.EOSTokens = append(cfg.EOSTokens, id)
				}
			}
		} else if val, err := readIntMetadata(v); err == nil {
			cfg.EOS = val
			if !cfg.IsEOS(val) {
				cfg.EOSTokens = append(cfg.EOSTokens, val)
			}
		}
	}

	// Read vocab size from output.weight or token_embd.weight if present.
	if embd, ok := model.TensorByName("token_embd.weight"); ok && len(embd.Shape) == 2 {
		cfg.VocabSize = int(embd.Shape[1])
		cfg.HiddenDim = int(embd.Shape[0])
	}

	return cfg, nil
}

func readIntArrayMetadata(v ggufindex.MetadataValue) ([]int, error) {
	var buf bytes.Buffer
	if err := v.WriteJSON(&buf); err != nil {
		return nil, err
	}
	var arr []int
	if err := json.Unmarshal(buf.Bytes(), &arr); err == nil {
		return arr, nil
	}
	var num int
	if err := json.Unmarshal(buf.Bytes(), &num); err == nil {
		return []int{num}, nil
	}
	return nil, fmt.Errorf("metadata value is not an int or int array")
}

func readIntMetadata(v ggufindex.MetadataValue) (int, error) {
	var buf bytes.Buffer
	if err := v.WriteJSON(&buf); err != nil {
		return 0, err
	}
	var num int
	if err := json.Unmarshal(buf.Bytes(), &num); err != nil {
		return 0, err
	}
	return num, nil
}

func readFloatMetadata(v ggufindex.MetadataValue) (float32, error) {
	var buf bytes.Buffer
	if err := v.WriteJSON(&buf); err != nil {
		return 0, err
	}
	var val float64
	if err := json.Unmarshal(buf.Bytes(), &val); err != nil {
		return 0, err
	}
	return float32(val), nil
}
