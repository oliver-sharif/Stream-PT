//go:build goexperiment.simd

package tests

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"reflect"
	"testing"

	. "Stream-PT/forward"
)

func TestQuantizedSIMDReference(t *testing.T) {
	const input, output = 2880, 7
	x := make([]float32, input)
	for i := range x {
		x[i] = float32(i%29-14) / 64
	}
	for _, typ := range []uint32{8, 39} {
		t.Run(fmt.Sprintf("type%d", typ), func(t *testing.T) {
			blockBytes := 34
			if typ == 39 {
				blockBytes = 17
			}
			rowBytes := input / 32 * blockBytes
			data := make([]byte, output*rowBytes)
			want := make([]float32, output)
			fp4 := [...]float32{0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6}
			for row := range output {
				var sum float64
				for block := range input / 32 {
					encoded := data[row*rowBytes+block*blockBytes : row*rowBytes+(block+1)*blockBytes]
					var scale float32
					if typ == 8 {
						bits := [...]uint16{0x3c00, 0xb800, 0x3400, 0x0001, 0x0400}[block%5]
						binary.LittleEndian.PutUint16(encoded, bits)
						scale = referenceFloat16(bits)
						for i := range 32 {
							encoded[2+i] = byte(row*31 + block*13 + i*7)
							sum += float64(scale*float32(int8(encoded[2+i]))) * float64(x[block*32+i])
						}
					} else {
						encoded[0] = byte(120 + block%15)
						scale = float32(math.Ldexp(1, int(encoded[0])-127))
						for i := range 16 {
							encoded[1+i] = byte(row*31 + block*13 + i*7)
							sum += float64(scale*fp4[encoded[1+i]&15]) * float64(x[block*32+i])
							sum += float64(scale*fp4[encoded[1+i]>>4]) * float64(x[block*32+i+16])
						}
					}
				}
				want[row] = float32(sum)
			}
			shape := []uint64{input, output}
			if typ == 39 {
				shape = append(shape, 1)
			}
			reader, tensor := tensorFixture(t, typ, shape, data)
			for _, workers := range []int{1, 3} {
				y := make([]float32, output)
				options := Q40Options{Workers: workers, WindowBytes: uint64(3 * rowBytes)}
				var err error
				if typ == 8 {
					err = MulQ80Into(context.Background(), reader, tensor, x, y, options)
				} else {
					err = MulMXFP4Expert(context.Background(), reader, tensor, 0, x, nil, y, options)
				}
				if err != nil {
					t.Fatal(err)
				}
				compareVectors(t, y, want)
				if typ == 8 {
					token, logit, err := MulQ80Argmax(context.Background(), reader, tensor, x, options)
					if err != nil {
						t.Fatal(err)
					}
					best := 0
					for i := range y {
						if y[i] > y[best] {
							best = i
						}
					}
					if token != best || logit != y[best] {
						t.Fatalf("argmax = %d/%g, want %d/%g", token, logit, best, y[best])
					}
				}
			}
		})
	}
}

func TestAttentionSIMDReference(t *testing.T) {
	for _, dim := range []int{2, 7, 64, 67} {
		for _, window := range []int{0, 1, 5} {
			t.Run(fmt.Sprintf("dim%d/window%d", dim, window), func(t *testing.T) {
				const heads, kvHeads, positions = 4, 2, 13
				cache := NewKVCache(1, positions, kvHeads, dim)
				q := make([]float32, heads*dim)
				k, v := make([]float32, kvHeads*dim), make([]float32, kvHeads*dim)
				out := make([]float32, len(q))
				for i := range q {
					q[i] = float32(i%17-8) / 16
				}
				for pos := range positions {
					for i := range k {
						k[i] = float32((i+pos)%19-9) / 16
						v[i] = float32((i+pos)%23-11) / 16
					}
					options := AttentionOptions{Sinks: []float32{0, 1, -1, 2}, SlidingWindow: window}
					ForwardAttention(q, k, v, cache, 0, pos, heads, kvHeads, dim, out, options)
					want := make([]float32, len(q))
					start := 0
					if window > 0 {
						start = max(0, pos-window+1)
					}
					for h := range heads {
						scores := make([]float64, pos+1)
						maxScore := float64(options.Sinks[h])
						for p := start; p <= pos; p++ {
							var dot float64
							for i := range dim {
								dot += float64(q[h*dim+i]) * float64(cache.Keys[0][p][(h/2)*dim+i])
							}
							scores[p] = dot / math.Sqrt(float64(dim))
							maxScore = max(maxScore, scores[p])
						}
						sum := math.Exp(float64(options.Sinks[h]) - maxScore)
						for p := start; p <= pos; p++ {
							scores[p] = math.Exp(scores[p] - maxScore)
							sum += scores[p]
						}
						for i := range dim {
							var value float64
							for p := start; p <= pos; p++ {
								value += scores[p] / sum * float64(cache.Values[0][p][h/2*dim+i])
							}
							want[h*dim+i] = float32(value)
						}
					}
					compareVectors(t, out, want)
				}
			})
		}
	}
}

func TestLayerScratchReuse(t *testing.T) {
	engine, fresh := prefillEngineFixture(t), prefillEngineFixture(t)
	for pos, token := range []int{0, 1, 2, 1, 0, 2} {
		for _, e := range []*Engine{engine, fresh} {
			e.Layers[0].FFNDownExpsBias = e.OutputNorm
			e.Layers[0].FFNDownExpsBias.Shape = []uint64{32, 1}
			if pos%2 != 0 {
				e.Layers[0].FFNDownExpsBias.Name = ""
			}
		}
		fresh.Scratch = NewLayerScratch(fresh.Config)
		got, err := engine.ForwardToken(context.Background(), token, pos)
		if err != nil {
			t.Fatal(err)
		}
		want, err := fresh.ForwardToken(context.Background(), token, pos)
		if err != nil {
			t.Fatal(err)
		}
		if got != want || !reflect.DeepEqual(engine.X, fresh.X) || !reflect.DeepEqual(engine.KVCache, fresh.KVCache) {
			t.Fatalf("reused scratch differs at position %d", pos)
		}
	}
}

func TestKVCacheStorageIsolation(t *testing.T) {
	cache := NewKVCache(2, 3, 2, 7)
	for layer := range cache.Keys {
		for pos := range cache.Keys[layer] {
			if len(cache.Keys[layer][pos]) != 14 || cap(cache.Keys[layer][pos]) != 14 ||
				len(cache.Values[layer][pos]) != 14 || cap(cache.Values[layer][pos]) != 14 {
				t.Fatal("KV slices can extend into adjacent positions")
			}
			cache.Keys[layer][pos][0] = float32(layer*3 + pos + 1)
			cache.Values[layer][pos][0] = -float32(layer*3 + pos + 1)
		}
	}
	for layer := range cache.Keys {
		for pos := range cache.Keys[layer] {
			want := float32(layer*3 + pos + 1)
			if cache.Keys[layer][pos][0] != want || cache.Values[layer][pos][0] != -want {
				t.Fatal("KV positions, layers, or keys/values overlap")
			}
		}
	}
}
