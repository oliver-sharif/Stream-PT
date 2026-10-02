//go:build goexperiment.simd

package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"time"

	"Stream-PT/forward"
	"Stream-PT/ggufindex"
	ggufmmap "Stream-PT/ggufmap"
)

func main() {
	threadsFlag := flag.Int("threads", runtime.GOMAXPROCS(0), "Number of parallel worker goroutines / compute threads")
	flag.IntVar(threadsFlag, "t", runtime.GOMAXPROCS(0), "Number of parallel worker goroutines (shorthand)")

	windowMBFlag := flag.Int("window-mb", 8, "Streaming mmap chunk window size in MiB")
	flag.IntVar(windowMBFlag, "w", 8, "Streaming mmap chunk window size in MiB (shorthand)")

	maxTokensFlag := flag.Int("max-tokens", 5, "Maximum number of tokens to generate")
	flag.IntVar(maxTokensFlag, "n", 5, "Maximum number of tokens to generate (shorthand)")

	promptFlag := flag.String("prompt", "", "Prompt / input text for generation")
	flag.StringVar(promptFlag, "p", "", "Prompt / input text for generation (shorthand)")

	flag.Parse()

	paths := []string{
		"model/gpt-oss-120b-Q4_0-00001-of-00002.gguf",
		"model/gpt-oss-120b-Q4_0-00002-of-00002.gguf",
	}

	model, err := ggufindex.Open(paths)
	if err != nil {
		log.Fatal(err)
	}

	reader, err := ggufmmap.Open(model)
	if err != nil {
		log.Fatal(err)
	}
	defer func() {
		if err := reader.Close(); err != nil {
			log.Print(err)
		}
	}()

	fmt.Println("╭────────────────────────────────────────────────────────────╮")
	fmt.Println("│                    SYSTEM & MODELL                         │")
	fmt.Println("├────────────────────────────────────────────────────────────┤")
	fmt.Printf("│ CPU:    %-51s │\n", cpuModel())
	fmt.Printf("│ RAM:    %-51s │\n", totalRAM())
	fmt.Printf("│ Modell: %-51s │\n", modelDescription(paths))
	fmt.Printf("│ Layer:  %-51d │\n", model.LayerCount)
	fmt.Println("╰────────────────────────────────────────────────────────────╯")
	fmt.Printf("Gewichte aus %d Dateien bleiben dateibasiert geladen.\n", len(model.Paths))

	threads := *threadsFlag
	if threads <= 0 {
		threads = runtime.GOMAXPROCS(0)
	}

	windowBytes := uint64(*windowMBFlag) * 1024 * 1024
	if windowBytes == 0 {
		windowBytes = forward.DefaultQ40WindowBytes
	}

	engineOpts := forward.EngineOptions{
		Workers:     threads,
		WindowBytes: windowBytes,
	}

	engine, err := forward.NewEngineWithOptions(model, reader, engineOpts)
	if err != nil {
		log.Fatalf("Failed to initialize inference engine: %v", err)
	}

	prompt := *promptFlag
	if prompt == "" {
		if flag.NArg() > 0 {
			prompt = strings.Join(flag.Args(), " ")
		} else {
			prompt = "Hallo, sag nur Ja!"
		}
	}

	fmt.Printf("\n--- Streaming Inference Pipeline Ready ---\n")
	fmt.Printf("Workers (Threads): %d\n", engine.Options.Workers)
	fmt.Printf("Chunk Window:      %d MiB\n", engine.Options.WindowBytes/(1024*1024))
	fmt.Printf("Prompt:            %s\n", prompt)

	promptTokens, err := engine.Tokenizer.EncodeChatPrompt(prompt)
	if err != nil {
		log.Fatalf("Failed to encode chat prompt: %v", err)
	}
	fmt.Printf("Encoded %d prompt tokens: %v\n", len(promptTokens), promptTokens)

	ctx := context.Background()
	fmt.Printf("\nGenerating tokens (streaming layer-by-layer):\n")
	inferenceStart := time.Now()
	_, err = engine.Generate(ctx, promptTokens, *maxTokensFlag, func(tokenID int, text string) bool {
		fmt.Printf("[Token %d: %q]\n", tokenID, text)
		return true
	})
	fmt.Printf("Inferenzzeit: %s\n", time.Since(inferenceStart).Round(time.Millisecond))
	if err != nil {
		log.Printf("Inference step info: %v", err)
	}
}

func cpuModel() string {
	data, err := os.ReadFile("/proc/cpuinfo")
	if err != nil {
		return runtime.GOARCH
	}

	for _, line := range strings.Split(string(data), "\n") {
		if key, value, ok := strings.Cut(line, ":"); ok && strings.TrimSpace(key) == "model name" {
			return strings.TrimSpace(value)
		}
	}
	return runtime.GOARCH
}

func totalRAM() string {
	data, err := os.ReadFile("/proc/meminfo")
	if err != nil {
		return "Unbekannt"
	}

	for _, line := range strings.Split(string(data), "\n") {
		fields := strings.Fields(line)
		if len(fields) >= 2 && fields[0] == "MemTotal:" {
			kilobytes, err := strconv.ParseUint(fields[1], 10, 64)
			if err == nil {
				return fmt.Sprintf("%.1f GiB", float64(kilobytes)/(1024*1024))
			}
		}
	}
	return "Unbekannt"
}

func modelDescription(paths []string) string {
	if len(paths) == 0 {
		return "Unbekannt"
	}

	name := strings.TrimSuffix(filepath.Base(paths[0]), ".gguf")
	if separator := strings.LastIndex(name, "-of-"); separator >= 0 {
		if shard := strings.LastIndex(name[:separator], "-"); shard >= 0 {
			name = name[:shard]
		}
	}
	if len(paths) > 1 {
		return fmt.Sprintf("%s (%d Dateien)", name, len(paths))
	}
	return name
}
