//go:build goexperiment.simd

package main

import (
	"os"
	"os/exec"
	"strings"
	"testing"
)

func TestCLIThreadDefaults(t *testing.T) {
	cmd := exec.Command("go", "run", ".", "-h")
	for _, value := range os.Environ() {
		if !strings.HasPrefix(value, "GOMAXPROCS=") {
			cmd.Env = append(cmd.Env, value)
		}
	}
	cmd.Env = append(cmd.Env, "GOMAXPROCS=8")
	output, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("CLI help: %v\n%s", err, output)
	}
	for _, flag := range []string{"-threads int", "-t int"} {
		_, description, ok := strings.Cut(string(output), flag+"\n")
		if !ok || !strings.HasPrefix(strings.TrimSpace(description), "Number of parallel worker goroutines") ||
			!strings.Contains(strings.SplitN(description, "\n", 2)[0], "(default 8)") {
			t.Fatalf("%s must preserve GOMAXPROCS=8:\n%s", flag, output)
		}
	}
}
