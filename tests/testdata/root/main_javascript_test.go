//go:build goexperiment.simd

package main

import (
	"os/exec"
	"strings"
	"testing"
)

func TestIndexEnglishLocale(t *testing.T) {
	for _, text := range []string{`<html lang="en">`, "Processing prompt", "Thinking complete", "Generating answer", "Thinking ended", "Processing complete", " reused"} {
		if !strings.Contains(indexHTML, text) {
			t.Errorf("missing English UI text %q", text)
		}
	}
	if strings.Contains(indexHTML, `lang="de"`) || strings.ContainsAny(indexHTML, "\u00e4\u00f6\u00fc\u00df\u00c4\u00d6\u00dc") {
		t.Error("unexpected non-English UI locale or text")
	}
}

func TestIndexJavaScriptSyntax(t *testing.T) {
	node, err := exec.LookPath("node")
	if err != nil {
		t.Skip("Node.js not available for optional browser script syntax check")
	}
	_, script, ok := strings.Cut(indexHTML, "<script>")
	if !ok {
		t.Fatal("missing browser script")
	}
	script, _, ok = strings.Cut(script, "</script>")
	if !ok {
		t.Fatal("unterminated browser script")
	}
	command := exec.Command(node, "--check")
	command.Stdin = strings.NewReader(script)
	if output, err := command.CombinedOutput(); err != nil {
		t.Fatalf("browser script syntax: %v\n%s", err, output)
	}
}
