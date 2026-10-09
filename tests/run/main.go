// Command run executes centralized tests in their original Go packages.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
)

func projectRoot(start string) (string, error) {
	dir, err := filepath.Abs(start)
	if err != nil {
		return "", err
	}
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir, nil
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return "", fmt.Errorf("go.mod not found above %s", start)
		}
		dir = parent
	}
}

func testOverlay(root string) (map[string]string, error) {
	replacements := make(map[string]string)
	packages := map[string]string{
		"root":    ".",
		"forward": "forward",
		"ggufmap": "ggufmap",
		"run":     "tests/run",
	}
	for source, target := range packages {
		dir := filepath.Join(root, "tests", "testdata", source)
		entries, err := os.ReadDir(dir)
		if err != nil {
			return nil, err
		}
		for _, entry := range entries {
			if entry.IsDir() || !strings.HasSuffix(entry.Name(), "_test.go") {
				continue
			}
			if entry.Type()&fs.ModeSymlink != 0 {
				return nil, fmt.Errorf("test source must not be a symlink: %s", entry.Name())
			}
			original := filepath.Join(root, target, entry.Name())
			if _, err := os.Lstat(original); !os.IsNotExist(err) {
				return nil, fmt.Errorf("overlay target must not exist: %s", original)
			}
			replacements[original] = filepath.Join(dir, entry.Name())
		}
	}
	if len(replacements) == 0 {
		return nil, fmt.Errorf("no centralized tests found")
	}
	return replacements, nil
}

func run(ctx context.Context, args []string) error {
	cwd, err := os.Getwd()
	if err != nil {
		return err
	}
	root, err := projectRoot(cwd)
	if err != nil {
		return err
	}
	replacements, err := testOverlay(root)
	if err != nil {
		return err
	}
	data, err := json.Marshal(struct {
		Replace map[string]string
	}{replacements})
	if err != nil {
		return err
	}
	dir, err := os.MkdirTemp(filepath.Join(root, "tests"), ".overlay-")
	if err != nil {
		return err
	}
	defer os.RemoveAll(dir)
	overlay := filepath.Join(dir, "overlay.json")
	if err := os.WriteFile(overlay, data, 0600); err != nil {
		return err
	}
	if len(args) == 0 {
		args = []string{"./..."}
	}
	command := exec.CommandContext(ctx, "go", append([]string{"test", "-overlay", overlay}, args...)...)
	command.Dir = root
	command.Stdin, command.Stdout, command.Stderr = os.Stdin, os.Stdout, os.Stderr
	return command.Run()
}

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	if err := run(ctx, os.Args[1:]); err != nil {
		var exit *exec.ExitError
		if errors.As(err, &exit) {
			os.Exit(exit.ExitCode())
		}
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
