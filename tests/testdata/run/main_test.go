package main

import (
	"os"
	"path/filepath"
	"testing"
)

func overlayFixture(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	for _, name := range []string{"root", "forward", "ggufmap", "run"} {
		if err := os.MkdirAll(filepath.Join(root, "tests", "testdata", name), 0700); err != nil {
			t.Fatal(err)
		}
	}
	return root
}

func TestOverlayPackagePaths(t *testing.T) {
	root := overlayFixture(t)
	for source, target := range map[string]string{"root": ".", "forward": "forward", "ggufmap": "ggufmap", "run": "tests/run"} {
		path := filepath.Join(root, "tests", "testdata", source, "example_test.go")
		if err := os.WriteFile(path, []byte("package example\n"), 0600); err != nil {
			t.Fatal(err)
		}
		got, err := testOverlay(root)
		if err != nil || got[filepath.Join(root, target, "example_test.go")] != path {
			t.Fatalf("package %s: overlay %v, error %v", source, got, err)
		}
	}
}

func TestOverlayRejectsEmptyAndExistingTargets(t *testing.T) {
	root := overlayFixture(t)
	if _, err := testOverlay(root); err == nil {
		t.Fatal("accepted empty test collection")
	}
	for _, path := range []string{filepath.Join(root, "example_test.go"), filepath.Join(root, "tests", "testdata", "root", "example_test.go")} {
		if err := os.WriteFile(path, nil, 0600); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := testOverlay(root); err == nil {
		t.Fatal("overlay would shadow an existing source file")
	}
}

func TestOverlayRejectsSymlinks(t *testing.T) {
	root := overlayFixture(t)
	if err := os.Symlink("missing", filepath.Join(root, "tests", "testdata", "root", "example_test.go")); err != nil {
		t.Fatal(err)
	}
	if _, err := testOverlay(root); err == nil {
		t.Fatal("accepted symlink test source")
	}
}

func TestProjectRoot(t *testing.T) {
	root := t.TempDir()
	if _, err := projectRoot(root); err == nil {
		t.Fatal("accepted directory without go.mod")
	}
	if err := os.WriteFile(filepath.Join(root, "go.mod"), []byte("module fixture\n"), 0600); err != nil {
		t.Fatal(err)
	}
	child := filepath.Join(root, "tests", "run")
	if err := os.MkdirAll(child, 0700); err != nil {
		t.Fatal(err)
	}
	got, err := projectRoot(child)
	if err != nil || got != root {
		t.Fatalf("root %q, error %v", got, err)
	}
}
