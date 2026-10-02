//go:build goexperiment.simd && linux

package tests

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"Stream-PT/desktop"
	"Stream-PT/inference"

	"fyne.io/fyne/v2"
	"fyne.io/fyne/v2/container"
	"fyne.io/fyne/v2/test"
	"fyne.io/fyne/v2/widget"
)

func TestDesktopRequestValidation(t *testing.T) {
	r, err := inference.ParseRequest(" a.gguf \n\nb.gguf\n", " 2 ", "8", "128", "Hello")
	if err != nil || len(r.Paths) != 2 || r.Paths[0] != "a.gguf" || r.Workers != 2 || r.WindowMiB != 8 || r.MaxTokens != 128 {
		t.Fatalf("parsed request = %+v, %v", r, err)
	}
	for _, tc := range []struct{ paths, workers, window, tokens, prompt string }{
		{"", "2", "8", "128", "Hello"},
		{"a.gguf\n./a.gguf", "2", "8", "128", "Hello"},
		{"a.gguf", "0", "8", "128", "Hello"},
		{"a.gguf", "2.5", "8", "128", "Hello"},
		{"a.gguf", "4097", "8", "128", "Hello"},
		{"a.gguf", "2", "-8", "128", "Hello"},
		{"a.gguf", "2", "1048577", "128", "Hello"},
		{"a.gguf", "2", "8", "0", "Hello"},
		{"a.gguf", "2", "8", "2049", "Hello"},
		{"a.gguf", "2", "8", "128", " \n "},
	} {
		if _, err := inference.ParseRequest(tc.paths, tc.workers, tc.window, tc.tokens, tc.prompt); err == nil {
			t.Errorf("accepted invalid settings: %+v", tc)
		}
	}
	for _, tc := range []struct {
		prompt, tokens int
		valid          bool
	}{
		{1, 2048, true}, {2048, 1, true}, {2047, 2, true}, {2048, 2, false}, {2049, 1, false}, {0, 1, false}, {1, 0, false},
	} {
		if err := inference.CheckContext(tc.prompt, tc.tokens); (err == nil) != tc.valid {
			t.Errorf("CheckContext(%d, %d) = %v", tc.prompt, tc.tokens, err)
		}
	}
}

func TestDesktopBackendErrors(t *testing.T) {
	r := inference.Request{Paths: []string{filepath.Join(t.TempDir(), "missing.gguf")}, Workers: 1, WindowMiB: 8, MaxTokens: 1, Prompt: "Hello"}
	if err := inference.Run(context.Background(), r, nil); err == nil || !strings.Contains(err.Error(), "open model") {
		t.Fatalf("missing model error = %v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := inference.Run(ctx, r, func(inference.Event) { t.Error("event after cancellation") }); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled run = %v", err)
	}
	if err := inference.Run(nil, r, nil); err == nil {
		t.Fatal("nil context accepted")
	}
}

// The Fyne test driver invokes background callbacks directly. Serialize them like
// the native event loop so asynchronous UI tests exercise the same ownership rule.
type studioTestDriver struct {
	fyne.Driver
	mu sync.Mutex
}

func (d *studioTestDriver) DoFromGoroutine(fn func(), _ bool) {
	d.mu.Lock()
	defer d.mu.Unlock()
	fn()
}

type studioTestApp struct {
	fyne.App
	driver *studioTestDriver
}

func (a *studioTestApp) Driver() fyne.Driver { return a.driver }

type studioTestWindow struct {
	fyne.Window
	clipboard fyne.Clipboard
	intercept func()
	closed    bool
}

func (w *studioTestWindow) Clipboard() fyne.Clipboard { return w.clipboard }

func (w *studioTestWindow) SetCloseIntercept(fn func()) { w.intercept = fn }

func (w *studioTestWindow) Close() {
	if w.closed {
		return
	}
	if w.intercept != nil {
		w.intercept()
		return
	}
	w.closed = true
	w.Window.Close()
}

func (a *studioTestApp) NewWindow(title string) fyne.Window {
	return &studioTestWindow{Window: a.App.NewWindow(title), clipboard: test.NewClipboard()}
}

func desktopFixture(t *testing.T, runner inference.Runner) fyne.Window {
	t.Helper()
	base := test.NewApp()
	a := &studioTestApp{App: base, driver: &studioTestDriver{Driver: base.Driver()}}
	fyne.SetCurrentApp(a)
	w := desktop.NewWindow(a, runner)
	t.Cleanup(func() {
		fyne.DoAndWait(func() { w.Close() })
		awaitStudio(t, func() bool { return w.(*studioTestWindow).closed })
		base.Quit()
	})
	return w
}

func studioObjects(root fyne.CanvasObject) []fyne.CanvasObject {
	objects := []fyne.CanvasObject{root}
	var children []fyne.CanvasObject
	switch o := root.(type) {
	case *fyne.Container:
		children = o.Objects
	case *container.Split:
		children = []fyne.CanvasObject{o.Leading, o.Trailing}
	case *container.Scroll:
		children = []fyne.CanvasObject{o.Content}
	case *widget.Card:
		children = []fyne.CanvasObject{o.Content}
	case *widget.Form:
		for _, item := range o.Items {
			children = append(children, item.Widget)
		}
	}
	for _, child := range children {
		if child != nil {
			objects = append(objects, studioObjects(child)...)
		}
	}
	return objects
}

func studioButton(w fyne.Window, text string) *widget.Button {
	for _, obj := range studioObjects(w.Content()) {
		if b, ok := obj.(*widget.Button); ok && b.Text == text {
			return b
		}
	}
	panic("missing button " + text)
}

func studioEntry(w fyne.Window, placeholder string) *widget.Entry {
	for _, obj := range studioObjects(w.Content()) {
		if e, ok := obj.(*widget.Entry); ok && e.PlaceHolder == placeholder {
			return e
		}
	}
	panic("missing entry " + placeholder)
}

func studioHasLabel(w fyne.Window, prefix string) bool {
	for _, obj := range studioObjects(w.Content()) {
		if l, ok := obj.(*widget.Label); ok && l.Visible() && strings.HasPrefix(l.Text, prefix) {
			return true
		}
	}
	return false
}

func awaitStudio(t *testing.T, check func() bool) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		ready := false
		fyne.DoAndWait(func() { ready = check() })
		if ready {
			return
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatal("desktop condition timed out")
}

func TestDesktopStreamsAndStops(t *testing.T) {
	requests := make(chan inference.Request, 2)
	runner := func(ctx context.Context, r inference.Request, emit func(inference.Event)) error {
		requests <- r
		emit(inference.Event{Status: "Prefilling…"})
		for _, token := range []string{"Hello ", "\xe2", "\x82\xac"} {
			emit(inference.Event{IsToken: true, Token: token})
		}
		<-ctx.Done()
		return ctx.Err()
	}
	w := desktopFixture(t, runner)
	fyne.DoAndWait(func() {
		studioEntry(w, "Ask a question or describe a task…").SetText("Test streaming")
		studioEntry(w, "Compute workers").SetText("2")
		test.Tap(studioButton(w, "Generate"))
		if !studioButton(w, "Generate").Disabled() || studioButton(w, "Stop").Disabled() {
			t.Error("incorrect busy controls")
		}
	})
	select {
	case r := <-requests:
		if r.Prompt != "Test streaming" || r.Workers != 2 {
			t.Fatalf("runner request = %+v", r)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("runner did not start")
	}
	awaitStudio(t, func() bool { return studioHasLabel(w, "3 tokens") })
	fyne.DoAndWait(func() {
		test.Tap(studioButton(w, "Copy"))
		if got := w.Clipboard().Content(); got != "Hello €" {
			t.Errorf("streamed response = %q", got)
		}
		if !studioEntry(w, "Compute workers").Disabled() {
			t.Error("settings editable during inference")
		}
		test.Tap(studioButton(w, "Stop"))
	})
	awaitStudio(t, func() bool { return studioHasLabel(w, "Stopped") && !studioButton(w, "Generate").Disabled() })
	fyne.DoAndWait(func() {
		if got := w.Clipboard().Content(); got != "Hello €" {
			t.Errorf("response lost after stopping: %q", got)
		}
		test.Tap(studioButton(w, "Clear"))
		if !studioButton(w, "Copy").Disabled() || !studioButton(w, "Export").Disabled() {
			t.Error("empty response actions enabled")
		}
		if fyne.CurrentApp().Preferences().String("workers") != "2" {
			t.Error("settings not saved")
		}
		if fyne.CurrentApp().Preferences().String("prompt") != "" {
			t.Error("prompt persisted unexpectedly")
		}
	})
}

func TestDesktopValidationCompletionAndFailure(t *testing.T) {
	calls := make(chan struct{}, 2)
	runner := func(_ context.Context, r inference.Request, emit func(inference.Event)) error {
		calls <- struct{}{}
		if r.Prompt == "fail" {
			return fmt.Errorf("unsupported model fixture")
		}
		emit(inference.Event{IsToken: true, Token: "**Done**"})
		return nil
	}
	w := desktopFixture(t, runner)
	fyne.DoAndWait(func() {
		test.Tap(studioButton(w, "Generate"))
		if !studioHasLabel(w, "enter a prompt") {
			t.Error("empty prompt error not visible")
		}
		select {
		case <-calls:
			t.Error("invalid request reached runner")
		default:
		}
		studioEntry(w, "Ask a question or describe a task…").SetText("finish")
		test.Tap(studioButton(w, "Generate"))
	})
	awaitStudio(t, func() bool { return studioHasLabel(w, "Complete") && !studioButton(w, "Generate").Disabled() })
	fyne.DoAndWait(func() {
		test.Tap(studioButton(w, "Copy"))
		if w.Clipboard().Content() != "**Done**" {
			t.Error("copy did not preserve raw Markdown")
		}
		studioEntry(w, "Ask a question or describe a task…").SetText("fail")
		test.Tap(studioButton(w, "Generate"))
	})
	awaitStudio(t, func() bool {
		return studioHasLabel(w, "unsupported model fixture") && !studioButton(w, "Generate").Disabled()
	})
	fyne.DoAndWait(func() {
		if !studioButton(w, "Stop").Disabled() {
			t.Error("Stop enabled after failure")
		}
	})
}

func TestDesktopCloseWaitsForWorker(t *testing.T) {
	started, stopping, release := make(chan struct{}), make(chan struct{}), make(chan struct{})
	runner := func(ctx context.Context, _ inference.Request, _ func(inference.Event)) error {
		close(started)
		<-ctx.Done()
		close(stopping)
		<-release
		return ctx.Err()
	}
	w := desktopFixture(t, runner)
	fyne.DoAndWait(func() {
		studioEntry(w, "Ask a question or describe a task…").SetText("Close test")
		test.Tap(studioButton(w, "Generate"))
	})
	select {
	case <-started:
	case <-time.After(5 * time.Second):
		t.Fatal("worker did not start")
	}
	fyne.DoAndWait(func() {
		w.Close()
		if w.(*studioTestWindow).closed {
			t.Error("closed before worker cleanup")
		}
	})
	select {
	case <-stopping:
	case <-time.After(5 * time.Second):
		t.Fatal("close did not cancel worker")
	}
	close(release)
	awaitStudio(t, func() bool { return w.(*studioTestWindow).closed })
}

func TestDesktopRendersWorkspace(t *testing.T) {
	w := desktopFixture(t, nil)
	fyne.DoAndWait(func() {
		image := w.Canvas().Capture()
		if image.Bounds().Dx() < 1100 || image.Bounds().Dy() < 700 {
			t.Errorf("unexpected workspace size: %v", image.Bounds())
		}
		if !studioButton(w, "Stop").Disabled() || studioButton(w, "Generate").Disabled() {
			t.Error("incorrect initial controls")
		}
	})
}

func TestDesktopMarkdownDoesNotFetchImages(t *testing.T) {
	var fetches atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		fetches.Add(1)
		w.WriteHeader(http.StatusNotFound)
	}))
	defer server.Close()
	runner := func(_ context.Context, _ inference.Request, emit func(inference.Event)) error {
		emit(inference.Event{IsToken: true, Token: "# Result\n\n- ![image](" + server.URL + "/image.png)\n\n| Image |\n| --- |\n| ![table](" + server.URL + "/table.png) |"})
		return nil
	}
	w := desktopFixture(t, runner)
	fyne.DoAndWait(func() {
		studioEntry(w, "Ask a question or describe a task…").SetText("Markdown test")
		test.Tap(studioButton(w, "Generate"))
	})
	awaitStudio(t, func() bool { return studioHasLabel(w, "Complete") && !studioButton(w, "Generate").Disabled() })
	fyne.DoAndWait(func() {
		w.Canvas().Capture()
		if fetches.Load() != 0 {
			t.Error("model-generated images were fetched")
		}
	})
}
