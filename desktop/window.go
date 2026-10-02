//go:build goexperiment.simd && linux

package desktop

import (
	"context"
	"errors"
	"fmt"
	"io"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"time"

	"Stream-PT/inference"

	"fyne.io/fyne/v2"
	"fyne.io/fyne/v2/container"
	"fyne.io/fyne/v2/dialog"
	"fyne.io/fyne/v2/driver/desktop"
	"fyne.io/fyne/v2/layout"
	"fyne.io/fyne/v2/storage"
	"fyne.io/fyne/v2/theme"
	"fyne.io/fyne/v2/widget"
)

const defaultPaths = "model/gpt-oss-120b-Q4_0-00001-of-00002.gguf\nmodel/gpt-oss-120b-Q4_0-00002-of-00002.gguf"

type studio struct {
	app                                              fyne.App
	window                                           fyne.Window
	run                                              inference.Runner
	paths, workers, windowMiB, maxTokens, prompt     *widget.Entry
	generate, stop, clear, copy, save, browse, reset *widget.Button
	response                                         *widget.RichText
	scroll                                           *container.Scroll
	status, details, metrics, issue                  *widget.Label
	progress                                         *widget.ProgressBarInfinite
	autoScroll, markdown                             *widget.Check
	output                                           string
	cancel                                           context.CancelFunc
	running, closing                                 bool
	sequence                                         int
}

func NewWindow(app fyne.App, run inference.Runner) fyne.Window {
	if run == nil {
		run = inference.Run
	}
	s := &studio{app: app, window: app.NewWindow("Stream-PT • Local inference studio"), run: run}
	prefs := app.Preferences()
	s.paths = widget.NewMultiLineEntry()
	s.paths.SetPlaceHolder("One GGUF shard path per line")
	s.paths.SetMinRowsVisible(3)
	s.paths.SetText(prefs.StringWithFallback("model-paths", defaultPaths))
	s.workers = settingEntry("Compute workers", prefs.StringWithFallback("workers", strconv.Itoa(runtime.GOMAXPROCS(0))))
	s.windowMiB = settingEntry("Window size in MiB", prefs.StringWithFallback("window-mib", "8"))
	s.maxTokens = settingEntry("Maximum new tokens", prefs.StringWithFallback("max-tokens", "128"))
	s.prompt = widget.NewMultiLineEntry()
	s.prompt.SetPlaceHolder("Ask a question or describe a task…")
	s.prompt.SetMinRowsVisible(4)
	s.prompt.Wrapping = fyne.TextWrapWord
	s.response = widget.NewRichTextWithText("Your response will appear here, one token at a time.")
	s.response.Wrapping = fyne.TextWrapWord
	s.scroll = container.NewVScroll(container.NewPadded(s.response))
	s.scroll.SetMinSize(fyne.NewSize(420, 240))
	s.status = widget.NewLabel("Ready • select your model and write a prompt")
	s.status.Wrapping = fyne.TextWrapWord
	s.details = widget.NewLabel("Local GPT-OSS GGUF • CPU inference • no cloud connection")
	s.details.Wrapping = fyne.TextWrapWord
	s.metrics = widget.NewLabel("0 tokens   ·   0.0s elapsed   ·   — tokens/s")
	s.issue = widget.NewLabel("")
	s.issue.Wrapping = fyne.TextWrapWord
	s.issue.Importance = widget.DangerImportance
	s.issue.Hide()
	s.progress = widget.NewProgressBarInfinite()
	s.progress.Hide()
	s.generate = widget.NewButtonWithIcon("Generate", theme.MediaPlayIcon(), s.start)
	s.generate.Importance = widget.HighImportance
	s.stop = widget.NewButtonWithIcon("Stop", theme.MediaStopIcon(), s.stopGeneration)
	s.stop.Disable()
	s.clear = widget.NewButtonWithIcon("Clear", theme.DeleteIcon(), s.clearResponse)
	s.copy = widget.NewButtonWithIcon("Copy", theme.ContentCopyIcon(), func() { s.window.Clipboard().SetContent(s.output) })
	s.save = widget.NewButtonWithIcon("Export", theme.DocumentSaveIcon(), s.export)
	s.copy.Disable()
	s.save.Disable()
	s.browse = widget.NewButtonWithIcon("Add GGUF shard", theme.FolderOpenIcon(), s.addShard)
	s.reset = widget.NewButton("Reset settings", s.resetSettings)
	s.autoScroll = widget.NewCheck("Follow streamed output", func(value bool) { prefs.SetBool("auto-scroll", value) })
	s.autoScroll.SetChecked(prefs.BoolWithFallback("auto-scroll", true))
	s.markdown = widget.NewCheck("Render Markdown when finished", func(value bool) {
		prefs.SetBool("markdown", value)
		if !s.running && s.output != "" {
			s.render(true)
		}
	})
	s.markdown.SetChecked(prefs.BoolWithFallback("markdown", true))

	appearance := widget.NewSelect([]string{"Dark", "Light", "System"}, nil)
	textSize := widget.NewSelect([]string{"Compact", "Standard", "Large"}, nil)
	applyTheme := func() {
		size := float32(14)
		if textSize.Selected == "Compact" {
			size = 12
		}
		if textSize.Selected == "Large" {
			size = 17
		}
		app.Settings().SetTheme(&studioTheme{mode: appearance.Selected, textSize: size})
	}
	appearance.OnChanged = func(value string) { prefs.SetString("appearance", value); applyTheme() }
	textSize.OnChanged = func(value string) { prefs.SetString("text-size", value); applyTheme() }
	appearance.SetSelected(prefs.StringWithFallback("appearance", "Dark"))
	if appearance.Selected == "" {
		appearance.SetSelected("Dark")
	}
	textSize.SetSelected(prefs.StringWithFallback("text-size", "Standard"))
	if textSize.Selected == "" {
		textSize.SetSelected("Standard")
	}

	modelHelp := wrapped("Add every shard belonging to the same model. Paths can be absolute or relative to the working directory.")
	engineHelp := wrapped("Greedy decoding · 2,048 cache positions\nThe mmap window is not a total RAM limit.\nEach request starts a fresh, independent context.")
	settings := container.NewVBox(
		widget.NewCard("Model", "File-backed weights", container.NewVBox(s.paths, s.browse, modelHelp)),
		widget.NewCard("Inference", "CPU & streaming", container.NewVBox(
			widget.NewForm(widget.NewFormItem("Workers", s.workers), widget.NewFormItem("Window (MiB)", s.windowMiB), widget.NewFormItem("New tokens", s.maxTokens)),
			engineHelp, s.reset)),
		widget.NewCard("Workspace", "Saved on this device", container.NewVBox(
			widget.NewForm(widget.NewFormItem("Appearance", appearance), widget.NewFormItem("Text size", textSize)), s.autoScroll, s.markdown)),
		wrapped(fmt.Sprintf("%s / %s · %d logical CPUs\nExperimental SIMD runtime", runtime.GOOS, runtime.GOARCH, runtime.NumCPU())),
	)
	settingsScroll := container.NewVScroll(settings)
	settingsScroll.SetMinSize(fyne.NewSize(320, 0))
	responseHeader := container.NewHBox(widget.NewLabelWithStyle("ASSISTANT", fyne.TextAlignLeading, fyne.TextStyle{Bold: true}), layout.NewSpacer(), s.copy, s.save, s.clear)
	responsePane := container.NewBorder(container.NewVBox(responseHeader, widget.NewSeparator()), nil, nil, nil, s.scroll)
	promptPane := widget.NewCard("Prompt", "Independent request • Ctrl+Enter to generate", container.NewVBox(s.prompt,
		container.NewHBox(layout.NewSpacer(), s.stop, s.generate)))
	workspace := container.NewBorder(container.NewVBox(s.details, widget.NewSeparator()),
		container.NewVBox(s.issue, s.progress, s.status, s.metrics, promptPane), nil, nil, responsePane)
	split := container.NewHSplit(settingsScroll, container.NewPadded(workspace))
	split.Offset = 0.29
	title := widget.NewLabelWithStyle("Stream-PT", fyne.TextAlignLeading, fyne.TextStyle{Bold: true})
	header := container.NewHBox(title, widget.NewLabel("/  Local inference studio"), layout.NewSpacer(), widget.NewLabel("PRIVATE BY DESIGN"))
	s.window.SetContent(container.NewPadded(container.NewBorder(container.NewVBox(header, widget.NewSeparator()), nil, nil, nil, split)))
	s.window.Resize(fyne.NewSize(1180, 820))
	s.window.SetCloseIntercept(func() {
		if s.running {
			s.closing = true
			s.stopGeneration()
			s.status.SetText("Stopping inference before closing…")
			return
		}
		s.window.SetCloseIntercept(nil)
		s.window.Close()
	})
	s.window.Canvas().AddShortcut(&desktop.CustomShortcut{KeyName: fyne.KeyReturn, Modifier: fyne.KeyModifierControl}, func(fyne.Shortcut) { s.start() })
	return s.window
}

func settingEntry(placeholder, value string) *widget.Entry {
	e := widget.NewEntry()
	e.SetPlaceHolder(placeholder)
	e.SetText(value)
	return e
}

func wrapped(text string) *widget.Label {
	l := widget.NewLabel(text)
	l.Wrapping = fyne.TextWrapWord
	return l
}

func (s *studio) start() {
	if s.running {
		return
	}
	r, err := inference.ParseRequest(s.paths.Text, s.workers.Text, s.windowMiB.Text, s.maxTokens.Text, s.prompt.Text)
	if err != nil {
		s.issue.SetText(err.Error())
		s.issue.Show()
		return
	}
	p := s.app.Preferences()
	p.SetString("model-paths", strings.Join(r.Paths, "\n"))
	p.SetString("workers", strconv.Itoa(r.Workers))
	p.SetString("window-mib", strconv.Itoa(r.WindowMiB))
	p.SetString("max-tokens", strconv.Itoa(r.MaxTokens))
	s.clearResponse()
	s.running = true
	s.sequence++
	id := s.sequence
	ctx, cancel := context.WithCancel(context.Background())
	s.cancel = cancel
	s.setBusy(true)
	s.status.SetText("Preparing inference…")
	started := time.Now()
	count := 0
	var firstToken time.Duration
	updateMetrics := func() {
		elapsed := time.Since(started)
		ttft := "—"
		if count > 0 {
			ttft = firstToken.Round(time.Millisecond).String()
		}
		s.metrics.SetText(fmt.Sprintf("%d tokens   ·   %.1fs elapsed   ·   %.2f tokens/s   ·   first token %s", count, elapsed.Seconds(), float64(count)/elapsed.Seconds(), ttft))
	}
	go func() {
		ticker := time.NewTicker(250 * time.Millisecond)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				fyne.DoAndWait(func() {
					if s.running && s.sequence == id {
						updateMetrics()
					}
				})
			}
		}
	}()
	go func() {
		var text strings.Builder
		err := s.run(ctx, r, func(event inference.Event) {
			if ctx.Err() != nil {
				return
			}
			if event.IsToken {
				text.WriteString(event.Token)
			}
			snapshot := strings.ToValidUTF8(text.String(), "")
			fyne.DoAndWait(func() {
				if ctx.Err() != nil {
					return
				}
				if event.Detail != "" {
					s.details.SetText(event.Detail)
				}
				if event.Status != "" {
					s.status.SetText(event.Status)
				}
				if event.IsToken {
					if count == 0 {
						firstToken = time.Since(started)
					}
					count++
					s.output = snapshot
					s.status.SetText("Streaming response…")
					s.render(false)
					updateMetrics()
				}
			})
		})
		cancel()
		fyne.DoAndWait(func() {
			s.running = false
			s.cancel = nil
			s.setBusy(false)
			updateMetrics()
			s.render(true)
			switch {
			case errors.Is(err, context.Canceled):
				s.status.SetText("Stopped • partial response retained")
			case err != nil:
				s.status.SetText("Inference failed")
				s.issue.SetText(err.Error())
				s.issue.Show()
			case count == r.MaxTokens:
				s.status.SetText("Complete • token limit reached")
			default:
				s.status.SetText("Complete • end of response")
			}
			if s.closing {
				s.window.SetCloseIntercept(nil)
				s.window.Close()
			}
		})
	}()
}

func (s *studio) setBusy(busy bool) {
	for _, e := range []*widget.Entry{s.paths, s.workers, s.windowMiB, s.maxTokens, s.prompt} {
		if busy {
			e.Disable()
		} else {
			e.Enable()
		}
	}
	for _, b := range []*widget.Button{s.generate, s.browse, s.reset, s.clear} {
		if busy {
			b.Disable()
		} else {
			b.Enable()
		}
	}
	if busy {
		s.stop.Enable()
		s.progress.Show()
		s.progress.Start()
	} else {
		s.stop.Disable()
		s.progress.Stop()
		s.progress.Hide()
	}
}

func (s *studio) stopGeneration() {
	if s.cancel != nil {
		s.cancel()
		s.stop.Disable()
		s.status.SetText("Stopping… waiting for the current operation")
	}
}

func (s *studio) render(finished bool) {
	if s.output == "" {
		return
	}
	if finished && s.markdown.Checked {
		parsed := widget.NewRichTextFromMarkdown(s.output)
		s.response.Segments = localSegments(parsed.Segments)
		s.response.Refresh()
	} else {
		s.response.Segments = []widget.RichTextSegment{&widget.TextSegment{Text: s.output, Style: widget.RichTextStyle{Inline: true}}}
		s.response.Refresh()
	}
	s.copy.Enable()
	s.save.Enable()
	if s.autoScroll.Checked {
		s.scroll.ScrollToBottom()
	}
}

func localSegments(segments []widget.RichTextSegment) []widget.RichTextSegment {
	for i, segment := range segments {
		switch v := segment.(type) {
		case *widget.ImageSegment:
			segments[i] = &widget.TextSegment{Text: "[Image omitted: " + v.Source.String() + "]", Style: widget.RichTextStyleInline}
		case *widget.ParagraphSegment:
			v.Texts = localSegments(v.Texts)
		case *widget.ListSegment:
			v.Items = localSegments(v.Items)
		case *widget.TableSegment:
			for j := range v.Headers {
				v.Headers[j] = localSegments(v.Headers[j])
			}
			for j := range v.Rows {
				for k := range v.Rows[j] {
					v.Rows[j][k] = localSegments(v.Rows[j][k])
				}
			}
		}
	}
	return segments
}

func (s *studio) clearResponse() {
	s.output = ""
	s.response.Segments = []widget.RichTextSegment{&widget.TextSegment{Text: "Your response will appear here, one token at a time."}}
	s.response.Refresh()
	s.scroll.ScrollToTop()
	s.issue.Hide()
	s.copy.Disable()
	s.save.Disable()
	s.metrics.SetText("0 tokens   ·   0.0s elapsed   ·   — tokens/s")
	s.status.SetText("Ready • write a prompt")
}

func (s *studio) resetSettings() {
	s.paths.SetText(defaultPaths)
	s.workers.SetText(strconv.Itoa(runtime.GOMAXPROCS(0)))
	s.windowMiB.SetText("8")
	s.maxTokens.SetText("128")
	p := s.app.Preferences()
	for _, key := range []string{"model-paths", "workers", "window-mib", "max-tokens"} {
		p.RemoveValue(key)
	}
}

func (s *studio) addShard() {
	d := dialog.NewFileOpen(func(reader fyne.URIReadCloser, err error) {
		if err != nil {
			dialog.ShowError(err, s.window)
			return
		}
		if reader == nil {
			return
		}
		path := reader.URI().Path()
		if err := reader.Close(); err != nil {
			dialog.ShowError(err, s.window)
			return
		}
		for _, existing := range strings.Split(s.paths.Text, "\n") {
			if filepath.Clean(strings.TrimSpace(existing)) == filepath.Clean(path) {
				return
			}
		}
		if strings.TrimSpace(s.paths.Text) == "" {
			s.paths.SetText(path)
		} else {
			s.paths.SetText(strings.TrimSpace(s.paths.Text) + "\n" + path)
		}
	}, s.window)
	d.SetFilter(storage.NewExtensionFileFilter([]string{".gguf"}))
	d.Resize(fyne.NewSize(800, 520))
	d.Show()
}

func (s *studio) export() {
	text := s.output
	d := dialog.NewFileSave(func(writer fyne.URIWriteCloser, err error) {
		if err != nil {
			dialog.ShowError(err, s.window)
			return
		}
		if writer == nil {
			return
		}
		_, err = io.WriteString(writer, text)
		err = errors.Join(err, writer.Close())
		if err != nil {
			dialog.ShowError(err, s.window)
		}
	}, s.window)
	d.SetFileName("stream-pt-response.txt")
	d.Resize(fyne.NewSize(800, 520))
	d.Show()
}
