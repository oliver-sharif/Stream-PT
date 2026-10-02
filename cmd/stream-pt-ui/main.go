//go:build goexperiment.simd && linux

package main

import (
	"Stream-PT/desktop"

	"fyne.io/fyne/v2/app"
)

func main() {
	a := app.NewWithID("io.streampt.studio")
	desktop.NewWindow(a, nil).ShowAndRun()
}
