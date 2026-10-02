//go:build goexperiment.simd && linux

package desktop

import (
	"image/color"

	"fyne.io/fyne/v2"
	"fyne.io/fyne/v2/theme"
)

type studioTheme struct {
	mode     string
	textSize float32
}

func (t *studioTheme) Color(name fyne.ThemeColorName, variant fyne.ThemeVariant) color.Color {
	if t.mode == "Dark" {
		variant = theme.VariantDark
	} else if t.mode == "Light" {
		variant = theme.VariantLight
	}
	if name == theme.ColorNamePrimary || name == theme.ColorNameFocus {
		return color.NRGBA{R: 82, G: 153, B: 244, A: 255}
	}
	if variant == theme.VariantDark {
		switch name {
		case theme.ColorNameBackground:
			return color.NRGBA{R: 15, G: 23, B: 38, A: 255}
		case theme.ColorNameInputBackground:
			return color.NRGBA{R: 23, G: 34, B: 52, A: 255}
		case theme.ColorNameForeground:
			return color.NRGBA{R: 229, G: 237, B: 248, A: 255}
		}
	}
	return theme.DefaultTheme().Color(name, variant)
}

func (t *studioTheme) Font(style fyne.TextStyle) fyne.Resource {
	return theme.DefaultTheme().Font(style)
}

func (t *studioTheme) Icon(name fyne.ThemeIconName) fyne.Resource {
	return theme.DefaultTheme().Icon(name)
}

func (t *studioTheme) Size(name fyne.ThemeSizeName) float32 {
	if name == theme.SizeNameText {
		return t.textSize
	}
	return theme.DefaultTheme().Size(name)
}
