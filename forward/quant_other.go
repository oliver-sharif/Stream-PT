//go:build goexperiment.simd && !amd64

package forward

func quantDotQ40(row []byte, x []float32) float32 {
	return quantDotPortable(row, x, true)
}

func quantDotQ80(row []byte, x []float32) float32 {
	return quantDotPortable(row, x, false)
}

func quantDotQ40Batch(row []byte, x, y []float32, input, output, rowIndex, batch int) {
	quantQ40BatchPortable(row, x, y, input, output, rowIndex, batch)
}
