package ggufindex

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

// Range represents [Start, End) as absolute byte offsets within a file.
type Range struct {
	File  string
	Start uint64
	End   uint64
}

type Tensor struct {
	Name  string
	Type  uint32
	Shape []uint64
	Range Range
}

type Layer struct {
	Number  int
	Tensors []Tensor

	// Spans enclose this layer's tensors, grouped by file.
	// They may also contain bytes belonging to other tensors.
	Spans []Range
}

type Model struct {
	Paths      []string
	LayerCount int
	Layers     []Layer
	Shared     []Tensor
	Alignment  uint64

	// Metadata contains all GGUF metadata values. The encoded values remain
	// in their source files and are read only when requested.
	Metadata map[string]MetadataValue
}

type tensorInfo struct {
	name   string
	shape  []uint64
	typ    uint32
	offset uint64
}

var layerName = regexp.MustCompile(`^blk\.([0-9]+)\.`)

// Open reads GGUF headers, metadata locations, and tensor directories.
// It does not load model weights. Split files may be supplied in any order.
func Open(paths []string) (*Model, error) {
	if len(paths) == 0 {
		return nil, fmt.Errorf("no GGUF files provided")
	}

	model := &Model{
		Paths:    append([]string(nil), paths...),
		Metadata: make(map[string]MetadataValue),
	}
	seenPaths := make(map[string]bool)
	seenTensors := make(map[string]bool)
	layers := make(map[int]*Layer)
	splitNumbers := make(map[uint64]bool)

	declaredCount := -1
	var splitCount uint64
	hasSplitCount := false

	for _, path := range paths {
		if seenPaths[path] {
			return nil, fmt.Errorf("duplicate file: %s", path)
		}
		seenPaths[path] = true

		tensors, numbers, entries, alignment, err := readFile(path)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", path, err)
		}

		if model.Alignment == 0 {
			model.Alignment = alignment
		} else if model.Alignment != alignment {
			return nil, fmt.Errorf("%s: inconsistent tensor alignment", path)
		}

		for key, entry := range entries {
			previous, exists := model.Metadata[key]
			if !exists {
				model.Metadata[key] = entry
				continue
			}

			// Each split has its own number. Retain the first entry while
			// validating every split number separately below.
			if key == "split.no" {
				continue
			}

			equal, err := sameMetadata(previous, entry)
			if err != nil {
				return nil, fmt.Errorf("%s: compare metadata %q: %w", path, key, err)
			}
			if !equal {
				return nil, fmt.Errorf("%s: inconsistent metadata %q", path, key)
			}
		}

		if count, ok := numbers["gpt_oss.block_count"]; ok {
			if count > uint64(^uint(0)>>1) {
				return nil, fmt.Errorf("%s: layer count is too large", path)
			}
			if declaredCount >= 0 && uint64(declaredCount) != count {
				return nil, fmt.Errorf("%s: inconsistent layer count", path)
			}
			declaredCount = int(count)
		}

		if count, ok := numbers["split.count"]; ok {
			if hasSplitCount && splitCount != count {
				return nil, fmt.Errorf("%s: inconsistent split count", path)
			}
			splitCount = count
			hasSplitCount = true
		}

		if number, ok := numbers["split.no"]; ok {
			if splitNumbers[number] {
				return nil, fmt.Errorf("%s: duplicate split %d", path, number)
			}
			splitNumbers[number] = true
		}

		for _, tensor := range tensors {
			if seenTensors[tensor.Name] {
				return nil, fmt.Errorf("%s: duplicate tensor %q", path, tensor.Name)
			}
			seenTensors[tensor.Name] = true

			match := layerName.FindStringSubmatch(tensor.Name)
			if match == nil {
				model.Shared = append(model.Shared, tensor)
				continue
			}

			number, err := strconv.Atoi(match[1])
			if err != nil {
				return nil, fmt.Errorf(
					"%s: invalid layer number in %q", path, tensor.Name,
				)
			}
			layer := layers[number]
			if layer == nil {
				layer = &Layer{Number: number}
				layers[number] = layer
			}
			layer.Tensors = append(layer.Tensors, tensor)
		}
	}

	if hasSplitCount && uint64(len(paths)) != splitCount {
		return nil, fmt.Errorf(
			"incomplete model: %d of %d splits provided",
			len(paths), splitCount,
		)
	}
	if len(splitNumbers) != 0 {
		if len(splitNumbers) != len(paths) {
			return nil, fmt.Errorf("a file is missing its split number")
		}
		for number := range paths {
			if !splitNumbers[uint64(number)] {
				return nil, fmt.Errorf("split %d is missing", number)
			}
		}
	}

	if declaredCount < 0 {
		declaredCount = len(layers)
	}
	model.LayerCount = declaredCount
	if len(layers) != declaredCount {
		return nil, fmt.Errorf(
			"expected %d layers, found %d", declaredCount, len(layers),
		)
	}

	for number := 0; number < declaredCount; number++ {
		layer := layers[number]
		if layer == nil {
			return nil, fmt.Errorf("layer %d is missing", number)
		}

		sort.Slice(layer.Tensors, func(i, j int) bool {
			return layer.Tensors[i].Name < layer.Tensors[j].Name
		})

		byFile := make(map[string]Range)
		for _, tensor := range layer.Tensors {
			span, exists := byFile[tensor.Range.File]
			if !exists {
				span = tensor.Range
			} else {
				if tensor.Range.Start < span.Start {
					span.Start = tensor.Range.Start
				}
				if tensor.Range.End > span.End {
					span.End = tensor.Range.End
				}
			}
			byFile[tensor.Range.File] = span
		}
		for _, span := range byFile {
			layer.Spans = append(layer.Spans, span)
		}
		sort.Slice(layer.Spans, func(i, j int) bool {
			return layer.Spans[i].File < layer.Spans[j].File
		})
		model.Layers = append(model.Layers, *layer)
	}

	sort.Slice(model.Shared, func(i, j int) bool {
		return model.Shared[i].Name < model.Shared[j].Name
	})
	return model, nil
}

func readFile(path string) (
	[]Tensor,
	map[string]uint64,
	map[string]MetadataValue,
	uint64,
	error,
) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, nil, 0, err
	}
	defer f.Close()

	stat, err := f.Stat()
	if err != nil {
		return nil, nil, nil, 0, err
	}
	if stat.Size() < 24 {
		return nil, nil, nil, 0, fmt.Errorf("file is too small")
	}

	var magic [4]byte
	if _, err := io.ReadFull(f, magic[:]); err != nil {
		return nil, nil, nil, 0, err
	}
	if string(magic[:]) != "GGUF" {
		return nil, nil, nil, 0, fmt.Errorf("not a GGUF file")
	}

	var version uint32
	if err := binary.Read(f, binary.LittleEndian, &version); err != nil {
		return nil, nil, nil, 0, err
	}
	if version != 2 && version != 3 {
		return nil, nil, nil, 0, fmt.Errorf(
			"unsupported GGUF version %d", version,
		)
	}

	var tensorCount, metadataCount uint64
	if err := binary.Read(f, binary.LittleEndian, &tensorCount); err != nil {
		return nil, nil, nil, 0, err
	}
	if err := binary.Read(f, binary.LittleEndian, &metadataCount); err != nil {
		return nil, nil, nil, 0, err
	}
	if tensorCount > 10_000_000 || metadataCount > 1_000_000 {
		return nil, nil, nil, 0, fmt.Errorf("implausible header count")
	}

	numbers := make(map[string]uint64)
	entries := make(map[string]MetadataValue)
	for i := uint64(0); i < metadataCount; i++ {
		key, err := readString(f)
		if err != nil {
			return nil, nil, nil, 0, err
		}
		if _, exists := entries[key]; exists {
			return nil, nil, nil, 0, fmt.Errorf(
				"duplicate metadata key %q", key,
			)
		}

		var typ uint32
		if err := binary.Read(f, binary.LittleEndian, &typ); err != nil {
			return nil, nil, nil, 0, err
		}
		start, err := f.Seek(0, io.SeekCurrent)
		if err != nil {
			return nil, nil, nil, 0, err
		}

		number, numeric, err := skipValue(f, typ, 0)
		if err != nil {
			return nil, nil, nil, 0, fmt.Errorf(
				"metadata %q: %w", key, err,
			)
		}
		end, err := f.Seek(0, io.SeekCurrent)
		if err != nil {
			return nil, nil, nil, 0, err
		}

		entries[key] = MetadataValue{
			Type: typ,
			Range: Range{
				File:  path,
				Start: uint64(start),
				End:   uint64(end),
			},
		}
		if numeric {
			numbers[key] = number
		}
	}

	infos := make([]tensorInfo, 0, int(tensorCount))
	for i := uint64(0); i < tensorCount; i++ {
		name, err := readString(f)
		if err != nil {
			return nil, nil, nil, 0, err
		}

		var dimensions uint32
		if err := binary.Read(f, binary.LittleEndian, &dimensions); err != nil {
			return nil, nil, nil, 0, err
		}
		if dimensions == 0 || dimensions > 8 {
			return nil, nil, nil, 0, fmt.Errorf(
				"tensor %q: invalid dimension count", name,
			)
		}

		info := tensorInfo{
			name:  name,
			shape: make([]uint64, dimensions),
		}
		for j := range info.shape {
			if err := binary.Read(
				f, binary.LittleEndian, &info.shape[j],
			); err != nil {
				return nil, nil, nil, 0, err
			}
		}
		if err := binary.Read(f, binary.LittleEndian, &info.typ); err != nil {
			return nil, nil, nil, 0, err
		}
		if err := binary.Read(f, binary.LittleEndian, &info.offset); err != nil {
			return nil, nil, nil, 0, err
		}
		infos = append(infos, info)
	}

	alignment := numbers["general.alignment"]
	if alignment == 0 {
		alignment = 32
	}
	if alignment > 1<<20 || alignment&(alignment-1) != 0 {
		return nil, nil, nil, 0, fmt.Errorf(
			"invalid alignment %d", alignment,
		)
	}

	position, err := f.Seek(0, io.SeekCurrent)
	if err != nil {
		return nil, nil, nil, 0, err
	}
	base := (uint64(position) + alignment - 1) &^ (alignment - 1)
	size := uint64(stat.Size())
	if base > size {
		return nil, nil, nil, 0, fmt.Errorf(
			"tensor data starts beyond the end of the file",
		)
	}

	tensors := make([]Tensor, 0, len(infos))
	for _, info := range infos {
		byteSize, err := tensorSize(info.typ, info.shape)
		if err != nil {
			return nil, nil, nil, 0, fmt.Errorf(
				"tensor %q: %w", info.name, err,
			)
		}
		if info.offset > size-base ||
			byteSize > size-base-info.offset {
			return nil, nil, nil, 0, fmt.Errorf(
				"tensor %q: byte range extends beyond the file",
				info.name,
			)
		}

		start := base + info.offset
		tensors = append(tensors, Tensor{
			Name:  info.name,
			Type:  info.typ,
			Shape: info.shape,
			Range: Range{
				File:  path,
				Start: start,
				End:   start + byteSize,
			},
		})
	}
	return tensors, numbers, entries, alignment, nil
}

func readString(r io.Reader) (string, error) {
	var length uint64
	if err := binary.Read(r, binary.LittleEndian, &length); err != nil {
		return "", err
	}
	if length > 1<<20 {
		return "", fmt.Errorf("GGUF string is too long: %d", length)
	}

	data := make([]byte, int(length))
	if _, err := io.ReadFull(r, data); err != nil {
		return "", err
	}
	return string(data), nil
}

func skipValue(f *os.File, typ uint32, depth int) (uint64, bool, error) {
	if depth > 8 {
		return 0, false, fmt.Errorf("metadata nesting is too deep")
	}

	if typ == 9 {
		var elementType uint32
		var count uint64
		if err := binary.Read(
			f, binary.LittleEndian, &elementType,
		); err != nil {
			return 0, false, err
		}
		if err := binary.Read(f, binary.LittleEndian, &count); err != nil {
			return 0, false, err
		}
		if count > 100_000_000 {
			return 0, false, fmt.Errorf("array is too large")
		}
		for i := uint64(0); i < count; i++ {
			if _, _, err := skipValue(
				f, elementType, depth+1,
			); err != nil {
				return 0, false, err
			}
		}
		return 0, false, nil
	}

	if typ == 8 {
		_, err := readString(f)
		return 0, false, err
	}

	width := map[uint32]int{
		0: 1, 1: 1,
		2: 2, 3: 2,
		4: 4, 5: 4, 6: 4,
		7:  1,
		10: 8, 11: 8, 12: 8,
	}[typ]
	if width == 0 {
		return 0, false, fmt.Errorf("unknown metadata type %d", typ)
	}

	var data [8]byte
	if _, err := io.ReadFull(f, data[:width]); err != nil {
		return 0, false, err
	}

	switch typ {
	case 0:
		return uint64(data[0]), true, nil
	case 2:
		return uint64(binary.LittleEndian.Uint16(data[:])), true, nil
	case 4:
		return uint64(binary.LittleEndian.Uint32(data[:])), true, nil
	case 10:
		return binary.LittleEndian.Uint64(data[:]), true, nil
	default:
		return 0, false, nil
	}
}

func tensorSize(typ uint32, shape []uint64) (uint64, error) {
	type layout struct {
		elements uint64
		bytes    uint64
	}
	layouts := map[uint32]layout{
		0:  {1, 4},
		1:  {1, 2},
		2:  {32, 18},
		3:  {32, 20},
		6:  {32, 22},
		7:  {32, 24},
		8:  {32, 34},
		9:  {32, 40},
		28: {1, 2},
		39: {32, 17},
	}
	spec, ok := layouts[typ]
	if !ok {
		return 0, fmt.Errorf(
			"unsupported GGML tensor type %d", typ,
		)
	}

	elements := uint64(1)
	for _, dimension := range shape {
		if dimension == 0 ||
			elements > ^uint64(0)/dimension {
			return 0, fmt.Errorf("invalid tensor shape")
		}
		elements *= dimension
	}
	if elements%spec.elements != 0 ||
		elements/spec.elements > ^uint64(0)/spec.bytes {
		return 0, fmt.Errorf(
			"invalid block size or tensor is too large",
		)
	}
	return elements / spec.elements * spec.bytes, nil
}

// TensorByName also searches tensors outside the transformer layers.
func (m *Model) TensorByName(name string) (Tensor, bool) {
	for _, tensor := range m.Shared {
		if tensor.Name == name {
			return tensor, true
		}
	}
	for _, layer := range m.Layers {
		for _, tensor := range layer.Tensors {
			if tensor.Name == name {
				return tensor, true
			}
		}
	}
	return Tensor{}, false
}

// IsLayerTensor reports whether a name matches the blk.<number>.* pattern.
func IsLayerTensor(name string) bool {
	return strings.HasPrefix(name, "blk.") &&
		layerName.MatchString(name)
}
