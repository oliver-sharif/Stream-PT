package ggufindex

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"sort"
)

// MetadataValue identifies a GGUF metadata value in its source file.
// Start and End delimit the encoded value, excluding its key and type.
type MetadataValue struct {
	Type  uint32
	Range Range
}

// WriteJSON writes the model index, including every metadata value and tensor.
// Metadata arrays are streamed instead of being retained in memory.
func (m *Model) WriteJSON(w io.Writer) error {
	if m == nil {
		return fmt.Errorf("nil model")
	}

	header := struct {
		Paths      []string `json:"paths"`
		LayerCount int      `json:"layer_count"`
		Alignment  uint64   `json:"alignment"`
		Shared     []Tensor `json:"shared"`
		Layers     []Layer  `json:"layers"`
	}{
		Paths: m.Paths, LayerCount: m.LayerCount, Alignment: m.Alignment,
		Shared: m.Shared, Layers: m.Layers,
	}

	encoded, err := json.Marshal(header)
	if err != nil {
		return err
	}
	if len(encoded) == 0 || encoded[len(encoded)-1] != '}' {
		return fmt.Errorf("invalid model JSON")
	}
	if _, err := w.Write(encoded[:len(encoded)-1]); err != nil {
		return err
	}
	if _, err := io.WriteString(w, `,"metadata":{`); err != nil {
		return err
	}

	keys := make([]string, 0, len(m.Metadata))
	for key := range m.Metadata {
		keys = append(keys, key)
	}
	sort.Strings(keys)

	for i, key := range keys {
		if i != 0 {
			if _, err := io.WriteString(w, ","); err != nil {
				return err
			}
		}
		name, err := json.Marshal(key)
		if err != nil {
			return err
		}
		if _, err := w.Write(name); err != nil {
			return err
		}

		entry := m.Metadata[key]
		if _, err := fmt.Fprintf(w, `:{"type":%d,"value":`, entry.Type); err != nil {
			return err
		}
		if err := entry.WriteJSON(w); err != nil {
			return fmt.Errorf("metadata %q: %w", key, err)
		}
		if _, err := io.WriteString(w, "}"); err != nil {
			return err
		}
	}
	_, err = io.WriteString(w, "}}\n")
	return err
}

// WriteJSON writes one decoded metadata value without retaining arrays.
func (v MetadataValue) WriteJSON(w io.Writer) error {
	if v.Range.End < v.Range.Start || v.Range.End > uint64(math.MaxInt64) {
		return fmt.Errorf("invalid metadata range")
	}

	f, err := os.Open(v.Range.File)
	if err != nil {
		return err
	}
	defer f.Close()

	length := v.Range.End - v.Range.Start
	r := io.NewSectionReader(f, int64(v.Range.Start), int64(length))

	if err := writeValueJSON(w, r, v.Type, 0); err != nil {
		return err
	}

	position, err := r.Seek(0, io.SeekCurrent)
	if err != nil {
		return err
	}
	if uint64(position) != length {
		return fmt.Errorf("metadata value contains unread bytes")
	}

	return nil
}

func writeValueJSON(w io.Writer, r io.Reader, typ uint32, depth int) error {
	if depth > 8 {
		return fmt.Errorf("metadata nesting is too deep")
	}
	if typ == 9 {
		var elementType uint32
		var count uint64
		if err := binary.Read(r, binary.LittleEndian, &elementType); err != nil {
			return err
		}
		if err := binary.Read(r, binary.LittleEndian, &count); err != nil {
			return err
		}
		if count > 100_000_000 {
			return fmt.Errorf("metadata array is too large")
		}
		if _, err := io.WriteString(w, "["); err != nil {
			return err
		}
		for i := uint64(0); i < count; i++ {
			if i != 0 {
				if _, err := io.WriteString(w, ","); err != nil {
					return err
				}
			}
			if err := writeValueJSON(w, r, elementType, depth+1); err != nil {
				return err
			}
		}
		_, err := io.WriteString(w, "]")
		return err
	}

	var value any
	if typ == 8 {
		s, err := readString(r)
		if err != nil {
			return err
		}
		value = s
	} else {
		width := map[uint32]int{
			0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4,
			6: 4, 7: 1, 10: 8, 11: 8, 12: 8,
		}[typ]
		if width == 0 {
			return fmt.Errorf("unknown metadata type %d", typ)
		}
		var b [8]byte
		if _, err := io.ReadFull(r, b[:width]); err != nil {
			return err
		}
		switch typ {
		case 0:
			value = uint8(b[0])
		case 1:
			value = int8(b[0])
		case 2:
			value = binary.LittleEndian.Uint16(b[:])
		case 3:
			value = int16(binary.LittleEndian.Uint16(b[:]))
		case 4:
			value = binary.LittleEndian.Uint32(b[:])
		case 5:
			value = int32(binary.LittleEndian.Uint32(b[:]))
		case 6:
			value = math.Float32frombits(binary.LittleEndian.Uint32(b[:]))
		case 7:
			if b[0] > 1 {
				return fmt.Errorf("invalid boolean %d", b[0])
			}
			value = b[0] == 1
		case 10:
			value = binary.LittleEndian.Uint64(b[:])
		case 11:
			value = int64(binary.LittleEndian.Uint64(b[:]))
		case 12:
			value = math.Float64frombits(binary.LittleEndian.Uint64(b[:]))
		}
	}

	encoded, err := json.Marshal(value)
	if err != nil {
		return err
	}
	_, err = w.Write(encoded)
	return err
}

func sameMetadata(a, b MetadataValue) (bool, error) {
	if a.Type != b.Type {
		return false, nil
	}
	aLength := a.Range.End - a.Range.Start
	bLength := b.Range.End - b.Range.Start
	if aLength != bLength {
		return false, nil
	}

	left, err := os.Open(a.Range.File)
	if err != nil {
		return false, err
	}
	defer left.Close()
	right, err := os.Open(b.Range.File)
	if err != nil {
		return false, err
	}
	defer right.Close()

	l := io.NewSectionReader(left, int64(a.Range.Start), int64(aLength))
	r := io.NewSectionReader(right, int64(b.Range.Start), int64(bLength))
	var lb, rb [32 * 1024]byte
	for remaining := aLength; remaining != 0; {
		n := uint64(len(lb))
		if remaining < n {
			n = remaining
		}
		if _, err := io.ReadFull(l, lb[:n]); err != nil {
			return false, err
		}
		if _, err := io.ReadFull(r, rb[:n]); err != nil {
			return false, err
		}
		if !bytes.Equal(lb[:n], rb[:n]) {
			return false, nil
		}
		remaining -= n
	}
	return true, nil
}
