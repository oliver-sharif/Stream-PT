//go:build goexperiment.simd

package forward

import (
	"slices"
	"sync"
	"unicode/utf8"
)

type tokenizerNode struct {
	id       int
	first    uint32
	count    uint16
	terminal bool
}

type tokenizerEdge struct {
	child uint32
	label byte
}

type tokenizerEncoder struct {
	once  sync.Once
	root  [256]uint32
	nodes []tokenizerNode
	edges []tokenizerEdge
}

func tokenizerCommonPrefix(a, b string) int {
	i := 0
	for i < len(a) && i < len(b) && a[i] == b[i] {
		i++
	}
	for i > 0 && i < len(b) && !utf8.RuneStart(b[i]) {
		i--
	}
	return i
}

func (e *tokenizerEncoder) next(node uint32, label byte) uint32 {
	n := e.nodes[node]
	edges := e.edges[n.first : n.first+uint32(n.count)]
	if len(edges) <= 8 {
		for _, edge := range edges {
			if edge.label == label {
				return edge.child
			}
		}
		return 0
	}
	lo, hi := 0, len(edges)
	for lo < hi {
		mid := lo + (hi-lo)/2
		if edges[mid].label < label {
			lo = mid + 1
		} else {
			hi = mid
		}
	}
	if lo < len(edges) && edges[lo].label == label {
		return edges[lo].child
	}
	return 0
}

func (t *Tokenizer) initEncoder() {
	t.encoder.once.Do(func() {
		u2b := t.unicodeToByte
		if u2b == nil {
			_, u2b = initByteToUnicode()
		}
		keys := make([]string, 0, len(t.TokenMap))
		for token := range t.TokenMap {
			valid := len(token) > 0
			for _, r := range token {
				if _, ok := u2b[r]; !ok {
					valid = false
					break
				}
			}
			// Input bytes can only produce glyphs from the BPE alphabet.
			if valid {
				keys = append(keys, token)
			}
		}
		slices.Sort(keys)
		nodeCount, maxDepth := 1, 0
		previous := ""
		for _, token := range keys {
			prefix := tokenizerCommonPrefix(previous, token)
			nodeCount += utf8.RuneCountInString(token[prefix:])
			maxDepth = max(maxDepth, utf8.RuneCountInString(token))
			previous = token
		}
		e := &t.encoder
		e.nodes = make([]tokenizerNode, 1, nodeCount)
		// During construction, child holds the parent; edge index + 1 is the child.
		parents := make([]tokenizerEdge, 0, nodeCount-1)
		path := make([]uint32, 1, maxDepth+1)
		previous = ""
		for _, token := range keys {
			prefix := tokenizerCommonPrefix(previous, token)
			path = path[:utf8.RuneCountInString(token[:prefix])+1]
			for _, r := range token[prefix:] {
				parent := path[len(path)-1]
				child := uint32(len(e.nodes))
				e.nodes = append(e.nodes, tokenizerNode{})
				e.nodes[parent].count++
				parents = append(parents, tokenizerEdge{child: parent, label: u2b[r]})
				path = append(path, child)
			}
			node := &e.nodes[path[len(path)-1]]
			node.id, node.terminal = t.TokenMap[token], true
			previous = token
		}
		e.edges = make([]tokenizerEdge, len(parents))
		var end uint32
		for i := range e.nodes {
			end += uint32(e.nodes[i].count)
			e.nodes[i].first = end
		}
		for i, edge := range parents {
			parent := &e.nodes[edge.child]
			parent.first--
			e.edges[parent.first] = tokenizerEdge{child: uint32(i + 1), label: edge.label}
		}
		for _, node := range e.nodes {
			slices.SortFunc(e.edges[node.first:node.first+uint32(node.count)], func(a, b tokenizerEdge) int {
				return int(a.label) - int(b.label)
			})
		}
		for _, edge := range e.edges[:e.nodes[0].count] {
			e.root[edge.label] = edge.child
		}
	})
}
