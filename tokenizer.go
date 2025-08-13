package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"math"
	"regexp"
	"strings"
)

// --- BPE Tokenizer ---

type Encoder struct {
	encoder     map[string]int
	decoder     map[int]string
	byteEncoder map[byte]rune
	byteDecoder map[rune]byte
	bpeRanks    map[[2]string]int
	cache       map[string]string
	pat         *regexp.Regexp
}

// TokenizerData represents the structure of tokenizer.json
type TokenizerData struct {
	Model struct {
		Type   string         `json:"type"`
		Vocab  map[string]int `json:"vocab"`
		Merges []string       `json:"merges"`
	} `json:"model"`
	PreTokenizer struct {
		Type    string `json:"type"`
		Pattern string `json:"pattern"`
	} `json:"pre_tokenizer"`
}

func bytesToUnicode() (map[byte]rune, map[rune]byte) {
	// Create the standard GPT-2 byte-to-unicode mapping
	bs := []int{}

	// Add printable ASCII characters
	for i := int('!'); i <= int('~'); i++ {
		bs = append(bs, i)
	}
	// Add Latin-1 supplement characters
	for i := int('¡'); i <= int('¬'); i++ {
		bs = append(bs, i)
	}
	for i := int('®'); i <= int('ÿ'); i++ {
		bs = append(bs, i)
	}

	cs := make([]int, len(bs))
	copy(cs, bs)

	// Add remaining bytes, mapping them to unused Unicode points
	n := 0
	for b := 0; b < 256; b++ {
		if !containsInt(bs, b) {
			bs = append(bs, b)
			cs = append(cs, 256+n)
			n++
		}
	}

	byteEncoder := make(map[byte]rune)
	byteDecoder := make(map[rune]byte)

	for i, b := range bs {
		byteEncoder[byte(b)] = rune(cs[i])
		byteDecoder[rune(cs[i])] = byte(b)
	}

	return byteEncoder, byteDecoder
}

func containsInt(slice []int, val int) bool {
	for _, v := range slice {
		if v == val {
			return true
		}
	}
	return false
}

func getPairs(word []string) [][2]string {
	if len(word) < 2 {
		return [][2]string{}
	}

	pairs := [][2]string{}
	for i := 0; i < len(word)-1; i++ {
		pairs = append(pairs, [2]string{word[i], word[i+1]})
	}
	return pairs
}

func NewEncoder(tokenizerPath string) (*Encoder, error) {
	file, err := ioutil.ReadFile(tokenizerPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read tokenizer file: %v", err)
	}

	var tokenizerData TokenizerData
	if err := json.Unmarshal(file, &tokenizerData); err != nil {
		return nil, fmt.Errorf("failed to parse tokenizer JSON: %v", err)
	}

	encoderMap := tokenizerData.Model.Vocab
	bpeMerges := tokenizerData.Model.Merges

	// Build BPE ranks from merges
	bpeRanks := make(map[[2]string]int)
	for i, mergeStr := range bpeMerges {
		parts := strings.Fields(mergeStr) // Use Fields instead of Split for better handling
		if len(parts) == 2 {
			bpeRanks[[2]string{parts[0], parts[1]}] = i
		}
	}

	// Build decoder map
	decoderMap := make(map[int]string)
	for k, v := range encoderMap {
		decoderMap[v] = k
	}

	// Create byte encoders/decoders
	byteEncoder, byteDecoder := bytesToUnicode()

	// GPT-2 regex pattern for tokenization
	// Original pattern has (?!\S) which isn't supported in Go, so we use a simpler approach
	pat := regexp.MustCompile(`'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+`)

	return &Encoder{
		encoder:     encoderMap,
		decoder:     decoderMap,
		byteEncoder: byteEncoder,
		byteDecoder: byteDecoder,
		bpeRanks:    bpeRanks,
		cache:       make(map[string]string),
		pat:         pat,
	}, nil
}

func (e *Encoder) bpe(token string) string {
	if val, ok := e.cache[token]; ok {
		return val
	}

	// Convert token to list of characters
	word := []string{}
	for _, r := range token {
		word = append(word, string(r))
	}

	if len(word) < 2 {
		e.cache[token] = token
		return token
	}

	pairs := getPairs(word)

	for len(pairs) > 0 {
		// Find the pair with the lowest rank (highest priority merge)
		minRank := math.MaxInt32
		var bigram [2]string
		found := false

		for _, pair := range pairs {
			if rank, ok := e.bpeRanks[pair]; ok {
				if rank < minRank {
					minRank = rank
					bigram = pair
					found = true
				}
			}
		}

		if !found {
			break
		}

		// Merge the bigram
		first, second := bigram[0], bigram[1]
		newWord := []string{}
		i := 0

		for i < len(word) {
			// Find next occurrence of first token
			j := -1
			for k := i; k < len(word); k++ {
				if word[k] == first {
					j = k
					break
				}
			}

			if j == -1 {
				// No more occurrences of first, add rest of word
				newWord = append(newWord, word[i:]...)
				break
			}

			// Add tokens before the match
			newWord = append(newWord, word[i:j]...)

			// Check if we can merge
			if j < len(word)-1 && word[j+1] == second {
				// Merge the pair
				newWord = append(newWord, first+second)
				i = j + 2
			} else {
				// Can't merge, just add the first token
				newWord = append(newWord, word[j])
				i = j + 1
			}
		}

		word = newWord
		if len(word) == 1 {
			break
		}
		pairs = getPairs(word)
	}

	result := strings.Join(word, " ")
	e.cache[token] = result
	return result
}

func (e *Encoder) Encode(text string) []int {
	if text == "" {
		return []int{}
	}

	var bpeTokens []int
	matches := e.pat.FindAllString(text, -1)

	for _, token := range matches {
		// Convert token bytes to unicode using byte encoder
		var encodedToken strings.Builder
		for _, b := range []byte(token) {
			encodedToken.WriteRune(e.byteEncoder[b])
		}

		// Apply BPE
		bpeResult := e.bpe(encodedToken.String())

		// Convert BPE tokens to integers
		if bpeResult != "" {
			bpeTokenStrs := strings.Fields(bpeResult)
			for _, bpeToken := range bpeTokenStrs {
				if tokenID, ok := e.encoder[bpeToken]; ok {
					bpeTokens = append(bpeTokens, tokenID)
				} else {
					// Handle unknown token - this shouldn't happen with proper tokenizer
					fmt.Printf("Warning: unknown token '%s'\n", bpeToken)
				}
			}
		}
	}

	return bpeTokens
}

func (e *Encoder) Decode(tokens []int) string {
	if len(tokens) == 0 {
		return ""
	}

	var text strings.Builder
	for _, token := range tokens {
		if tokenStr, ok := e.decoder[token]; ok {
			text.WriteString(tokenStr)
		} else {
			// Handle unknown token ID
			fmt.Printf("Warning: unknown token ID %d\n", token)
		}
	}

	// Convert unicode back to bytes
	var decodedBytes []byte
	for _, r := range text.String() {
		if b, ok := e.byteDecoder[r]; ok {
			decodedBytes = append(decodedBytes, b)
		} else {
			// This shouldn't happen with proper encoding
			fmt.Printf("Warning: unknown unicode point %v\n", r)
		}
	}

	return string(decodedBytes)
}

// Helper methods for testing and debugging

func (e *Encoder) VocabSize() int {
	return len(e.encoder)
}

func (e *Encoder) TokenToID(token string) (int, bool) {
	id, ok := e.encoder[token]
	return id, ok
}

func (e *Encoder) IDToToken(id int) (string, bool) {
	token, ok := e.decoder[id]
	return token, ok
}

// Example usage function
func ExampleUsage(tokenizerPath string) {
	encoder, err := NewEncoder(tokenizerPath)
	if err != nil {
		fmt.Printf("Error loading tokenizer: %v\n", err)
		return
	}

	text := "Hello, world! This is a test."
	tokens := encoder.Encode(text)
	decoded := encoder.Decode(tokens)

	fmt.Printf("Original: %s\n", text)
	fmt.Printf("Tokens: %v\n", tokens)
	fmt.Printf("Decoded: %s\n", decoded)
	fmt.Printf("Vocab size: %d\n", encoder.VocabSize())
}
