package main

import (
	"encoding/binary"
	"fmt"
	"io"
	"log"
	"math"
	"math/rand"
	"os"
	"sort"

	"gonum.org/v1/gonum/mat"
)

// --- Model Parameters ---

type MLP struct {
	CFcW, CFcB, CProjW, CProjB *mat.Dense
}

type Attention struct {
	CAttnW, CAttnB, CProjW, CProjB *mat.Dense
}

type Block struct {
	MLP                    MLP
	Attn                   Attention
	Ln1G, Ln1B, Ln2G, Ln2B *mat.Dense
}

type GPT2Model struct {
	WTE, WPE   *mat.Dense
	Blocks     []Block
	LnFG, LnFB *mat.Dense
	LMHead     *mat.Dense
}

// splitQKV splits the QKV matrix into separate Q, K, V matrices
func splitQKV(qkv *mat.Dense, nHead int) (*mat.Dense, *mat.Dense, *mat.Dense) {
	rows, cols := qkv.Dims()
	dModel := cols / 3

	qData := make([]float64, rows*dModel)
	kData := make([]float64, rows*dModel)
	vData := make([]float64, rows*dModel)

	qkvData := qkv.RawMatrix().Data

	for i := 0; i < rows; i++ {
		for j := 0; j < dModel; j++ {
			qData[i*dModel+j] = qkvData[i*cols+j]          // Q
			kData[i*dModel+j] = qkvData[i*cols+dModel+j]   // K
			vData[i*dModel+j] = qkvData[i*cols+2*dModel+j] // V
		}
	}

	q := mat.NewDense(rows, dModel, qData)
	k := mat.NewDense(rows, dModel, kData)
	v := mat.NewDense(rows, dModel, vData)

	return q, k, v
}

// createCausalMask creates a causal (lower triangular) mask
func createCausalMask(seqLen int) *mat.Dense {
	mask := mat.NewDense(seqLen, seqLen, nil)

	for i := 0; i < seqLen; i++ {
		for j := 0; j < seqLen; j++ {
			if j > i {
				mask.Set(i, j, math.Inf(-1)) // Set to negative infinity for masked positions
			}
		}
	}

	return mask
}

func mha(x *mat.Dense, attn Attention, nHead int) *mat.Dense {
	rows, cols := x.Dims()

	// Linear projection to get Q, K, V
	qkv := linear(x, attn.CAttnW, attn.CAttnB)

	// Split into Q, K, V
	q, k, v := splitQKV(qkv, nHead)

	// Create causal mask
	mask := createCausalMask(rows)

	// For multi-head attention, we'd normally reshape and process each head
	// This is a simplified version that processes all heads together
	headDim := cols / nHead

	// Scale Q by sqrt(head_dim) instead of full dimension
	scale := math.Sqrt(float64(headDim))
	q.Scale(1/scale, q)

	// Apply attention
	attnOut := attention(q, k, v, mask)

	// Output projection
	out := linear(attnOut, attn.CProjW, attn.CProjB)

	return out
}

func transformerBlock(x *mat.Dense, block Block, nHead int) *mat.Dense {

	// First residual connection: x = x + mha(layer_norm(x, ln_1))
	ln1Out := layerNorm(x, block.Ln1G, block.Ln1B, 1e-5)
	mhaOut := mha(ln1Out, block.Attn, nHead)

	// Add residual connection
	var x1 mat.Dense
	x1.Add(x, mhaOut)

	// Second residual connection: x = x + ffn(layer_norm(x, ln_2))
	ln2Out := layerNorm(&x1, block.Ln2G, block.Ln2B, 1e-5)
	ffnOut := ffn(ln2Out, block.MLP)

	// Add residual connection
	var x2 mat.Dense
	x2.Add(&x1, ffnOut)

	return &x2
}

// getEmbedding extracts rows from embedding matrix based on indices
func getEmbedding(embMatrix *mat.Dense, indices []int) *mat.Dense {
	_, embDim := embMatrix.Dims()
	seqLen := len(indices)

	result := mat.NewDense(seqLen, embDim, nil)

	for i, idx := range indices {
		row := embMatrix.RawRowView(idx)
		result.SetRow(i, row)
	}

	return result
}

func (params *GPT2Model) Forward(inputs []int, nHead int) *mat.Dense {
	seqLen := len(inputs)
	if seqLen == 0 {
		return mat.NewDense(0, 0, nil)
	}

	// Get token embeddings: wte[inputs]
	tokEmb := getEmbedding(params.WTE, inputs)

	// Get position embeddings: wpe[range(len(inputs))]
	posIndices := make([]int, seqLen)
	for i := 0; i < seqLen; i++ {
		posIndices[i] = i
	}
	posEmb := getEmbedding(params.WPE, posIndices)

	// Add token and position embeddings
	var x mat.Dense
	x.Add(tokEmb, posEmb)

	// Pass through transformer blocks
	currentX := &x
	for _, block := range params.Blocks {
		currentX = transformerBlock(currentX, block, nHead)
	}

	// Final layer norm
	lnOut := layerNorm(currentX, params.LnFG, params.LnFB, 1e-5)

	// Language modeling head (projection to vocabulary)
	// This is typically wte.T (weight tying) or a separate linear layer
	var logits mat.Dense
	if params.LMHead != nil {
		logits.Mul(lnOut, params.LMHead)
	} else {
		// Use weight tying: multiply by transpose of token embeddings
		logits.Mul(lnOut, params.WTE.T())
	}

	return &logits
}

func (params *GPT2Model) Generate(inputs []int, nHead, nTokensToGenerate, topk int) []int {
	for i := 0; i < nTokensToGenerate; i++ {
		fmt.Printf("Generating token %d/%d\n", i+1, nTokensToGenerate)
		logits := params.Forward(inputs, nHead)

		// Get last logit row
		lastLogits := logits.RawRowView(logits.RawMatrix().Rows - 1)

		// Top-k sampling
		type kv struct {
			Key   int
			Value float64
		}
		var ss []kv
		for i, v := range lastLogits {
			ss = append(ss, kv{i, v})
		}
		sort.Slice(ss, func(i, j int) bool {
			return ss[i].Value > ss[j].Value
		})

		topKLogits := ss[:topk]

		// Random choice from top-k
		choice := topKLogits[rand.Intn(len(topKLogits))]
		nextID := choice.Key

		inputs = append(inputs, nextID)
	}
	return inputs[len(inputs)-nTokensToGenerate:]
}

func (params *GPT2Model) LoadModel() map[string]int {
	// THIS IS A PLACEHOLDER
	// In a real implementation, you would load pre-converted weights from a file (e.g., JSON, binary)
	// and populate the GPT2Model struct with `mat.Dense` matrices.
	log.Println("Loading placeholder model. Replace with actual model loading logic.")

	hparams := map[string]int{
		"n_head": 12,
		"n_ctx":  1024,
	}

	// Return empty model
	return hparams
}

func LoadGPT2ModelFromBinary(filePath string) (*GPT2Model, error) {
	// GPT-2 base model constants
	const (
		hiddenDim = 768
		nCtx      = 1024
		nVocab    = 50257
		nHead     = 12
		fcDim     = 3072
	)

	file, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open model file: %v", err)
	}
	defer file.Close()

	// Helper function to read float32 values
	readFloats := func(count int) ([]float64, error) {
		bytes := make([]byte, count*4) // 4 bytes per float32
		_, err := io.ReadFull(file, bytes)
		if err != nil {
			return nil, err
		}

		floats := make([]float64, count)
		for i := 0; i < count; i++ {
			bits := binary.LittleEndian.Uint32(bytes[i*4 : (i+1)*4])
			floats[i] = float64(math.Float32frombits(bits))
		}
		return floats, nil
	}

	// Load token embeddings (wte.weight): [n_vocab, hidden_dim]
	wteData, err := readFloats(nVocab * hiddenDim)
	if err != nil {
		return nil, fmt.Errorf("failed to read token embeddings: %v", err)
	}
	wte := mat.NewDense(nVocab, hiddenDim, wteData)

	// Load position embeddings (wpe.weight): [n_ctx, hidden_dim]
	wpeData, err := readFloats(nCtx * hiddenDim)
	if err != nil {
		return nil, fmt.Errorf("failed to read position embeddings: %v", err)
	}
	wpe := mat.NewDense(nCtx, hiddenDim, wpeData)

	// Load final layer norm (ln_f)
	lnFGData, err := readFloats(hiddenDim)
	if err != nil {
		return nil, fmt.Errorf("failed to read final layer norm weight: %v", err)
	}
	lnFG := mat.NewDense(1, hiddenDim, lnFGData)

	lnFBData, err := readFloats(hiddenDim)
	if err != nil {
		return nil, fmt.Errorf("failed to read final layer norm bias: %v", err)
	}
	lnFB := mat.NewDense(1, hiddenDim, lnFBData)

	// Load transformer blocks
	blocks := make([]Block, nHead)
	for i := 0; i < nHead; i++ {
		// Layer norm 1
		ln1GData, err := readFloats(hiddenDim)
		if err != nil {
			return nil, fmt.Errorf("failed to read layer norm 1 weight for block %d: %v", i, err)
		}
		ln1G := mat.NewDense(1, hiddenDim, ln1GData)

		ln1BData, err := readFloats(hiddenDim)
		if err != nil {
			return nil, fmt.Errorf("failed to read layer norm 1 bias for block %d: %v", i, err)
		}
		ln1B := mat.NewDense(1, hiddenDim, ln1BData)

		// Layer norm 2
		ln2GData, err := readFloats(hiddenDim)
		if err != nil {
			return nil, fmt.Errorf("failed to read layer norm 2 weight for block %d: %v", i, err)
		}
		ln2G := mat.NewDense(1, hiddenDim, ln2GData)

		ln2BData, err := readFloats(hiddenDim)
		if err != nil {
			return nil, fmt.Errorf("failed to read layer norm 2 bias for block %d: %v", i, err)
		}
		ln2B := mat.NewDense(1, hiddenDim, ln2BData)

		// Attention Q, K, V weights and biases
		qkvWData, err := readFloats(hiddenDim * hiddenDim * 3)
		if err != nil {
			return nil, fmt.Errorf("failed to read QKV weight for block %d: %v", i, err)
		}
		cAttnW := mat.NewDense(hiddenDim, hiddenDim*3, qkvWData)

		qkvBData, err := readFloats(hiddenDim * 3)
		if err != nil {
			return nil, fmt.Errorf("failed to read QKV bias for block %d: %v", i, err)
		}
		cAttnB := mat.NewDense(1, hiddenDim*3, qkvBData)

		// Attention output projection
		cProjWData, err := readFloats(hiddenDim * hiddenDim)
		if err != nil {
			return nil, fmt.Errorf("failed to read attention projection weight for block %d: %v", i, err)
		}
		cProjW := mat.NewDense(hiddenDim, hiddenDim, cProjWData)

		cProjBData, err := readFloats(hiddenDim)
		if err != nil {
			return nil, fmt.Errorf("failed to read attention projection bias for block %d: %v", i, err)
		}
		cProjB := mat.NewDense(1, hiddenDim, cProjBData)

		// MLP weights
		mlpFcWData, err := readFloats(hiddenDim * fcDim)
		if err != nil {
			return nil, fmt.Errorf("failed to read MLP fc weight for block %d: %v", i, err)
		}
		mlpFcW := mat.NewDense(hiddenDim, fcDim, mlpFcWData)

		mlpFcBData, err := readFloats(fcDim)
		if err != nil {
			return nil, fmt.Errorf("failed to read MLP fc bias for block %d: %v", i, err)
		}
		mlpFcB := mat.NewDense(1, fcDim, mlpFcBData)

		mlpProjWData, err := readFloats(fcDim * hiddenDim)
		if err != nil {
			return nil, fmt.Errorf("failed to read MLP projection weight for block %d: %v", i, err)
		}
		mlpProjW := mat.NewDense(fcDim, hiddenDim, mlpProjWData)

		mlpProjBData, err := readFloats(hiddenDim)
		if err != nil {
			return nil, fmt.Errorf("failed to read MLP projection bias for block %d: %v", i, err)
		}
		mlpProjB := mat.NewDense(1, hiddenDim, mlpProjBData)

		blocks[i] = Block{
			MLP: MLP{
				CFcW:   mlpFcW,
				CFcB:   mlpFcB,
				CProjW: mlpProjW,
				CProjB: mlpProjB,
			},
			Attn: Attention{
				CAttnW: cAttnW,
				CAttnB: cAttnB,
				CProjW: cProjW,
				CProjB: cProjB,
			},
			Ln1G: ln1G,
			Ln1B: ln1B,
			Ln2G: ln2G,
			Ln2B: ln2B,
		}
	}

	return &GPT2Model{
		WTE:    wte,
		WPE:    wpe,
		Blocks: blocks,
		LnFG:   lnFG,
		LnFB:   lnFB,
		LMHead: nil, // Use weight tying with WTE
	}, nil
}

// --- Example Usage Function ---

func CreateDummyGPT2Model(vocabSize, dModel, nLayer, nHead int) (GPT2Model, map[string]int) {
	// This is just for demonstration - in practice you'd load pretrained weights
	wte := mat.NewDense(vocabSize, dModel, nil)
	wpe := mat.NewDense(1024, dModel, nil) // max position embeddings

	blocks := make([]Block, nLayer)
	for i := 0; i < nLayer; i++ {
		blocks[i] = Block{
			MLP: MLP{
				CFcW:   mat.NewDense(dModel, dModel*4, nil),
				CFcB:   mat.NewDense(1, dModel*4, nil),
				CProjW: mat.NewDense(dModel*4, dModel, nil),
				CProjB: mat.NewDense(1, dModel, nil),
			},
			Attn: Attention{
				CAttnW: mat.NewDense(dModel, dModel*3, nil), // Q, K, V combined
				CAttnB: mat.NewDense(1, dModel*3, nil),
				CProjW: mat.NewDense(dModel, dModel, nil),
				CProjB: mat.NewDense(1, dModel, nil),
			},
			Ln1G: mat.NewDense(1, dModel, nil),
			Ln1B: mat.NewDense(1, dModel, nil),
			Ln2G: mat.NewDense(1, dModel, nil),
			Ln2B: mat.NewDense(1, dModel, nil),
		}
	}

	lnFG := mat.NewDense(1, dModel, nil)
	lnFB := mat.NewDense(1, dModel, nil)

	hparams := map[string]int{
		"n_head": nHead,
		"n_ctx":  1024,
	}

	model := GPT2Model{
		WTE:    wte,
		WPE:    wpe,
		Blocks: blocks,
		LnFG:   lnFG,
		LnFB:   lnFB,
		LMHead: nil, // Use weight tying
	}
	return model, hparams
}
