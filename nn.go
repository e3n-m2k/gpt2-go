package main

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

// --- Neural Network Functions ---

func gelu(x *mat.Dense) *mat.Dense {
	rows, cols := x.Dims()
	data := make([]float64, rows*cols)
	for i, val := range x.RawMatrix().Data {
		data[i] = 0.5 * val * (1 + math.Tanh(math.Sqrt(2/math.Pi)*(val+0.044715*math.Pow(val, 3))))
	}
	return mat.NewDense(rows, cols, data)
}

func softmax(x *mat.Dense) *mat.Dense {
	rows, cols := x.Dims()
	out := mat.NewDense(rows, cols, nil)
	for r := range rows {
		row := x.RawRowView(r)
		maxVal := mat.Max(mat.NewVecDense(len(row), row))
		var sumExp float64
		expRow := make([]float64, len(row))
		for i, val := range row {
			expVal := math.Exp(val - maxVal)
			expRow[i] = expVal
			sumExp += expVal
		}
		for i, val := range expRow {
			expRow[i] = val / sumExp
		}
		out.SetRow(r, expRow)
	}
	return out
}

func layerNorm(x, g, b *mat.Dense, eps float64) *mat.Dense {
	rows, cols := x.Dims()
	out := mat.NewDense(rows, cols, nil)
	for r := range rows {
		row := x.RawRowView(r)
		mean := mat.Sum(mat.NewVecDense(len(row), row)) / float64(len(row))

		var variance float64
		for _, val := range row {
			variance += math.Pow(val-mean, 2)
		}
		variance /= float64(len(row))

		std := math.Sqrt(variance + eps)
		normRow := make([]float64, len(row))
		for i, val := range row {
			normRow[i] = (val - mean) / std
		}

		gRow := g.RawRowView(0)
		bRow := b.RawRowView(0)
		for i, val := range normRow {
			normRow[i] = val*gRow[i] + bRow[i]
		}
		out.SetRow(r, normRow)
	}
	return out
}

func linear(x, w, b *mat.Dense) *mat.Dense {
	var z mat.Dense
	z.Mul(x, w)

	// Broadcast bias b to each row of z
	rows, cols := z.Dims()
	bRows, bCols := b.Dims()

	// Handle different bias shapes
	if bRows == 1 && bCols == cols {
		// Standard case: bias is (1, output_features)
		biasRow := b.RawRowView(0)
		for r := range rows {
			for c := range cols {
				z.Set(r, c, z.At(r, c)+biasRow[c])
			}
		}
	} else if bRows == rows && bCols == cols {
		// Bias has same shape as output - direct addition
		z.Add(&z, b)
	} else if bRows == 1 && bCols == 1 {
		// Scalar bias - add to all elements
		scalar := b.At(0, 0)
		for r := range rows {
			for c := range cols {
				z.Set(r, c, z.At(r, c)+scalar)
			}
		}
	} else {
		// Invalid bias shape
		panic("bias shape is incompatible for broadcasting")
	}

	return &z
}

func ffn(x *mat.Dense, mlp MLP) *mat.Dense {
	x1 := linear(x, mlp.CFcW, mlp.CFcB)
	x2 := gelu(x1)
	x3 := linear(x2, mlp.CProjW, mlp.CProjB)
	return x3
}

func attention(q, k, v *mat.Dense, mask *mat.Dense) *mat.Dense {
	// rows, _ := q.Dims()
	_, colsK := k.Dims()

	var score mat.Dense
	score.Mul(q, k.T())

	scale := math.Sqrt(float64(colsK))
	score.Scale(1/scale, &score)

	score.Add(&score, mask)

	smax := softmax(&score)

	var out mat.Dense
	out.Mul(smax, v)
	return &out
}
