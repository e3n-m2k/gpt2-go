# GPT-Go

A from-scratch implementation of the GPT-2 model in Go, designed for simplicity and educational purposes. This project provides a complete GPT-2 inference engine in pure Go.

This repository includes:
*   A fully functional **Byte-Pair Encoding (BPE) tokenizer** in Go.
*   A Go implementation of the **GPT-2 model architecture**.
*   A Python script to **convert Hugging Face's PyTorch models** into a simple binary format for the Go application.
*   A `Makefile` for easy setup and execution.

## Project Status

**Functional.** The project can load pre-trained GPT-2 models, tokenize input text, and generate new text.

## Dependencies

### Go
- `gonum.org/v1/gonum`

### Python
- `numpy`
- `torch`
- `tqdm`
- `regex`

## Setup

The `Makefile` provides a convenient way to set up the project, including installing Python dependencies, downloading the GPT-2 model, and converting the model weights.

```bash
make setup
```

This command will:
1.  Install the required Python packages.
2.  Download the GPT-2 model from Hugging Face.
3.  Convert the PyTorch model to a Go-compatible binary format.

## Usage

To generate text, use the `run` target in the `Makefile`:

```bash
make run
```

You can also run the program directly and specify the model and tokenizer paths:

```bash
go run . --model=<path_to_model.bin> --tokenizer=<path_to_tokenizer.json>
```

## How It Works

### Tokenizer (`tokenizer.go`)

This file contains a pure Go implementation of the GPT-2 tokenizer.
1.  **Pre-tokenization**: It uses a regular expression to split the input text into initial chunks.
2.  **Byte-Pair Encoding (BPE)**: It loads the vocabulary and merge rules from the `tokenizer.json` file and uses the BPE algorithm to tokenize the input.
3.  **Byte-level Mapping**: It handles all possible byte values by mapping them to Unicode characters.

### Model Converter (`python/gpt2_model_converter.py`)

This script converts the PyTorch model into a simple binary format that the Go application can read.
1.  It loads the GPT-2 model's `state_dict`.
2.  It iterates through the model's weights in a predefined order.
3.  It writes the raw float values to a binary file.

### GPT-2 Model in Go (`gpt.go` and `nn.go`)

The GPT-2 model is implemented in Go:
- **`nn.go`**: Contains the basic neural network layers and activation functions, such as `linear`, `softmax`, `gelu`, and `layerNorm`.
- **`gpt.go`**: Implements the GPT-2 model architecture, including the transformer blocks, multi-head self-attention, and the final language model head. It also includes the logic for loading the converted model weights and generating text.

## Project Roadmap

-   [x] **Implement GPT-2 Model Architecture**: Build the Transformer blocks, including multi-head self-attention, layer normalization, and the position-wise feed-forward network in Go.
-   [x] **Load Weights**: Write the Go code to read the `model.bin` file into the model struct.
-   [x] **Inference Loop**: Implement the forward pass to generate token probabilities.
-   [x] **Text Generation**: Add sampling logic (e.g., top-k-sampling) to generate new text.
-   [x] **CLI**: Create a simple command-line interface to run text generation.
-   [ ] **Improve Performance**: Optimize the Go implementation for better performance.
-   [ ] **Add More Sampling Methods**: Implement other sampling methods like nucleus sampling.

## License

This project is licensed under the MIT License.