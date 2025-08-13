# GPT-Go

A from-scratch implementation of the GPT-2 model in Go, designed for simplicity and educational purposes. This project aims to build a complete GPT-2 inference engine in pure Go.

This repository currently includes:
*   A fully functional **Byte-Pair Encoding (BPE) tokenizer** in Go.
*   A Python script to **convert Hugging Face's PyTorch models** into a simple binary format for the Go application.

## Project Status

**Under Development.** The tokenizer is complete and functional. The next major step is to implement the GPT-2 model architecture in Go to perform inference using the converted weights.

## Getting Started

Follow these steps to set up the project, convert a pre-trained model, and test the tokenizer.

### Prerequisites

*   [Go](https://go.dev/doc/install) (1.18 or later)
*   [Python 3](https://www.python.org/downloads/)
*   [Git LFS](https://git-lfs.github.com/) for downloading the model files.
*   Required Python libraries:
    ```bash
    pip install torch transformers
    ```

### 1. Download a GPT-2 Model and Tokenizer

You can download the original GPT-2 model files from the Hugging Face Hub.

```bash
# Make sure you have Git LFS installed
git lfs install

# Clone the gpt2 model repository
git clone https://huggingface.co/gpt2
```

This will create a `gpt2` directory containing `pytorch_model.bin`, `tokenizer.json`, and other configuration files.

### 2. Convert the Model Weights

The Go application will read model weights from a simple, flat binary file. Run the provided Python script to convert the downloaded `pytorch_model.bin` into this format.

```bash
python python/gpt2_model_converter.py gpt2/pytorch_model.bin gpt2/model_weights.bin
```

This will create a new file `gpt2/model_weights.bin` containing all the necessary weights for the model.

### 3. Run the Tokenizer

The Go tokenizer can be tested independently. To make it easy to run, you can place the `tokenizer.json` file in the root of this project directory.

```bash
# Copy the tokenizer file to the project root for easy access
cp gpt2/tokenizer.json .
```

Now, you can run the main Go program which demonstrates the tokenizer's `Encode` and `Decode` functionality.

```bash
go run .
```

You should see output similar to this:
```
--- Loading Tokenizer from tokenizer.json ---
Original: Hello, world! This is a test.
Tokens: [15496 11 995 0 220 632 318 257 1332 13]
Decoded: Hello, world! This is a test.
Vocab size: 50257
```

## How It Works

### Tokenizer (`tokenizer.go`)

This file contains a pure Go implementation of the GPT-2 tokenizer.
1.  **Pre-tokenization**: It uses a regular expression to split the input text into initial chunks. This is based on the GPT-2 paper's approach.
2.  **Byte-Pair Encoding (BPE)**: It loads the vocabulary (`vocab`) and merge rules (`merges`) from the `tokenizer.json` file. The BPE algorithm iteratively merges the most frequent character pairs in the vocabulary until the token is broken down into its final subword units.
3.  **Byte-level Mapping**: GPT-2 uses a clever trick to handle all possible byte values. The tokenizer maps every byte (0-255) to a specific Unicode character, ensuring that no token is ever "unknown". The `Decode` function reverses this process.

### Model Converter (`python/gpt2_model_converter.py`)

This script acts as a bridge between the complex PyTorch model format and a simple format for our Go application.
1.  It loads the GPT-2 model's `state_dict` using `torch.load`.
2.  It iterates through all the required weights (token embeddings, positional embeddings, attention weights, layer norm parameters, etc.) in a **predefined order**.
3.  For each weight tensor, it flattens the data into a 1D array of `float32` values.
4.  It writes these raw float values sequentially to a binary file.

This process results in a single, contiguous file of model parameters that can be easily read into a Go struct, mapping directly to the model's architecture without requiring a complex serialization library in Go.

## Project Roadmap

-   [ ] **Implement GPT-2 Model Architecture**: Build the Transformer blocks, including multi-head self-attention, layer normalization, and the position-wise feed-forward network in Go.
-   [ ] **Load Weights**: Write the Go code to read the `model_weights.bin` file into the model struct.
-   [ ] **Inference Loop**: Implement the forward pass to generate token probabilities.
-   [ ] **Text Generation**: Add sampling logic (e.g., top-k-sampling) to generate new text.
-   [ ] **CLI**: Create a simple command-line interface to run text generation.

## License

This project is licensed under the MIT License.
