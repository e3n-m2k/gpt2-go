# Go parameters
GOPROGRAM := gpt2-go
GOFILES   := $(wildcard *.go)

# Python parameters
PYTHON       := python3
PIP          := pip3
REQUIREMENTS := python/requirements.txt

# Model parameters
MODEL_URL        := https://huggingface.co/gpt2
PY_CONVERTER     := python/gpt2_model_converter.py
PY_MODEL_BIN     := $(MODEL_DIR)/pytorch_model.bin
TOKENIZER_JSON   := $(MODEL_DIR)/tokenizer.json
GO_MODEL_WEIGHTS := $(MODEL_DIR)/model.bin

# Default target
.PHONY: all
all: build

# ====================================================================================
# Go Compilation and Execution
# ====================================================================================

.PHONY: build
build: $(GOPROGRAM)

$(GOPROGRAM): $(GOFILES)
	@echo "Building Go application..."
	go build -o $(GOPROGRAM) .

.PHONY: run
run:
	@echo "Running GPT-Go..."
	go run . --model=$(GO_MODEL_WEIGHTS) --tokenizer=$(TOKENIZER_JSON)

# ====================================================================================
# Project Setup
# ====================================================================================

.PHONY: setup
setup: setup-python download-model convert-model
	@echo "Setup complete. You can now run the application with 'make run'."

.PHONY: setup-python
setup-python: $(REQUIREMENTS)
	@echo "Installing Python dependencies..."
	$(PIP) install -r $(REQUIREMENTS)

download-model: $(MODEL_DIR)

$(MODEL_DIR):
	@echo "Downloading GPT-2 model from Hugging Face..."
	@if ! command -v git-lfs &> /dev/null; then \
		echo "git-lfs could not be found. Please install it first (see https://git-lfs.github.com/)."; \
		exit 1; \
	fi
	git lfs install
	git clone $(MODEL_URL) $(MODEL_DIR)

.PHONY: convert-model
convert-model: $(GO_MODEL_WEIGHTS)

$(GO_MODEL_WEIGHTS): $(PY_CONVERTER) $(PY_MODEL_BIN)
	@echo "Converting PyTorch model to Go-compatible binary format..."
	@echo "NOTE: This assumes you have a 'gpt2_model_converter.py' script in the 'python/' directory as mentioned in the README."
	$(PYTHON) $(PY_CONVERTER) $(PY_MODEL_BIN) $(GO_MODEL_WEIGHTS)

# ====================================================================================
# Cleaning
# ====================================================================================

.PHONY: clean
clean:
	@echo "Cleaning up..."
	rm -f $(GOPROGRAM)
