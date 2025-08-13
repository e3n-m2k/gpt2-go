package main

import (
	"flag"
	"fmt"
	"log"
)

func main() {

	tokenizerPath := flag.String("tokenizer", "./tokenizer.json", "Path to tokenizer.json")
	modelPath := flag.String("model", "./model.bin", "Path to model.bin")

	flag.Parse()

	fmt.Println("Tokenizer Path:", *tokenizerPath)
	fmt.Println("Model Path:", *modelPath)

	// NOTE: You need to download tokenizer.json for GPT-2 and place it in the same directory.
	// For example, from: https://huggingface.co/gpt2/blob/main/tokenizer.json
	encoder, err := NewEncoder(*tokenizerPath)
	if err != nil {
		log.Fatalf("Failed to create encoder: %v", err)
	}

	// model, hparams := CreateDummyGPT2Model(50257, 768, 4, 6)
	model, err := LoadGPT2ModelFromBinary(*modelPath)

	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}

	hparams := map[string]int{
		"n_head": 12,
		"n_ctx":  1024,
	}
	nTokensToGenerate := 20
	topk := 5
	prompt := "In physics, string theory is a theoretical framework in which the point-like particles of particle physics are replaced by one-dimensional objects called strings."

	inputIDs := encoder.Encode(prompt)
	if len(inputIDs)+nTokensToGenerate > hparams["n_ctx"] {
		log.Fatalf("Prompt is too long!")
	}

	// The generation will currently produce random garbage because the model is not loaded.
	outputIDs := model.Generate(inputIDs, hparams["n_head"], nTokensToGenerate, topk)
	outputText := encoder.Decode(outputIDs)

	fmt.Println("\n--- Generated Text ---")
	fmt.Println(outputText)
}
