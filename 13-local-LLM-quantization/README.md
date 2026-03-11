# Local LLM Quantization

Run full LLMs locally on Apple Silicon using HuggingFace Transformers with quantization.

## What it does
- Load models with 4-bit quantization to reduce memory
- Run on MPS (Metal Performance Shaders)
- Experiment with Gemma and DeepSeek-R1 models
- Track memory footprint
- Stream outputs in real-time

## Requirements
- Python 3.8+
- transformers
- torch
- bitsandbytes
- accelerate
- huggingface-hub token

## Usage
python main.py