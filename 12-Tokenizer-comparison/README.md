# Tokenizer Comparison

Experimenting with tokenizers from different LLMs to understand how they handle text differently.

## What It Does

- Compare tokenization across: Llama 3.1, Phi-4, DeepSeek, Qwen
- Show character/word/token counts
- Demonstrate chat templates
- Encode and decode tokens

## Requirements

- Python 3.8+
- transformers
- python-dotenv
- huggingface-hub
- HuggingFace token (in .env as HF_TOKEN)

## Usage

```bash
python main.py