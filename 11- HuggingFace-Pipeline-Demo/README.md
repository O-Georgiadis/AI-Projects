# HuggingFace Pipeline Demo

A comprehensive demo of HuggingFace Transformers running on Apple Silicon (MPS).

## Features

- Sentiment Analysis (base + multilingual BERT)
- Named Entity Recognition
- Question Answering
- Text Summarization
- Translation (English → Greek)
- Zero-shot Classification
- Text Generation
- Text-to-Speech (Audio)
- Image Generation (SDXL Turbo)

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- Diffusers
- Datasets
- SoundFile
- HuggingFace Hub
- python-dotenv
- Gradio
- HuggingFace token in .env (HF_TOKEN)
- Apple Silicon Mac (MPS support)

## Usage

```bash
python main.py