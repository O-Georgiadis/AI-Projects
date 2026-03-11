# Meeting Minutes Generator

Transcribe audio and generate meeting minutes using local LLMs from Hugging Face.

## What it does
- Transcribe audio with OpenAI Whisper (local)
- Generate meeting minutes from transcript
- Extract: summary, discussion points, takeaways, action items

## Requirements
- Python 3.8+
- transformers
- torch
- huggingface-hub

## Usage

```bash
python main.py