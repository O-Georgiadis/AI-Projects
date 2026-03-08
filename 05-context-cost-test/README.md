# Context Cost Calculator

Compare LLM token usage and costs between short prompts and context-rich prompts using LiteLLM.

## What It Does

- Tests prompt costs with and without full context (Hamlet)
- Shows token breakdown (input/output/total)
- Displays cost in cents per request
- Uses LiteLLM for model-agnostic pricing

## Requirements

- Python 3.8+
- OpenAI API key
- LiteLLM
- openai

## Usage

```bash
python main.py