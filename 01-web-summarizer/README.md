# Web Summarizer

A command-line tool that fetches web content and generates AI-powered summaries using a local Ollama LLM.

## Features

- Fetches and parses website content
- Generates concise summaries using Llama 3.2 via Ollama
- Filters out navigation, scripts, and irrelevant elements
- Clean markdown-formatted output

## Requirements

- Python 3.8+
- Ollama running locally on `http://localhost:11434`
- Required Python packages (see `requirements.txt`)

## Installation

1. Install Ollama from [ollama.ai](https://ollama.ai)
2. Pull the Llama model: `ollama pull llama3.2`
3. Start Ollama: `ollama serve`
4. Install dependencies: `pip install -r requirements.txt`

## Usage

```bash
python main.py
```

Enter a URL when prompted, and the tool will fetch the page and display an AI-generated summary.

## How It Works

- `scraper.py`: Fetches and parses HTML content using BeautifulSoup
- `main.py`: Uses the OpenAI-compatible API to query Ollama for summaries 