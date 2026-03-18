# Production RAG Pipeline

Full retrieval-augmented generation system with separated ingestion and UI.

## Structure
- `ingest.py` - Creates vector database from knowledge base
- `answer.py` - RAG answering with chat history
- `app.py` - Gradio chat interface

## What it does
- Load markdown documents from knowledge base
- Split into overlapping chunks (500 chars, 200 overlap)
- Create OpenAI embeddings (text-embedding-3-large)
- Store in Chroma vector database
- Handle chat history context
- Display retrieved sources in UI

## Requirements
- Python 3.8+
- langchain
- langchain-chroma
- langchain-openai
- gradio
- python-dotenv

## Usage
# Step 1: Ingest documents
```bash
python ingest.py

# Step 2: Run chatbot
```bash
python app.py