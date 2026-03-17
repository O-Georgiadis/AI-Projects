# RAG Chatbot

Full Retrieval-Augmented Generation chatbot using Chroma + GPT, using the DB created in Project 18.

## What it does
- Load vector database with embeddings
- Retrieve relevant documents based on user query
- Build context from retrieved docs
- Generate answer using LLM with retrieved context
- Gradio chat interface

## Requirements
- Python 3.8+
- langchain
- langchain-chroma
- langchain-openai
- langchain-huggingface
- gradio
- python-dotenv

## Usage
```bash
python main.py

