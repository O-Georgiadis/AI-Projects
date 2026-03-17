# Chroma RAG with Vector Visualization

Full RAG pipeline with vector store visualization.

## What it does
- Load documents from multiple folders
- Split into chunks with RecursiveCharacterTextSplitter
- Create embeddings with HuggingFace
- Store in Chroma vector database
- Visualize vectors in 2D/3D using t-SNE

## Requirements
- Python 3.8+
- langchain
- langchain-chroma
- langchain-huggingface
- tiktoken
- scikit-learn
- plotly

## Usage
```bash
python main.py