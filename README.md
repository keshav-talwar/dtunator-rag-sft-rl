
## Overview

This project extends the original DTUnator system by building an adaptive RAG architecture designed to improve response quality through fine-tuning and reinforcement learning strategies.

The system retrieves relevant information from a vector database and generates context-aware responses using a language model. Planned extensions include supervised fine-tuning (SFT) and reinforcement learning (RL) to optimize response quality and retrieval efficiency.

---

## Architecture

The system follows a modular retrieval and generation pipeline:

### 1. Document Ingestion & Preprocessing
- PDF scanning and text extraction  
- OCR-based text detection (Tesseract)  
- Noise reduction and cleaning  

### 2. Text Processing
- Semantic-aware text chunking  
- Embedding generation using transformer-based models  

### 3. Storage & Indexing
- Vector database storage (ChromaDB / FAISS)  

### 4. Retrieval Pipeline
- Similarity search retrieval  
- Cross-encoder reranking for semantic relevance refinement  

### 5. Generation Layer
- Context injection into LLM  
- Response generation  

### 6. Application Layer
- FastAPI backend  
- Streamlit interface  

### 7. Conversation Memory
- History retention architecture for contextual continuity  

---

## Previous Work

The original DTUnator hackathon prototype (BrainWave 2.0) is available here:  
[DTUnator](https://github.com/Keshav-cse/DTUnator)

This repository builds upon and extends that system with architectural improvements and adaptive optimization strategies.

---

## Planned Enhancements

- Supervised Fine-Tuning (SFT) on curated DTU-specific datasets  
- Reinforcement Learning (RL) for response optimization  
- Reward modeling for evaluation  
- Adaptive retrieval strategies  

---

## Tech Stack

- Python  
- python-dotenv  
- Pillow  
- pytesseract  
- PyMuPDF  
- ChromaDB  
- sentence-transformers  
- transformers  
- FastAPI  
- Uvicorn  
- scikit-learn  

---
## Current Status
- Baseline RAG implemented  
- SFT integration in progress  
- RL optimization pipeline under development  

> This project is actively under development. Features and documentation will be updated as progress is made.
