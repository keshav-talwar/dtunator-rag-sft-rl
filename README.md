# DTUnator v2: Adaptive RAG with SFT and RL

Second version of [DTUnator](https://github.com/Keshav-cse/DTUnator), the DTU doubt-solving
assistant. It moves the pipeline behind a **FastAPI** backend with a custom web UI, switches the
vector store to **ChromaDB**, and is the base for **supervised fine-tuning (SFT)** and
**reinforcement learning (RL)** on top of the RAG answers.

History: `DTUnator` (hackathon) → `rag-prototype` → **`dtunator-rag-sft-rl`** (this).

## Status

| Part | State |
|---|---|
| Ingestion (PDF + Tesseract OCR), cleaning, token-based chunking (200 tokens) | implemented (`main.py`) |
| ChromaDB store (`./chroma_db`), top-10 retrieval, cross-encoder rerank to the top 5 | implemented, but see Known issues |
| Refuses when nothing is retrieved ("I do not have enough verified DTU information…") | implemented, hook point for the reward model |
| FastAPI `POST /chat` with streaming response | implemented |
| Web UI (`index.html`, calls `http://localhost:8000/chat`) + Streamlit wrapper (`app.py`) | implemented |
| SFT on DTU Q&A pairs | **planned** (`test.py` is a Llama-3-8B-Instruct loading/generation check) |
| RL / reward model (RLHF) | **planned** |
| Hybrid dense + sparse retrieval (BGE-M3), bge-reranker | **being evaluated** (`Models_for_Rag.txt`) |

## Files

| File | What it is |
|---|---|
| `main.py` | Backend: ingestion → Chroma → rerank → OnDemand LLM (`predefined-openai-gpt4.1-nano`) → `/chat` |
| `index.html` | Frontend chat UI (streams from `/chat`) |
| `app.py` | Streamlit page that embeds `index.html` |
| `test.run.py` | Snapshot of the v1 Streamlit + Weaviate app, kept for comparison |
| `test.py` | Loads `meta-llama/Llama-3-8b-instruct` with transformers and generates, a first step toward SFT |
| `Models_for_Rag.txt` | Notes on embedding/reranker choices (BGE-M3 hybrid retrieval, bge-reranker, NV-Embed) |

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt streamlit
export ONDEMAND_API_KEY=...            # keep it in .env, which is gitignored
uvicorn main:app --port 8000           # backend
streamlit run app.py                   # UI (or open index.html directly)
```

## Known issues

- In `main.py`, the tokenizer and cross-encoder model names are wrapped in placeholder strings
  (`'''"sentence-transformers/all-MiniLM-L6-v2"Place holder'''`), so model loading fails until
  they're replaced with the plain names.
- `requirements.txt` is missing `streamlit` (needed for `app.py`), and `torch` (needed for `test.py`).
- `chromadb.Client(Settings(persist_directory=…))` doesn't persist with newer chromadb;
  `chromadb.PersistentClient(path=…)` does.
