# Project log: dtunator-rag-sft-rl
<!-- last-updated: 2026-11-22T09:46:19Z (maintained by /updaterepo) -->

## Current state
- Baseline RAG in `main.py`: PDF + OCR ingestion, 200-token chunks, ChromaDB, top-10 → cross-encoder top-5, OnDemand LLM, `POST /chat` streaming.
- UI: `index.html` (calls localhost:8000/chat), `app.py` (Streamlit wrapper). `test.run.py` = v1 snapshot, `test.py` = Llama-3-8B load test.
- SFT and RL: planned, not implemented. Retrieval choices being evaluated in `Models_for_Rag.txt` (BGE-M3 hybrid, bge-reranker, NV-Embed).

## Tried: don't repeat
| Approach | Last tried | Result | Why it failed / note |
|---|---|---|---|
| Loading models with the current names in `main.py` | 2026-11-22 | ❌ will fail | Names are wrapped in placeholder strings (`'''"..."Place holder'''`). Replace with plain names |
| `chromadb.Client(Settings(persist_directory=...))` for persistence | 2026-11-22 | ⚠️ doesn't persist on newer chromadb | Use `chromadb.PersistentClient(path=...)` |
| `pip install -r requirements.txt` then running app.py/test.py | 2026-11-22 | ❌ missing streamlit/torch | Add them to requirements.txt |

## What worked
- Old-laptop commit `fe7b4b7` (FastAPI app, web UI, test runner) fast-forwarded cleanly onto the GitHub history

## Decisions
- 2026-11-22: Honest status table in the README: implemented vs planned

## Open questions / next steps
- [ ] Fix the placeholder model names
- [ ] Fix requirements.txt
- [ ] Start SFT on DTU Q&A pairs

## History
### 2026-11-22: laptop merge + README
- Did: fast-forwarded the backup commit; rewrote README with status and known issues; pushed
- Result: bugs documented, not fixed
