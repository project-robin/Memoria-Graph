# Multimodal Photo Memory Search

Local-first Streamlit application for indexing and searching large personal photo libraries with LangGraph, Google multimodal models, Qdrant, and SQLite.

## What it does

- Persistent vector index with Qdrant on disk
- Persistent metadata/session/job storage in SQLite
- Folder indexing and upload indexing
- Background indexing worker with retries and resume
- Query planning with LangGraph
- Clarification interrupts for ambiguous searches
- Metadata fusion across captions, OCR, EXIF, albums, places, and face-cluster aliases
- Multi-stage ranking: vector retrieval -> metadata fusion -> cheap rerank -> strict multimodal judge -> fallback search
- Face-cluster alias management in the UI
- Search session memory and heuristic cost telemetry

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export GEMINI_API_KEY="your-api-key"
streamlit run app.py
```

The app also loads `GEMINI_API_KEY` from a local `.env.local` file if present.

## Local storage

- Uploaded files: `./temp_images/`
- Persistent app data: `./.local_data/`
- SQLite metadata DB: `./.local_data/metadata.sqlite3`
- LangGraph checkpoints: `./.local_data/langgraph_checkpoints.sqlite3`
- Qdrant on-disk store: `./.local_data/qdrant/`

## Notes

- The current implementation is optimized for local-first desktop usage.
- Large-scale indexing is still bounded by Gemini embedding quota and API latency.
- Python 3.12 or 3.13 is the safer runtime target today because some LangChain compatibility layers warn on Python 3.14.
