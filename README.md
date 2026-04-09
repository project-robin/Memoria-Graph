# Memoria Graph

Local-first multimodal photo memory search built with Streamlit, LangGraph, Google multimodal models, Qdrant, and SQLite.

Memoria Graph is designed for the "I know this photo exists, but I cannot find it" problem. It indexes a personal image library, enriches each image with searchable signals, and lets users retrieve photos with natural language such as:

- `family photo with the Toyota truck near the lake`
- `find the wedding pictures where my father is standing near the red car`
- `show me screenshots with the invoice number on them`
- `photos from Goa in 2021 with two people on the beach`

## Why this project exists

Traditional galleries are weak at memory-style search. Users remember fragments:

- an object
- a person
- a place
- a time range
- text inside the image
- a vague scene description

Memoria Graph combines vector retrieval, metadata fusion, clarification prompts, and multimodal reranking so the search flow behaves more like a memory assistant than a filename filter.

## Current capabilities

- Local-first Streamlit UI
- Persistent on-disk Qdrant vector store
- Persistent SQLite metadata, jobs, and search session storage
- Upload indexing and folder indexing
- Background indexing worker with retry/resume behavior
- LangGraph-powered search flow
- Query parsing and rewrite
- Clarification interrupt for ambiguous queries
- Metadata fusion across:
  - captions
  - OCR text
  - EXIF date
  - albums/folders
  - place hints
  - face-cluster aliases
- Multi-stage ranking:
  - vector retrieval
  - metadata retrieval
  - cheap rerank
  - strict multimodal judge
  - fallback search
- Face-cluster alias management
- Search session memory
- Heuristic cost telemetry

## Architecture

### Indexing pipeline

1. Source images are uploaded or discovered from a folder.
2. Images are normalized and thumbnail derivatives are created.
3. Metadata is extracted or generated:
   - EXIF
   - OCR-like text summary
   - captions/tags
   - place hints
   - face crops / cluster assignments
4. The image is embedded with Gemini multimodal embeddings.
5. Metadata is stored in SQLite and vectors are stored in Qdrant.
6. A background worker processes queued jobs and retries failures.

### Search pipeline

1. The query is parsed into structured intent.
2. The graph detects ambiguity and can pause for clarification.
3. The query is rewritten for retrieval.
4. Vector candidates and metadata candidates are fetched.
5. Candidates are fused and reranked.
6. A strict multimodal judge verifies the final shortlist.
7. Results and session state are persisted.

## Stack

- `streamlit`
- `langgraph`
- `langchain-core`
- `google-genai`
- `qdrant-client`
- `sqlite3`
- `pillow`
- `opencv-python-headless`

## Project layout

```text
app.py
src/multimodal_search/
  app_runtime.py
  clients.py
  config.py
  db.py
  embeddings.py
  graph.py
  image_processing.py
  llm.py
  services.py
  storage.py
  types.py
  worker.py
```

## Setup

### Prerequisites

- Python `3.12` or `3.13` recommended
- A Google API key with access to the configured Gemini models

### Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Configure

Either export the key:

```bash
export GEMINI_API_KEY="your-api-key"
```

Or create a local `.env.local` file:

```bash
GEMINI_API_KEY=your-api-key
```

### Run

```bash
streamlit run app.py
```

## Local data

Runtime-generated data is intentionally kept outside the tracked source tree:

- `./temp_images/` uploaded image copies
- `./.local_data/metadata.sqlite3` metadata, job state, search sessions
- `./.local_data/langgraph_checkpoints.sqlite3` LangGraph checkpoints
- `./.local_data/qdrant/` on-disk vector store
- `./.local_data/thumbnails/` generated thumbnails
- `./.local_data/faces/` face crops

## Troubleshooting

### Qdrant lock error

If you see:

`Storage folder .../.local_data/qdrant is already accessed by another instance of Qdrant client`

it means more than one Streamlit process is trying to open the same local Qdrant store. The app now uses a shared cached runtime to reduce this, but you still should not run multiple separate Streamlit processes against the same local data directory.

Fix:

1. Stop older `streamlit run app.py` processes.
2. Start one fresh app instance.

### Browser console warnings

If your browser console shows messages from domains like:

- `youtube.com`
- `studio.youtube.com`
- `googleads.g.doubleclick.net`

those are not Memoria Graph application errors. They come from other tabs, extensions, embedded browser surfaces, or browser-level CSP/CORS behavior. They are unrelated unless the message references your local app origin directly, such as `http://127.0.0.1:8504` or your own JavaScript bundle.

### Python 3.14 warning

Some LangChain compatibility layers still warn on Python `3.14`. The app can run, but `3.12` or `3.13` is the safer target right now.

### Embedding quota failures

Large indexing jobs can hit Gemini embedding quota limits. If you see `429 RESOURCE_EXHAUSTED`, reduce concurrency, wait for quota reset, or increase quota.

## Current limitations

- Search quality still depends heavily on model availability and quota
- OCR and face clustering are intentionally lightweight in the current local-first implementation
- The app is optimized for one-machine local usage, not multi-tenant SaaS deployment
- Cost telemetry is heuristic, not billing-export-accurate

## Roadmap

- Better OCR pipeline
- Stronger person identity linking
- Better event/time inference
- Batch indexing and more aggressive quota-aware scheduling
- Cleaner result explanations
- Optional cloud deployment mode

## Open source expectations

If you open source this project, do not commit:

- `.env.local`
- API keys
- `.local_data/`
- `.venv/`
- personal test images

This repository is set up to ignore those by default.
