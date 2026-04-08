from __future__ import annotations

from dataclasses import dataclass

import streamlit as st
from langgraph.checkpoint.sqlite import SqliteSaver

from multimodal_search.clients import create_genai_client
from multimodal_search.config import CHECKPOINT_DB_PATH, ensure_app_dirs
from multimodal_search.db import MetadataStore
from multimodal_search.graph import build_ingestion_graph, build_search_graph
from multimodal_search.storage import VectorStore
from multimodal_search.worker import IndexingWorker


@dataclass
class AppRuntime:
    genai_client: object
    metadata_store: MetadataStore
    vector_store: VectorStore
    search_graph: object
    ingestion_graph: object
    indexing_worker: IndexingWorker
    checkpoint_cm: object
    checkpoint_store: object


def get_or_create_runtime() -> AppRuntime:
    ensure_app_dirs()
    if "app_runtime" not in st.session_state:
        genai_client = create_genai_client()
        metadata_store = MetadataStore()
        vector_store = VectorStore()
        checkpoint_cm = SqliteSaver.from_conn_string(str(CHECKPOINT_DB_PATH))
        checkpoint_store = checkpoint_cm.__enter__()
        ingestion_graph = build_ingestion_graph(genai_client, metadata_store, vector_store)
        search_graph = build_search_graph(
            genai_client,
            metadata_store,
            vector_store,
            checkpointer=checkpoint_store,
        )
        indexing_worker = IndexingWorker(metadata_store, ingestion_graph)
        indexing_worker.start()
        st.session_state.app_runtime = AppRuntime(
            genai_client=genai_client,
            metadata_store=metadata_store,
            vector_store=vector_store,
            search_graph=search_graph,
            ingestion_graph=ingestion_graph,
            indexing_worker=indexing_worker,
            checkpoint_cm=checkpoint_cm,
            checkpoint_store=checkpoint_store,
        )
    runtime: AppRuntime = st.session_state.app_runtime
    runtime.indexing_worker.start()
    return runtime
