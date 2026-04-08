from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from langgraph.types import Command

from multimodal_search.image_processing import (
    ImageProcessingError,
    discover_image_files,
    resolve_mime_type,
    save_uploaded_source,
    validate_image_bytes,
)


def enqueue_uploaded_files(runtime: object, uploaded_files: list[object]) -> str:
    saved_paths: list[str] = []
    for uploaded_file in uploaded_files:
        image_bytes = uploaded_file.getvalue()
        validate_image_bytes(image_bytes)
        mime_type = resolve_mime_type(uploaded_file.name, getattr(uploaded_file, "type", None))
        destination = save_uploaded_source(uploaded_file.name, image_bytes, mime_type)
        saved_paths.append(str(destination))
    if not saved_paths:
        raise ImageProcessingError("No valid upload files were provided.")
    return runtime.metadata_store.create_index_job(
        source_label="Uploaded files",
        source_paths=saved_paths,
        mode="upload",
        options={"deep_enrichment": True},
    )


def enqueue_folder_index(runtime: object, folder_path: str) -> str:
    files = discover_image_files(folder_path)
    if not files:
        raise ImageProcessingError("No supported images found in the selected folder.")
    return runtime.metadata_store.create_index_job(
        source_label=str(Path(folder_path).expanduser().resolve()),
        source_paths=files,
        mode="folder",
        options={"deep_enrichment": True},
    )


def ensure_search_session(runtime: object, session_id: str | None = None) -> str:
    if session_id:
        existing = runtime.metadata_store.get_search_session(session_id)
        if existing:
            return session_id
    return runtime.metadata_store.create_search_session()


def _search_response_from_state(state: dict[str, object]) -> dict[str, object]:
    return {
        "status": "completed",
        "thread_id": state.get("thread_id", ""),
        "query": state.get("raw_query", ""),
        "clarification": None,
        "retrieved_images": state.get("fused_candidates", []),
        "final_images": state.get("final_matches", []),
        "fallback_used": bool(state.get("fallback_used", False)),
        "errors": state.get("errors", []),
        "notices": state.get("notices", []),
    }


def start_search(runtime: object, session_id: str, query: str) -> dict[str, object]:
    if not query.strip():
        raise RuntimeError("Enter a search query before running the graph.")
    thread_id = uuid4().hex
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = {
        "session_id": session_id,
        "thread_id": thread_id,
        "raw_query": query.strip(),
        "errors": [],
        "notices": [],
        "fallback_used": False,
    }
    result = runtime.search_graph.invoke(initial_state, config=config)
    interrupts = result.get("__interrupt__", [])
    if interrupts:
        payload = interrupts[0].value
        runtime.metadata_store.set_pending_clarification(
            session_id,
            thread_id,
            query.strip(),
            str(payload.get("question", "")),
        )
        response = {
            "status": "needs_clarification",
            "thread_id": thread_id,
            "query": query.strip(),
            "clarification": payload,
            "retrieved_images": [],
            "final_images": [],
            "fallback_used": False,
            "errors": result.get("errors", []),
            "notices": result.get("notices", []),
        }
        runtime.metadata_store.save_search_turn(
            session_id=session_id,
            query_text=query.strip(),
            status="needs_clarification",
            graph_state=result,
            results=response,
            clarification=payload,
        )
        return response

    runtime.metadata_store.set_pending_clarification(session_id, None, None, None)
    response = _search_response_from_state(result)
    runtime.metadata_store.save_search_turn(
        session_id=session_id,
        query_text=query.strip(),
        status="completed",
        graph_state=result,
        results=response,
    )
    return response


def resume_search(runtime: object, session_id: str, answer: str) -> dict[str, object]:
    if not answer.strip():
        raise RuntimeError("Enter a clarification answer before resuming the graph.")
    session = runtime.metadata_store.get_search_session(session_id)
    if not session or not session.get("pending_thread_id"):
        raise RuntimeError("No pending clarification is active for this session.")

    config = {"configurable": {"thread_id": session["pending_thread_id"]}}
    result = runtime.search_graph.invoke(Command(resume=answer), config=config)
    runtime.metadata_store.set_pending_clarification(session_id, None, None, None)
    response = _search_response_from_state(result)
    runtime.metadata_store.save_search_turn(
        session_id=session_id,
        query_text=str(session.get("pending_query") or ""),
        status="completed",
        graph_state=result,
        results=response,
        clarification={"answer": answer},
    )
    return response


def rename_face_cluster(runtime: object, cluster_id: str, alias_name: str) -> None:
    runtime.metadata_store.rename_face_cluster(cluster_id, alias_name)


def get_dashboard_state(runtime: object) -> dict[str, object]:
    vector_count = runtime.vector_store.count_indexed_images()
    return {
        "stats": runtime.metadata_store.get_library_stats(vector_count),
        "jobs": runtime.metadata_store.list_index_jobs(),
        "sessions": runtime.metadata_store.list_search_sessions(),
        "face_clusters": runtime.metadata_store.list_face_clusters(),
    }
