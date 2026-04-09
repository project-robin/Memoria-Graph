from __future__ import annotations

from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from multimodal_search.config import TOP_K_JUDGE, TOP_K_METADATA, TOP_K_RERANK, TOP_K_VECTOR
from multimodal_search.db import MetadataStore
from multimodal_search.embeddings import EmbeddingError, embed_image, embed_text
from multimodal_search.image_processing import decode_thumbnail_bytes, detect_face_crops, prepare_image_asset
from multimodal_search.llm import (
    LLMError,
    cheap_rerank_candidates,
    enrich_image_metadata,
    parse_search_query,
    strict_visual_judge,
)
from multimodal_search.storage import VectorStore
from multimodal_search.types import CandidateRecord, IngestionState, SearchState
from multimodal_search.db import _stable_image_uuid


def build_ingestion_graph(
    genai_client: object,
    metadata_store: MetadataStore,
    vector_store: VectorStore,
):
    def load_source_node(state: IngestionState) -> IngestionState:
        metadata_store.update_job_item_stage(state["job_item_id"], "load_source")
        prepared = prepare_image_asset(state["source_path"])
        return {
            **state,
            "mime_type": prepared["mime_type"],
            "source_hash": prepared["source_hash"],
            "original_width": prepared["width"],
            "original_height": prepared["height"],
            "thumbnail_path": prepared["thumbnail_path"],
            "thumbnail_bytes_b64": prepared["thumbnail_b64"],
            "album": prepared["album"],
            "exif_datetime": prepared.get("exif_datetime", ""),
            "gps_lat": prepared.get("gps_lat"),
            "gps_lng": prepared.get("gps_lng"),
        }

    def dedupe_node(state: IngestionState) -> IngestionState:
        metadata_store.update_job_item_stage(state["job_item_id"], "dedupe")
        existing = metadata_store.find_image_by_path(state["source_path"])
        canonical_image_id = _stable_image_uuid(state["source_hash"], state["source_path"])
        if existing and existing.get("source_hash") == state["source_hash"] and vector_store.has_image_vector(canonical_image_id):
            return {
                **state,
                "image_id": canonical_image_id,
                "skip_reason": "Already indexed with unchanged source hash.",
            }
        return {
            **state,
            "image_id": canonical_image_id,
        }

    def enrich_node(state: IngestionState) -> IngestionState:
        metadata_store.update_job_item_stage(state["job_item_id"], "enrich")
        if state.get("skip_reason"):
            return state
        if not state.get("deep_enrichment", True):
            return {
                **state,
                "caption": "",
                "ocr_text": "",
                "place_text": "",
                "people_summary": "",
                "tags": [],
            }
        thumbnail_bytes = decode_thumbnail_bytes(state["thumbnail_bytes_b64"])
        source_context = (
            f"album={state.get('album', '')}; "
            f"exif_datetime={state.get('exif_datetime', '')}; "
            f"source_path={state['source_path']}"
        )
        enriched = enrich_image_metadata(genai_client, thumbnail_bytes, source_context)
        return {
            **state,
            "caption": str(enriched.get("caption", "")),
            "ocr_text": str(enriched.get("ocr_text", "")),
            "place_text": str(enriched.get("place_text", "")),
            "people_summary": str(enriched.get("people_summary", "")),
            "tags": list(enriched.get("tags", [])),
        }

    def face_cluster_node(state: IngestionState) -> IngestionState:
        metadata_store.update_job_item_stage(state["job_item_id"], "face_cluster")
        if state.get("skip_reason"):
            return state
        provisional_image_id = state.get("image_id") or _stable_image_uuid(state["source_hash"], state["source_path"])
        detected_faces = detect_face_crops(state["thumbnail_path"], provisional_image_id)
        return {
            **state,
            "image_id": provisional_image_id,
            "face_count": len(detected_faces),
            "detected_faces": detected_faces,
        }

    def embed_node(state: IngestionState) -> IngestionState:
        metadata_store.update_job_item_stage(state["job_item_id"], "embed")
        if state.get("skip_reason"):
            return state
        thumbnail_bytes = decode_thumbnail_bytes(state["thumbnail_bytes_b64"])
        vector = embed_image(genai_client, thumbnail_bytes, "image/jpeg")
        return {
            **state,
            "vector": vector,
        }

    def persist_node(state: IngestionState) -> IngestionState:
        metadata_store.update_job_item_stage(state["job_item_id"], "persist")
        if state.get("skip_reason"):
            return state
        existing = metadata_store.find_image_by_path(state["source_path"])
        image_id = str(existing["image_id"]) if existing else (state.get("image_id") or _stable_image_uuid(state["source_hash"], state["source_path"]))
        image_record = {
            "image_id": image_id,
            "source_path": state["source_path"],
            "source_hash": state["source_hash"],
            "filename": Path(state["source_path"]).name,
            "album": state.get("album", ""),
            "mime_type": state["mime_type"],
            "width": state.get("original_width"),
            "height": state.get("original_height"),
            "thumb_path": state["thumbnail_path"],
            "status": "indexed",
            "exif_datetime": state.get("exif_datetime") or None,
            "exif_year": int(str(state["exif_datetime"])[:4]) if state.get("exif_datetime") else None,
            "exif_month": int(str(state["exif_datetime"])[5:7]) if state.get("exif_datetime") and len(str(state["exif_datetime"])) >= 7 else None,
            "gps_lat": state.get("gps_lat"),
            "gps_lng": state.get("gps_lng"),
            "place_text": state.get("place_text", ""),
            "caption": state.get("caption", ""),
            "ocr_text": state.get("ocr_text", ""),
            "people_summary": state.get("people_summary", ""),
            "people_text": "",
            "tags": state.get("tags", []),
            "face_count": state.get("face_count", 0),
            "metadata_json": {
                "face_clusters": state.get("face_clusters", []),
                "notices": state.get("notices", []),
            },
        }
        canonical_image_id = metadata_store.upsert_image(image_record)
        clusters = metadata_store.replace_image_faces(canonical_image_id, state.get("detected_faces", []))

        vector_store.upsert_image_vector(
            canonical_image_id,
            state["vector"],
            payload={
                "source_path": state["source_path"],
                "thumb_path": state["thumbnail_path"],
                "caption": state.get("caption", ""),
                "mime_type": state["mime_type"],
                "album": state.get("album", ""),
                "place_text": state.get("place_text", ""),
            },
        )
        return {
            **state,
            "image_id": canonical_image_id,
            "face_clusters": clusters,
        }

    graph = StateGraph(IngestionState)
    graph.add_node("load_source", load_source_node)
    graph.add_node("dedupe", dedupe_node)
    graph.add_node("enrich", enrich_node)
    graph.add_node("face_cluster", face_cluster_node)
    graph.add_node("embed", embed_node)
    graph.add_node("persist", persist_node)
    graph.add_edge(START, "load_source")
    graph.add_edge("load_source", "dedupe")
    graph.add_edge("dedupe", "enrich")
    graph.add_edge("enrich", "face_cluster")
    graph.add_edge("face_cluster", "embed")
    graph.add_edge("embed", "persist")
    graph.add_edge("persist", END)
    return graph.compile()


def build_search_graph(
    genai_client: object,
    metadata_store: MetadataStore,
    vector_store: VectorStore,
    checkpointer: object | None = None,
):
    def parse_query_node(state: SearchState) -> SearchState:
        recent_context = metadata_store.get_recent_turn_context(state["session_id"], limit=3)
        parsed = parse_search_query(genai_client, state["raw_query"], recent_context)
        filters = {
            "date_from": parsed.get("date_from", ""),
            "date_to": parsed.get("date_to", ""),
            "albums": list(parsed.get("albums", [])),
            "locations": list(parsed.get("locations", [])),
            "people": list(parsed.get("people", [])),
            "text_terms": list(parsed.get("text_terms", [])),
            "negative_terms": list(parsed.get("negative_terms", [])),
        }
        return {
            **state,
            "query": str(parsed.get("normalized_query", state["raw_query"])),
            "parsed_query": parsed,
            "filters": filters,
            "clarification_question": str(parsed.get("clarification_question", "")),
            "clarification_options": list(parsed.get("clarification_options", [])),
            "ambiguity_reason": str(parsed.get("ambiguity_reason", "")),
        }

    def clarification_node(state: SearchState) -> SearchState:
        question = state.get("clarification_question", "").strip()
        if not question or state.get("clarification_answer"):
            return state
        answer = interrupt(
            {
                "question": question,
                "options": state.get("clarification_options", []),
                "reason": state.get("ambiguity_reason", ""),
            }
        )
        return {
            **state,
            "clarification_answer": str(answer),
        }

    def rewrite_query_node(state: SearchState) -> SearchState:
        parsed = state.get("parsed_query", {})
        rewritten_parts = [str(parsed.get("retrieval_query") or state.get("query") or state["raw_query"])]
        if state.get("clarification_answer"):
            rewritten_parts.append(f"clarified as {state['clarification_answer']}")
        if state.get("filters", {}).get("people"):
            rewritten_parts.append("people=" + ", ".join(state["filters"]["people"]))
        if state.get("filters", {}).get("locations"):
            rewritten_parts.append("locations=" + ", ".join(state["filters"]["locations"]))
        if state.get("filters", {}).get("negative_terms"):
            rewritten_parts.append("avoid=" + ", ".join(state["filters"]["negative_terms"]))
        return {
            **state,
            "rewritten_query": " | ".join(part for part in rewritten_parts if part),
        }

    def retrieve_vector_node(state: SearchState) -> SearchState:
        errors = list(state.get("errors", []))
        try:
            query_vector = embed_text(genai_client, state["rewritten_query"])
            candidates = vector_store.query_similar_images(query_vector, TOP_K_VECTOR)
        except EmbeddingError as exc:
            errors.append(str(exc))
            candidates = []
        return {
            **state,
            "vector_candidates": candidates,
            "errors": errors,
        }

    def retrieve_metadata_node(state: SearchState) -> SearchState:
        candidates = metadata_store.search_metadata_candidates(
            state.get("filters", {}),
            state["rewritten_query"],
            TOP_K_METADATA,
        )
        return {
            **state,
            "metadata_candidates": candidates,
        }

    def fuse_candidates_node(state: SearchState) -> SearchState:
        merged: dict[str, CandidateRecord] = {}
        for item in state.get("vector_candidates", []):
            merged[item["image_id"]] = {
                **item,
                "metadata_score": 0.0,
                "fused_score": item.get("vector_score", 0.0) * 0.7,
            }
        for item in state.get("metadata_candidates", []):
            existing = merged.get(item["image_id"], {})
            merged[item["image_id"]] = {
                **existing,
                **item,
                "vector_score": existing.get("vector_score", 0.0),
                "metadata_score": item.get("metadata_score", 0.0),
                "fused_score": existing.get("fused_score", 0.0) + (item.get("metadata_score", 0.0) * 0.45),
            }
        fused = sorted(merged.values(), key=lambda item: float(item.get("fused_score", 0.0)), reverse=True)
        return {
            **state,
            "fused_candidates": fused[: max(TOP_K_VECTOR, TOP_K_METADATA)],
        }

    def cheap_rerank_node(state: SearchState) -> SearchState:
        fused = state.get("fused_candidates", [])[:TOP_K_RERANK]
        if not fused:
            return {
                **state,
                "reranked_candidates": [],
            }
        errors = list(state.get("errors", []))
        try:
            reranked = cheap_rerank_candidates(genai_client, state["raw_query"], fused)
            ordered_ids = list(reranked.get("ordered_ids", []))
            reasons = {
                item["image_id"]: item["reason"]
                for item in reranked.get("reasons", [])
                if item.get("image_id")
            }
            order_map = {image_id: index for index, image_id in enumerate(ordered_ids)}
            ranked = sorted(
                fused,
                key=lambda item: order_map.get(item["image_id"], 999),
            )
            for index, item in enumerate(ranked):
                item["rerank_score"] = float(len(ranked) - index)
                item["metadata_reason"] = reasons.get(item["image_id"], item.get("metadata_reason", "Cheap reranker"))
        except LLMError as exc:
            errors.append(str(exc))
            ranked = fused
        return {
            **state,
            "reranked_candidates": ranked,
            "errors": errors,
        }

    def judge_node(state: SearchState) -> SearchState:
        errors = list(state.get("errors", []))
        candidates = state.get("reranked_candidates", [])[:TOP_K_JUDGE]
        if not candidates:
            return {
                **state,
                "final_matches": [],
                "errors": errors,
            }
        try:
            judged = strict_visual_judge(genai_client, state["raw_query"], candidates, strict=True)
            reason_map = {
                item["image_id"]: item["reason"]
                for item in judged.get("reasons", [])
                if item.get("image_id")
            }
            matched_ids = set(judged.get("matched_ids", []))
            final_matches = []
            for candidate in candidates:
                if candidate["image_id"] in matched_ids:
                    candidate["judge_reason"] = reason_map.get(candidate["image_id"], "Strict visual match")
                    final_matches.append(candidate)
        except LLMError as exc:
            errors.append(str(exc))
            final_matches = []
        return {
            **state,
            "final_matches": final_matches,
            "errors": errors,
        }

    def fallback_node(state: SearchState) -> SearchState:
        if state.get("final_matches"):
            return state
        errors = list(state.get("errors", []))
        candidates = state.get("reranked_candidates", [])[:TOP_K_JUDGE]
        if not candidates:
            return {
                **state,
                "final_matches": [],
                "fallback_used": True,
                "errors": errors,
            }
        try:
            fallback = strict_visual_judge(genai_client, state["raw_query"], candidates, strict=False)
            reason_map = {
                item["image_id"]: item["reason"]
                for item in fallback.get("reasons", [])
                if item.get("image_id")
            }
            matched_ids = set(fallback.get("matched_ids", []))
            final_matches = []
            for candidate in candidates:
                if candidate["image_id"] in matched_ids:
                    candidate["judge_reason"] = reason_map.get(candidate["image_id"], "Fallback visual match")
                    final_matches.append(candidate)
        except LLMError as exc:
            errors.append(str(exc))
            final_matches = candidates[:3]
        return {
            **state,
            "final_matches": final_matches,
            "fallback_used": True,
            "errors": errors,
        }

    def finalize_node(state: SearchState) -> SearchState:
        final_matches = state.get("final_matches", [])
        if not final_matches:
            return state
        image_ids = [item["image_id"] for item in final_matches]
        image_map = {
            item["image_id"]: item
            for item in metadata_store.get_images_by_ids(image_ids)
        }
        hydrated: list[CandidateRecord] = []
        for item in final_matches:
            hydrated_item = {**image_map.get(item["image_id"], {}), **item}
            hydrated.append(hydrated_item)
        return {
            **state,
            "final_matches": hydrated,
        }

    def clarification_route(state: SearchState) -> str:
        if state.get("clarification_question") and not state.get("clarification_answer"):
            return "clarify"
        return "rewrite"

    def fallback_route(state: SearchState) -> str:
        return "fallback" if not state.get("final_matches") else "finalize"

    graph = StateGraph(SearchState)
    graph.add_node("parse_query", parse_query_node)
    graph.add_node("clarify", clarification_node)
    graph.add_node("rewrite", rewrite_query_node)
    graph.add_node("retrieve_vector", retrieve_vector_node)
    graph.add_node("retrieve_metadata", retrieve_metadata_node)
    graph.add_node("fuse", fuse_candidates_node)
    graph.add_node("cheap_rerank", cheap_rerank_node)
    graph.add_node("judge", judge_node)
    graph.add_node("fallback", fallback_node)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "parse_query")
    graph.add_conditional_edges(
        "parse_query",
        clarification_route,
        {
            "clarify": "clarify",
            "rewrite": "rewrite",
        },
    )
    graph.add_edge("clarify", "rewrite")
    graph.add_edge("rewrite", "retrieve_vector")
    graph.add_edge("retrieve_vector", "retrieve_metadata")
    graph.add_edge("retrieve_metadata", "fuse")
    graph.add_edge("fuse", "cheap_rerank")
    graph.add_edge("cheap_rerank", "judge")
    graph.add_conditional_edges(
        "judge",
        fallback_route,
        {
            "fallback": "fallback",
            "finalize": "finalize",
        },
    )
    graph.add_edge("fallback", "finalize")
    graph.add_edge("finalize", END)

    if checkpointer is not None:
        return graph.compile(checkpointer=checkpointer)
    return graph.compile()
