from __future__ import annotations

from typing import Any, NotRequired, TypedDict


class QueryFilters(TypedDict, total=False):
    date_from: str
    date_to: str
    albums: list[str]
    locations: list[str]
    people: list[str]
    text_terms: list[str]
    negative_terms: list[str]


class CandidateRecord(TypedDict, total=False):
    image_id: str
    source_path: str
    thumb_path: str
    caption: str
    ocr_text: str
    place_text: str
    album: str
    people_text: str
    exif_datetime: str
    metadata_reason: str
    vector_score: float
    metadata_score: float
    fused_score: float
    rerank_score: float
    judge_reason: str
    mime_type: str
    tags: list[str]


class SearchState(TypedDict, total=False):
    session_id: str
    thread_id: str
    raw_query: str
    query: str
    parsed_query: dict[str, Any]
    filters: QueryFilters
    rewritten_query: str
    ambiguity_reason: str
    clarification_question: str
    clarification_options: list[str]
    clarification_answer: str
    vector_candidates: list[CandidateRecord]
    metadata_candidates: list[CandidateRecord]
    fused_candidates: list[CandidateRecord]
    reranked_candidates: list[CandidateRecord]
    final_matches: list[CandidateRecord]
    fallback_used: bool
    notices: list[str]
    errors: list[str]


class IngestionState(TypedDict, total=False):
    job_id: str
    job_item_id: str
    source_path: str
    source_label: str
    deep_enrichment: bool
    image_id: str
    mime_type: str
    source_hash: str
    original_width: int
    original_height: int
    thumbnail_path: str
    thumbnail_bytes_b64: str
    exif_datetime: str
    gps_lat: float
    gps_lng: float
    album: str
    caption: str
    ocr_text: str
    place_text: str
    tags: list[str]
    people_summary: str
    face_count: int
    face_clusters: list[dict[str, str]]
    detected_faces: list[dict[str, Any]]
    vector: list[float]
    skip_reason: str
    errors: list[str]
    notices: list[str]
