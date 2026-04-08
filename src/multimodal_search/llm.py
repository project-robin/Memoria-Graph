from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from google.genai import types

from multimodal_search.config import ENRICHMENT_MODEL, JUDGE_MODEL, QUERY_MODEL, RERANK_MODEL


class LLMError(RuntimeError):
    """Raised when a structured LLM call fails."""


def generate_json(
    genai_client: object,
    model: str,
    prompt: str,
    schema: dict[str, object],
    parts: list[object] | None = None,
    temperature: float = 0.1,
) -> dict[str, object]:
    contents: list[object] = [prompt]
    if parts:
        contents.extend(parts)

    try:
        response = genai_client.models.generate_content(
            model=model,
            contents=contents,
            config={
                "response_mime_type": "application/json",
                "response_json_schema": schema,
                "temperature": temperature,
            },
        )
    except Exception as exc:
        raise LLMError(f"Model '{model}' failed: {exc}") from exc

    try:
        return json.loads(response.text)
    except Exception as exc:
        raise LLMError(f"Model '{model}' returned invalid JSON: {exc}") from exc


PARSE_QUERY_SCHEMA = {
    "type": "object",
    "properties": {
        "normalized_query": {"type": "string"},
        "retrieval_query": {"type": "string"},
        "date_from": {"type": "string"},
        "date_to": {"type": "string"},
        "albums": {"type": "array", "items": {"type": "string"}},
        "locations": {"type": "array", "items": {"type": "string"}},
        "people": {"type": "array", "items": {"type": "string"}},
        "text_terms": {"type": "array", "items": {"type": "string"}},
        "negative_terms": {"type": "array", "items": {"type": "string"}},
        "ambiguity_reason": {"type": "string"},
        "clarification_question": {"type": "string"},
        "clarification_options": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "normalized_query",
        "retrieval_query",
        "albums",
        "locations",
        "people",
        "text_terms",
        "negative_terms",
        "ambiguity_reason",
        "clarification_question",
        "clarification_options",
    ],
}

IMAGE_ENRICHMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "caption": {"type": "string"},
        "ocr_text": {"type": "string"},
        "place_text": {"type": "string"},
        "people_summary": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["caption", "ocr_text", "place_text", "people_summary", "tags"],
}

RERANK_SCHEMA = {
    "type": "object",
    "properties": {
        "ordered_ids": {"type": "array", "items": {"type": "string"}},
        "reasons": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "image_id": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["image_id", "reason"],
            },
        },
    },
    "required": ["ordered_ids", "reasons"],
}

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "matched_ids": {"type": "array", "items": {"type": "string"}},
        "reasons": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "image_id": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["image_id", "reason"],
            },
        },
    },
    "required": ["matched_ids", "reasons"],
}


def parse_search_query(genai_client: object, raw_query: str, session_context: list[dict[str, object]]) -> dict[str, object]:
    context_lines = []
    for turn in session_context:
        context_lines.append(
            f"- prior_query={turn['query_text']} status={turn['status']} clarification={turn.get('clarification', {})}"
        )
    context_block = "\n".join(context_lines) or "- no prior context"

    prompt = (
        "You are the query planner for a personal photo-memory search app.\n"
        "Extract useful structured filters and detect ambiguity. "
        "Only ask for clarification when the ambiguity would materially change retrieval.\n\n"
        f"Recent session context:\n{context_block}\n\n"
        f"User query: {raw_query}\n"
        "Normalize dates to ISO strings when possible. Use empty strings/arrays when absent."
    )
    return generate_json(genai_client, QUERY_MODEL, prompt, PARSE_QUERY_SCHEMA, temperature=0)


def enrich_image_metadata(
    genai_client: object,
    thumbnail_bytes: bytes,
    source_context: str,
) -> dict[str, object]:
    prompt = (
        "You are indexing a personal photo library.\n"
        "Return a concise caption, visible text (OCR), likely place/event cues, "
        "a short people summary, and 5-10 retrieval-friendly tags.\n"
        "If visible text is absent, return an empty string.\n"
        f"Source context: {source_context}"
    )
    parts = [types.Part.from_bytes(data=thumbnail_bytes, mime_type="image/jpeg")]
    return generate_json(genai_client, ENRICHMENT_MODEL, prompt, IMAGE_ENRICHMENT_SCHEMA, parts=parts, temperature=0.2)


def cheap_rerank_candidates(
    genai_client: object,
    query: str,
    candidates: list[dict[str, object]],
) -> dict[str, object]:
    manifest = []
    for item in candidates:
        manifest.append(
            {
                "image_id": item["image_id"],
                "caption": item.get("caption", ""),
                "ocr_text": item.get("ocr_text", ""),
                "place_text": item.get("place_text", ""),
                "album": item.get("album", ""),
                "people_text": item.get("people_text", ""),
                "exif_datetime": item.get("exif_datetime", ""),
                "tags": item.get("tags", []),
            }
        )
    prompt = (
        "You are a cheap text reranker for a photo search system.\n"
        "Given the user query and the candidate metadata summaries, rank the best candidates. "
        "Favor exact metadata matches, OCR, date/location/person constraints, and obvious object/event relevance.\n\n"
        f"User query: {query}\n"
        f"Candidates: {json.dumps(manifest)}"
    )
    return generate_json(genai_client, RERANK_MODEL, prompt, RERANK_SCHEMA, temperature=0)


def strict_visual_judge(
    genai_client: object,
    query: str,
    candidates: list[dict[str, object]],
    strict: bool = True,
) -> dict[str, object]:
    strictness = (
        "Apply strict matching. Reject any image that fails requested positive details or violates negative constraints."
        if strict
        else "Apply softer matching. Keep the best semantically relevant images even if some details are approximate."
    )
    manifest = "\n".join(
        f"- {item['image_id']} | path={item['source_path']} | caption={item.get('caption', '')}"
        for item in candidates
    )
    prompt = (
        "You are the final multimodal judge for a personal photo search system.\n"
        f"{strictness}\n"
        "Return only image IDs from the candidate list.\n\n"
        f"User query: {query}\n"
        f"Candidates:\n{manifest}"
    )
    parts: list[object] = []
    for item in candidates:
        image_path = Path(item["thumb_path"] or item["source_path"])
        parts.append(
            types.Part.from_bytes(
                data=image_path.read_bytes(),
                mime_type="image/jpeg" if image_path.suffix.lower() == ".jpg" else item.get("mime_type", "image/jpeg"),
            )
        )
    return generate_json(genai_client, JUDGE_MODEL, prompt=prompt, schema=JUDGE_SCHEMA, parts=parts, temperature=0)
