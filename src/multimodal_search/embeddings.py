from __future__ import annotations

from google.genai import types

from multimodal_search.config import EMBEDDING_MODEL, VECTOR_SIZE


class EmbeddingError(RuntimeError):
    """Raised when an embedding request fails or returns an invalid payload."""


def _extract_vector(response: object) -> list[float]:
    embeddings = getattr(response, "embeddings", None)
    if not embeddings:
        raise EmbeddingError("Embedding API returned no vectors.")

    first_embedding = embeddings[0]
    values = getattr(first_embedding, "values", None)
    if not values:
        raise EmbeddingError("Embedding API returned an empty vector.")

    return list(values)


def embed_text(genai_client: object, text: str) -> list[float]:
    try:
        response = genai_client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=VECTOR_SIZE,
            ),
        )
    except Exception as exc:
        raise EmbeddingError(f"Text embedding failed: {exc}") from exc

    return _extract_vector(response)


def embed_image(genai_client: object, image_bytes: bytes, mime_type: str) -> list[float]:
    try:
        response = genai_client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=[
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type=mime_type,
                )
            ],
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=VECTOR_SIZE,
            ),
        )
    except Exception as exc:
        raise EmbeddingError(f"Image embedding failed: {exc}") from exc

    return _extract_vector(response)
