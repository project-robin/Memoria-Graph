from __future__ import annotations

import threading
from pathlib import Path

from qdrant_client import QdrantClient, models

from multimodal_search.config import COLLECTION_NAME, QDRANT_DIR, VECTOR_SIZE


class VectorStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.client = QdrantClient(path=str(QDRANT_DIR))
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        try:
            exists = self.client.collection_exists(COLLECTION_NAME)
        except AttributeError:
            collections = self.client.get_collections().collections
            exists = any(collection.name == COLLECTION_NAME for collection in collections)

        if exists:
            return

        self.client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE,
                distance=models.Distance.COSINE,
            ),
        )

    def upsert_image_vector(
        self,
        image_id: str,
        vector: list[float],
        payload: dict[str, object],
    ) -> None:
        with self._lock:
            self.client.upsert(
                collection_name=COLLECTION_NAME,
                wait=True,
                points=[
                    models.PointStruct(
                        id=image_id,
                        vector=vector,
                        payload=payload,
                    )
                ],
            )

    def query_similar_images(self, vector: list[float], limit: int) -> list[dict[str, object]]:
        with self._lock:
            try:
                response = self.client.query_points(
                    collection_name=COLLECTION_NAME,
                    query=vector,
                    limit=limit,
                    with_payload=True,
                )
                points = response.points
            except AttributeError:
                points = self.client.search(
                    collection_name=COLLECTION_NAME,
                    query_vector=vector,
                    limit=limit,
                    with_payload=True,
                )

        results: list[dict[str, object]] = []
        for point in points:
            payload = point.payload or {}
            source_path = str(payload.get("source_path", ""))
            if not source_path or not Path(source_path).exists():
                continue
            results.append(
                {
                    "image_id": str(point.id),
                    "source_path": source_path,
                    "thumb_path": str(payload.get("thumb_path", "")),
                    "caption": str(payload.get("caption", "")),
                    "mime_type": str(payload.get("mime_type", "image/jpeg")),
                    "vector_score": float(point.score),
                }
            )
        return results

    def count_indexed_images(self) -> int:
        return int(self.client.count(collection_name=COLLECTION_NAME, exact=True).count)

    def has_image_vector(self, image_id: str) -> bool:
        try:
            points = self.client.retrieve(
                collection_name=COLLECTION_NAME,
                ids=[image_id],
                with_payload=False,
                with_vectors=False,
            )
            return bool(points)
        except Exception:
            return False
