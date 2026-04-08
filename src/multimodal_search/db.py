from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator
from uuid import uuid4

from multimodal_search.config import (
    CHECKPOINT_DB_PATH,
    DB_PATH,
    EMBEDDING_COST_PER_M_TOKENS,
    ESTIMATED_IMAGE_TOKENS,
    ESTIMATED_JUDGE_OUTPUT_TOKENS,
    ESTIMATED_RERANK_PROMPT_TOKENS,
    FLASH_LITE_INPUT_COST_PER_M_TOKENS,
    FLASH_LITE_OUTPUT_COST_PER_M_TOKENS,
    INDEX_MAX_RETRIES,
)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _json_dumps(value: object) -> str:
    return json.dumps(value, sort_keys=True)


def _json_loads(value: str | None, fallback: object) -> object:
    if not value:
        return fallback
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return fallback


def _sanitize_fts_query(text: str) -> str:
    tokens = [token.strip().lower() for token in text.replace(",", " ").split() if len(token.strip()) > 1]
    if not tokens:
        return ""
    return " OR ".join(f'"{token}"' for token in tokens[:12])


def _hamming_distance(hex_a: str, hex_b: str) -> int:
    return (int(hex_a, 16) ^ int(hex_b, 16)).bit_count()


class MetadataStore:
    def __init__(self, db_path: Path = DB_PATH) -> None:
        self.db_path = db_path
        self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.db_path, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL;")
        connection.execute("PRAGMA foreign_keys=ON;")
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _initialize(self) -> None:
        CHECKPOINT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS images (
                    image_id TEXT PRIMARY KEY,
                    source_path TEXT UNIQUE NOT NULL,
                    source_hash TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    album TEXT,
                    mime_type TEXT NOT NULL,
                    width INTEGER,
                    height INTEGER,
                    thumb_path TEXT,
                    status TEXT NOT NULL,
                    exif_datetime TEXT,
                    exif_year INTEGER,
                    exif_month INTEGER,
                    gps_lat REAL,
                    gps_lng REAL,
                    place_text TEXT,
                    caption TEXT,
                    ocr_text TEXT,
                    people_summary TEXT,
                    people_text TEXT,
                    tags_json TEXT,
                    face_count INTEGER DEFAULT 0,
                    metadata_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE VIRTUAL TABLE IF NOT EXISTS image_fts USING fts5(
                    image_id UNINDEXED,
                    caption,
                    ocr_text,
                    tags,
                    album,
                    place_text,
                    people_text
                );

                CREATE TABLE IF NOT EXISTS face_clusters (
                    cluster_id TEXT PRIMARY KEY,
                    alias_name TEXT,
                    representative_hash TEXT NOT NULL,
                    sample_face_path TEXT,
                    face_count INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS image_faces (
                    face_id TEXT PRIMARY KEY,
                    image_id TEXT NOT NULL,
                    cluster_id TEXT NOT NULL,
                    face_path TEXT NOT NULL,
                    face_hash TEXT NOT NULL,
                    bbox_json TEXT,
                    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE,
                    FOREIGN KEY(cluster_id) REFERENCES face_clusters(cluster_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS index_jobs (
                    job_id TEXT PRIMARY KEY,
                    source_label TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    status TEXT NOT NULL,
                    options_json TEXT,
                    total_items INTEGER DEFAULT 0,
                    completed_items INTEGER DEFAULT 0,
                    failed_items INTEGER DEFAULT 0,
                    queued_items INTEGER DEFAULT 0,
                    running_items INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_error TEXT
                );

                CREATE TABLE IF NOT EXISTS index_job_items (
                    item_id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    status TEXT NOT NULL,
                    attempts INTEGER DEFAULT 0,
                    last_stage TEXT,
                    last_error TEXT,
                    image_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(job_id) REFERENCES index_jobs(job_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS search_sessions (
                    session_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    pending_thread_id TEXT,
                    pending_query TEXT,
                    pending_question TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS search_turns (
                    turn_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    query_text TEXT NOT NULL,
                    status TEXT NOT NULL,
                    clarification_json TEXT,
                    graph_state_json TEXT,
                    results_json TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES search_sessions(session_id) ON DELETE CASCADE
                );
                """
            )

            running_items = connection.execute(
                "UPDATE index_job_items SET status='queued', updated_at=? WHERE status='running'",
                (_now(),),
            )
            if running_items.rowcount:
                connection.execute(
                    "UPDATE index_jobs SET status='queued', updated_at=? WHERE status='running'",
                    (_now(),),
                )

    def create_search_session(self, title: str | None = None) -> str:
        session_id = uuid4().hex
        now = _now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO search_sessions (
                    session_id, title, created_at, updated_at
                ) VALUES (?, ?, ?, ?)
                """,
                (session_id, title or "Search session", now, now),
            )
        return session_id

    def list_search_sessions(self) -> list[dict[str, object]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT session_id, title, pending_thread_id, pending_query, pending_question, created_at, updated_at
                FROM search_sessions
                ORDER BY updated_at DESC
                LIMIT 25
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def get_search_session(self, session_id: str) -> dict[str, object] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT session_id, title, pending_thread_id, pending_query, pending_question, created_at, updated_at
                FROM search_sessions
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
        return dict(row) if row else None

    def set_pending_clarification(
        self,
        session_id: str,
        thread_id: str | None,
        query_text: str | None,
        question: str | None,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE search_sessions
                SET pending_thread_id = ?, pending_query = ?, pending_question = ?, updated_at = ?
                WHERE session_id = ?
                """,
                (thread_id, query_text, question, _now(), session_id),
            )

    def save_search_turn(
        self,
        session_id: str,
        query_text: str,
        status: str,
        graph_state: dict[str, object],
        results: dict[str, object],
        clarification: dict[str, object] | None = None,
    ) -> str:
        turn_id = uuid4().hex
        now = _now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO search_turns (
                    turn_id, session_id, query_text, status, clarification_json,
                    graph_state_json, results_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    turn_id,
                    session_id,
                    query_text,
                    status,
                    _json_dumps(clarification or {}),
                    _json_dumps(graph_state),
                    _json_dumps(results),
                    now,
                ),
            )
            connection.execute(
                """
                UPDATE search_sessions
                SET updated_at = ?, title = CASE
                    WHEN title = 'Search session' THEN ?
                    ELSE title
                END
                WHERE session_id = ?
                """,
                (now, query_text[:80], session_id),
            )
        return turn_id

    def get_recent_turn_context(self, session_id: str, limit: int = 3) -> list[dict[str, object]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT query_text, status, clarification_json, results_json, created_at
                FROM search_turns
                WHERE session_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
        context: list[dict[str, object]] = []
        for row in rows:
            item = dict(row)
            item["clarification"] = _json_loads(item.pop("clarification_json", None), {})
            item["results"] = _json_loads(item.pop("results_json", None), {})
            context.append(item)
        return context

    def create_index_job(
        self,
        source_label: str,
        source_paths: list[str],
        mode: str,
        options: dict[str, object] | None = None,
    ) -> str:
        job_id = uuid4().hex
        now = _now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO index_jobs (
                    job_id, source_label, mode, status, options_json, total_items,
                    completed_items, failed_items, queued_items, running_items,
                    created_at, updated_at
                ) VALUES (?, ?, ?, 'queued', ?, ?, 0, 0, ?, 0, ?, ?)
                """,
                (
                    job_id,
                    source_label,
                    mode,
                    _json_dumps(options or {}),
                    len(source_paths),
                    len(source_paths),
                    now,
                    now,
                ),
            )
            connection.executemany(
                """
                INSERT INTO index_job_items (
                    item_id, job_id, source_path, status, attempts,
                    created_at, updated_at
                ) VALUES (?, ?, ?, 'queued', 0, ?, ?)
                """,
                [
                    (uuid4().hex, job_id, source_path, now, now)
                    for source_path in source_paths
                ],
            )
        return job_id

    def list_index_jobs(self) -> list[dict[str, object]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT job_id, source_label, mode, status, total_items, completed_items,
                       failed_items, queued_items, running_items, created_at, updated_at, last_error
                FROM index_jobs
                ORDER BY updated_at DESC
                LIMIT 25
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def fetch_next_job_batch(self, batch_size: int) -> list[dict[str, object]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT item.item_id, item.job_id, item.source_path, item.status, item.attempts,
                       item.last_stage, job.options_json
                FROM index_job_items
                AS item
                JOIN index_jobs AS job ON job.job_id = item.job_id
                WHERE item.status = 'queued'
                ORDER BY item.created_at ASC
                LIMIT ?
                """,
                (batch_size,),
            ).fetchall()
            batch = [dict(row) for row in rows]
            for row in batch:
                claimed = connection.execute(
                    """
                    UPDATE index_job_items
                    SET status = 'running', updated_at = ?
                    WHERE item_id = ? AND status = 'queued'
                    """,
                    (_now(), row["item_id"]),
                )
                if claimed.rowcount:
                    connection.execute(
                        """
                        UPDATE index_jobs
                        SET status = 'running', updated_at = ?
                        WHERE job_id = ?
                        """,
                        (_now(), row["job_id"]),
                    )
                else:
                    row["status"] = "skipped"
                row["options"] = _json_loads(row.pop("options_json", None), {})
        return [row for row in batch if row.get("status") != "skipped"]

    def update_job_item_stage(self, item_id: str, stage: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE index_job_items
                SET last_stage = ?, updated_at = ?
                WHERE item_id = ?
                """,
                (stage, _now(), item_id),
            )

    def mark_job_item_complete(self, item_id: str, job_id: str, image_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE index_job_items
                SET status = 'completed', image_id = ?, updated_at = ?
                WHERE item_id = ?
                """,
                (image_id, _now(), item_id),
            )
        self._refresh_job_counts(job_id)

    def mark_job_item_failed(self, item_id: str, job_id: str, error: str) -> None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT attempts FROM index_job_items WHERE item_id = ?",
                (item_id,),
            ).fetchone()
            attempts = int(row["attempts"]) + 1 if row else 1
            next_status = "queued" if attempts < INDEX_MAX_RETRIES else "failed"
            connection.execute(
                """
                UPDATE index_job_items
                SET status = ?, attempts = ?, last_error = ?, updated_at = ?
                WHERE item_id = ?
                """,
                (next_status, attempts, error[:2000], _now(), item_id),
            )
            connection.execute(
                """
                UPDATE index_jobs
                SET last_error = ?, updated_at = ?
                WHERE job_id = ?
                """,
                (error[:2000], _now(), job_id),
            )
        self._refresh_job_counts(job_id)

    def _refresh_job_counts(self, job_id: str) -> None:
        with self._connect() as connection:
            stats = {
                row["status"]: row["count"]
                for row in connection.execute(
                    """
                    SELECT status, COUNT(*) AS count
                    FROM index_job_items
                    WHERE job_id = ?
                    GROUP BY status
                    """,
                    (job_id,),
                ).fetchall()
            }
            queued = int(stats.get("queued", 0))
            running = int(stats.get("running", 0))
            failed = int(stats.get("failed", 0))
            completed = int(stats.get("completed", 0))
            status = "completed"
            if running:
                status = "running"
            elif queued:
                status = "queued"
            elif failed and not completed:
                status = "failed"
            elif failed and completed:
                status = "completed_with_errors"

            connection.execute(
                """
                UPDATE index_jobs
                SET status = ?, completed_items = ?, failed_items = ?,
                    queued_items = ?, running_items = ?, updated_at = ?
                WHERE job_id = ?
                """,
                (status, completed, failed, queued, running, _now(), job_id),
            )

    def find_image_by_path(self, source_path: str) -> dict[str, object] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM images WHERE source_path = ?",
                (source_path,),
            ).fetchone()
        return dict(row) if row else None

    def upsert_image(self, record: dict[str, object]) -> str:
        now = _now()
        image_id = str(record.get("image_id") or uuid4().hex)
        tags = record.get("tags") or []
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO images (
                    image_id, source_path, source_hash, filename, album, mime_type,
                    width, height, thumb_path, status, exif_datetime, exif_year,
                    exif_month, gps_lat, gps_lng, place_text, caption, ocr_text,
                    people_summary, people_text, tags_json, face_count, metadata_json,
                    created_at, updated_at
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                ON CONFLICT(source_path) DO UPDATE SET
                    source_hash = excluded.source_hash,
                    filename = excluded.filename,
                    album = excluded.album,
                    mime_type = excluded.mime_type,
                    width = excluded.width,
                    height = excluded.height,
                    thumb_path = excluded.thumb_path,
                    status = excluded.status,
                    exif_datetime = excluded.exif_datetime,
                    exif_year = excluded.exif_year,
                    exif_month = excluded.exif_month,
                    gps_lat = excluded.gps_lat,
                    gps_lng = excluded.gps_lng,
                    place_text = excluded.place_text,
                    caption = excluded.caption,
                    ocr_text = excluded.ocr_text,
                    people_summary = excluded.people_summary,
                    people_text = excluded.people_text,
                    tags_json = excluded.tags_json,
                    face_count = excluded.face_count,
                    metadata_json = excluded.metadata_json,
                    updated_at = excluded.updated_at
                """,
                (
                    image_id,
                    record["source_path"],
                    record["source_hash"],
                    record["filename"],
                    record.get("album"),
                    record["mime_type"],
                    record.get("width"),
                    record.get("height"),
                    record.get("thumb_path"),
                    record.get("status", "indexed"),
                    record.get("exif_datetime"),
                    record.get("exif_year"),
                    record.get("exif_month"),
                    record.get("gps_lat"),
                    record.get("gps_lng"),
                    record.get("place_text"),
                    record.get("caption"),
                    record.get("ocr_text"),
                    record.get("people_summary"),
                    record.get("people_text", ""),
                    _json_dumps(tags),
                    record.get("face_count", 0),
                    _json_dumps(record.get("metadata_json", {})),
                    now,
                    now,
                ),
            )
            connection.execute("DELETE FROM image_fts WHERE image_id = ?", (image_id,))
            connection.execute(
                """
                INSERT INTO image_fts (image_id, caption, ocr_text, tags, album, place_text, people_text)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    image_id,
                    record.get("caption", ""),
                    record.get("ocr_text", ""),
                    " ".join(tags),
                    record.get("album", ""),
                    record.get("place_text", ""),
                    record.get("people_text", ""),
                ),
            )
        return image_id

    def replace_image_faces(self, image_id: str, detected_faces: list[dict[str, object]]) -> list[dict[str, object]]:
        clusters: list[dict[str, object]] = []
        with self._connect() as connection:
            prior_cluster_rows = connection.execute(
                "SELECT DISTINCT cluster_id FROM image_faces WHERE image_id = ?",
                (image_id,),
            ).fetchall()
            connection.execute("DELETE FROM image_faces WHERE image_id = ?", (image_id,))

            existing_clusters = [
                dict(row)
                for row in connection.execute(
                    """
                    SELECT cluster_id, alias_name, representative_hash, sample_face_path, face_count
                    FROM face_clusters
                    """
                ).fetchall()
            ]

            for face in detected_faces:
                cluster = None
                face_hash = str(face["face_hash"])
                for candidate in existing_clusters:
                    if _hamming_distance(face_hash, str(candidate["representative_hash"])) <= 10:
                        cluster = candidate
                        break
                if cluster is None:
                    cluster = {
                        "cluster_id": uuid4().hex,
                        "alias_name": None,
                        "representative_hash": face_hash,
                        "sample_face_path": face["face_path"],
                        "face_count": 0,
                    }
                    existing_clusters.append(cluster)
                    connection.execute(
                        """
                        INSERT INTO face_clusters (
                            cluster_id, alias_name, representative_hash, sample_face_path,
                            face_count, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, 0, ?, ?)
                        """,
                        (
                            cluster["cluster_id"],
                            cluster["alias_name"],
                            cluster["representative_hash"],
                            cluster["sample_face_path"],
                            _now(),
                            _now(),
                        ),
                    )

                connection.execute(
                    """
                    INSERT INTO image_faces (
                        face_id, image_id, cluster_id, face_path, face_hash, bbox_json
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        uuid4().hex,
                        image_id,
                        cluster["cluster_id"],
                        face["face_path"],
                        face_hash,
                        _json_dumps(face.get("bbox", {})),
                    ),
                )
                connection.execute(
                    """
                    UPDATE face_clusters
                    SET face_count = face_count + 1, updated_at = ?
                    WHERE cluster_id = ?
                    """,
                    (_now(), cluster["cluster_id"]),
                )
                clusters.append(
                    {
                        "cluster_id": cluster["cluster_id"],
                        "alias_name": cluster.get("alias_name") or "",
                        "face_path": face["face_path"],
                    }
                )

            self._refresh_image_people_text(connection, image_id)
            touched_clusters = {row["cluster_id"] for row in prior_cluster_rows}
            touched_clusters.update(cluster["cluster_id"] for cluster in clusters)
            for cluster_id in touched_clusters:
                connection.execute(
                    """
                    UPDATE face_clusters
                    SET face_count = (
                        SELECT COUNT(*)
                        FROM image_faces
                        WHERE cluster_id = ?
                    ), updated_at = ?
                    WHERE cluster_id = ?
                    """,
                    (cluster_id, _now(), cluster_id),
                )
        return clusters

    def _refresh_image_people_text(self, connection: sqlite3.Connection, image_id: str) -> None:
        rows = connection.execute(
            """
            SELECT fc.cluster_id, COALESCE(fc.alias_name, '') AS alias_name
            FROM image_faces imgf
            JOIN face_clusters fc ON fc.cluster_id = imgf.cluster_id
            WHERE imgf.image_id = ?
            """,
            (image_id,),
        ).fetchall()
        people_terms = [
            row["alias_name"] or f"person-cluster-{row['cluster_id'][:8]}"
            for row in rows
        ]
        connection.execute(
            """
            UPDATE images
            SET face_count = ?, people_text = ?, updated_at = ?
            WHERE image_id = ?
            """,
            (len(rows), " ".join(people_terms), _now(), image_id),
        )
        row = connection.execute("SELECT caption, ocr_text, tags_json, album, place_text, people_text FROM images WHERE image_id = ?", (image_id,)).fetchone()
        if row:
            connection.execute("DELETE FROM image_fts WHERE image_id = ?", (image_id,))
            connection.execute(
                """
                INSERT INTO image_fts (image_id, caption, ocr_text, tags, album, place_text, people_text)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    image_id,
                    row["caption"] or "",
                    row["ocr_text"] or "",
                    " ".join(_json_loads(row["tags_json"], [])),  # type: ignore[arg-type]
                    row["album"] or "",
                    row["place_text"] or "",
                    row["people_text"] or "",
                ),
            )

    def list_face_clusters(self, limit: int = 18) -> list[dict[str, object]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT cluster_id, alias_name, representative_hash, sample_face_path, face_count, updated_at
                FROM face_clusters
                ORDER BY face_count DESC, updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def rename_face_cluster(self, cluster_id: str, alias_name: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE face_clusters
                SET alias_name = ?, updated_at = ?
                WHERE cluster_id = ?
                """,
                (alias_name.strip(), _now(), cluster_id),
            )
            image_rows = connection.execute(
                "SELECT DISTINCT image_id FROM image_faces WHERE cluster_id = ?",
                (cluster_id,),
            ).fetchall()
            for row in image_rows:
                self._refresh_image_people_text(connection, row["image_id"])

    def get_candidate_summaries(self, image_ids: list[str]) -> list[dict[str, object]]:
        if not image_ids:
            return []
        placeholders = ", ".join("?" for _ in image_ids)
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT image_id, source_path, thumb_path, caption, ocr_text, place_text,
                       album, people_text, exif_datetime, mime_type, tags_json
                FROM images
                WHERE image_id IN ({placeholders})
                """,
                image_ids,
            ).fetchall()
        items = [dict(row) for row in rows]
        order = {image_id: index for index, image_id in enumerate(image_ids)}
        for item in items:
            item["tags"] = _json_loads(item.pop("tags_json", None), [])
        return sorted(items, key=lambda item: order.get(item["image_id"], 9999))

    def get_images_by_ids(self, image_ids: list[str]) -> list[dict[str, object]]:
        return self.get_candidate_summaries(image_ids)

    def search_metadata_candidates(
        self,
        filters: dict[str, object],
        text_query: str,
        limit: int,
    ) -> list[dict[str, object]]:
        clauses = ["1=1"]
        params: list[object] = []
        fts_query = _sanitize_fts_query(text_query)
        cte = ""
        join = ""
        score_expr = "0.0"

        if fts_query:
            cte = """
            WITH fts AS (
                SELECT image_id, MAX(0.001, -bm25(image_fts)) AS rank
                FROM image_fts
                WHERE image_fts MATCH ?
                LIMIT ?
            )
            """
            join = "LEFT JOIN fts ON fts.image_id = images.image_id"
            score_expr = "COALESCE(fts.rank, 0.0)"
            params.extend([fts_query, limit * 3])

        date_from = filters.get("date_from")
        date_to = filters.get("date_to")
        albums = [str(value).lower() for value in filters.get("albums", [])]
        locations = [str(value).lower() for value in filters.get("locations", [])]
        people = [str(value).lower() for value in filters.get("people", [])]

        if date_from:
            clauses.append("images.exif_datetime IS NOT NULL AND images.exif_datetime >= ?")
            params.append(date_from)
        if date_to:
            clauses.append("images.exif_datetime IS NOT NULL AND images.exif_datetime <= ?")
            params.append(date_to)
        for album in albums[:4]:
            clauses.append("LOWER(images.album) LIKE ?")
            params.append(f"%{album}%")
        for location in locations[:4]:
            clauses.append("LOWER(images.place_text) LIKE ?")
            params.append(f"%{location}%")
        for person in people[:4]:
            clauses.append("LOWER(images.people_text) LIKE ?")
            params.append(f"%{person}%")

        sql = f"""
        {cte}
        SELECT images.image_id, images.source_path, images.thumb_path, images.caption,
               images.ocr_text, images.place_text, images.album, images.people_text,
               images.exif_datetime, images.mime_type, images.tags_json,
               {score_expr} AS metadata_score
        FROM images
        {join}
        WHERE {' AND '.join(clauses)}
        ORDER BY metadata_score DESC, images.updated_at DESC
        LIMIT ?
        """
        params.append(limit)

        with self._connect() as connection:
            rows = connection.execute(sql, params).fetchall()
        results = [dict(row) for row in rows]
        for item in results:
            item["tags"] = _json_loads(item.pop("tags_json", None), [])
            item["metadata_reason"] = "Metadata/OCR/EXIF fusion candidate"
        return results

    def get_library_stats(self, vector_count: int) -> dict[str, object]:
        with self._connect() as connection:
            image_count = int(connection.execute("SELECT COUNT(*) FROM images").fetchone()[0])
            face_cluster_count = int(connection.execute("SELECT COUNT(*) FROM face_clusters").fetchone()[0])
            search_count = int(connection.execute("SELECT COUNT(*) FROM search_turns WHERE status = 'completed'").fetchone()[0])
            pending_jobs = int(connection.execute("SELECT COUNT(*) FROM index_jobs WHERE status IN ('queued', 'running')").fetchone()[0])

        estimated_embedding_cost = (image_count * ESTIMATED_IMAGE_TOKENS / 1_000_000) * EMBEDDING_COST_PER_M_TOKENS
        estimated_search_cost = search_count * (
            ((ESTIMATED_IMAGE_TOKENS * 5) + ESTIMATED_RERANK_PROMPT_TOKENS) / 1_000_000 * FLASH_LITE_INPUT_COST_PER_M_TOKENS
            + (ESTIMATED_JUDGE_OUTPUT_TOKENS / 1_000_000 * FLASH_LITE_OUTPUT_COST_PER_M_TOKENS)
        )
        return {
            "image_count": image_count,
            "vector_count": vector_count,
            "face_cluster_count": face_cluster_count,
            "search_count": search_count,
            "pending_jobs": pending_jobs,
            "estimated_embedding_cost_usd": round(estimated_embedding_cost, 4),
            "estimated_search_cost_usd": round(estimated_search_cost, 4),
        }
