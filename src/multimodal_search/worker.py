from __future__ import annotations

import threading
import time

from multimodal_search.config import WORKER_BATCH_SIZE, WORKER_POLL_SECONDS


class IndexingWorker:
    def __init__(self, metadata_store: object, ingestion_graph: object) -> None:
        self.metadata_store = metadata_store
        self.ingestion_graph = ingestion_graph
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, name="indexing-worker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            batch = self.metadata_store.fetch_next_job_batch(WORKER_BATCH_SIZE)
            if not batch:
                time.sleep(WORKER_POLL_SECONDS)
                continue

            for item in batch:
                if self._stop_event.is_set():
                    return
                try:
                    result = self.ingestion_graph.invoke(
                        {
                            "job_id": item["job_id"],
                            "job_item_id": item["item_id"],
                            "source_path": item["source_path"],
                            "deep_enrichment": bool(item.get("options", {}).get("deep_enrichment", True)),
                            "errors": [],
                            "notices": [],
                        }
                    )
                    image_id = str(result.get("image_id", ""))
                    if not image_id:
                        raise RuntimeError("Ingestion completed without an image_id.")
                    self.metadata_store.mark_job_item_complete(item["item_id"], item["job_id"], image_id)
                except Exception as exc:
                    self.metadata_store.mark_job_item_failed(item["item_id"], item["job_id"], str(exc))
