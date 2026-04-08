from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from multimodal_search.app_runtime import get_or_create_runtime
from multimodal_search.config import APP_SUBTITLE, APP_TITLE, LOCAL_DATA_DIR
from multimodal_search.image_processing import ImageProcessingError
from multimodal_search.services import (
    enqueue_folder_index,
    enqueue_uploaded_files,
    ensure_search_session,
    get_dashboard_state,
    rename_face_cluster,
    resume_search,
    start_search,
)


def render_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=Manrope:wght@400;500;600;700&display=swap');

        :root {
            --paper: #f7f1e7;
            --ink: #1f1a17;
            --accent: #bf4b2c;
            --accent-soft: rgba(191, 75, 44, 0.12);
            --line: rgba(31, 26, 23, 0.14);
            --card: rgba(255, 251, 246, 0.88);
            --shadow: 0 22px 48px rgba(38, 20, 14, 0.12);
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(191, 75, 44, 0.16), transparent 28%),
                radial-gradient(circle at bottom right, rgba(31, 26, 23, 0.08), transparent 32%),
                linear-gradient(135deg, #f8f4ee 0%, #efe5d6 48%, #f7f1e7 100%);
            color: var(--ink);
            font-family: "Manrope", sans-serif;
        }

        .block-container {
            padding-top: 2.2rem;
            padding-bottom: 3rem;
            max-width: 1220px;
        }

        h1, h2, h3 {
            font-family: "DM Serif Display", serif;
            letter-spacing: -0.03em;
        }

        .hero-panel, .section-panel, .mini-card {
            background: rgba(255, 250, 245, 0.8);
            border: 1px solid rgba(31, 26, 23, 0.14);
            box-shadow: 0 22px 48px rgba(38, 20, 14, 0.10);
            border-radius: 24px;
        }

        .hero-panel {
            padding: 1.6rem 1.8rem;
            margin-bottom: 1rem;
        }

        .section-panel {
            padding: 1rem 1.2rem 1.2rem;
        }

        .mini-card {
            padding: 0.9rem 1rem;
            min-height: 108px;
        }

        .eyebrow {
            display: inline-block;
            font-size: 0.74rem;
            text-transform: uppercase;
            letter-spacing: 0.16em;
            font-weight: 800;
            color: var(--accent);
            margin-bottom: 0.55rem;
        }

        .hero-copy {
            max-width: 52rem;
            line-height: 1.6;
            color: rgba(31, 26, 23, 0.78);
            margin: 0;
        }

        .metric-label {
            font-size: 0.74rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: rgba(31, 26, 23, 0.55);
            margin-bottom: 0.4rem;
            font-weight: 800;
        }

        .metric-value {
            font-size: 1.22rem;
            font-weight: 800;
        }

        .stButton > button {
            border-radius: 999px;
            border: 1px solid rgba(191, 75, 44, 0.22);
            background: linear-gradient(135deg, #bf4b2c, #8c351f);
            color: white;
            font-weight: 700;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header(stats: dict[str, object]) -> None:
    st.markdown(
        f"""
        <div class="hero-panel">
            <div class="eyebrow">LangGraph Photo Memory Search</div>
            <h1>{APP_TITLE}</h1>
            <p class="hero-copy">{APP_SUBTITLE}</p>
            <p class="hero-copy">Persistent data directory: {LOCAL_DATA_DIR}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    columns = st.columns(5)
    metrics = [
        ("Indexed images", stats["image_count"]),
        ("Vectors", stats["vector_count"]),
        ("Face clusters", stats["face_cluster_count"]),
        ("Search turns", stats["search_count"]),
        ("Pending jobs", stats["pending_jobs"]),
    ]
    for column, (label, value) in zip(columns, metrics):
        with column:
            st.markdown(
                f"""
                <div class="mini-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_search_results(response: dict[str, object] | None) -> None:
    if not response:
        return

    if response.get("errors"):
        for error in response["errors"]:
            st.error(error)

    if response.get("status") == "needs_clarification":
        payload = response["clarification"] or {}
        st.warning(payload.get("reason") or "The graph needs clarification before it can continue.")
        if payload.get("options"):
            st.caption("Suggested options: " + ", ".join(payload["options"]))
        return

    if response.get("notices"):
        for notice in response["notices"]:
            st.info(notice)

    if response.get("fallback_used"):
        st.info("The strict judge returned no perfect matches, so the graph retried with a softer fallback path.")

    retrieved = response.get("retrieved_images", [])
    if retrieved:
        with st.expander("Candidate set after retrieval fusion", expanded=False):
            for item in retrieved[:12]:
                st.write(
                    f"{item.get('image_id')} | fused={item.get('fused_score', 0):.3f} | "
                    f"{item.get('caption', '')}"
                )

    final_images = response.get("final_images", [])
    if not final_images:
        st.info("No matching images returned yet.")
        return

    st.markdown("### Final matches")
    columns = st.columns(3)
    for index, item in enumerate(final_images):
        with columns[index % 3]:
            image_path = item.get("source_path")
            caption = item.get("judge_reason") or item.get("caption") or Path(image_path).name
            st.image(image_path, caption=caption, use_container_width=True)
            if item.get("caption"):
                st.caption(item["caption"])
            if item.get("ocr_text"):
                st.caption(f"OCR: {item['ocr_text']}")
            meta = []
            if item.get("place_text"):
                meta.append(f"place={item['place_text']}")
            if item.get("album"):
                meta.append(f"album={item['album']}")
            if item.get("people_text"):
                meta.append(f"people={item['people_text']}")
            if item.get("exif_datetime"):
                meta.append(f"date={item['exif_datetime']}")
            if meta:
                st.caption(" | ".join(meta))


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="🖼️", layout="wide")
    render_styles()

    try:
        runtime = get_or_create_runtime()
    except Exception as exc:
        st.error(f"Initialization failed: {exc}")
        st.stop()

    dashboard = get_dashboard_state(runtime)
    render_header(dashboard["stats"])

    sessions = dashboard["sessions"]
    if "active_session_id" not in st.session_state:
        st.session_state.active_session_id = ensure_search_session(
            runtime,
            sessions[0]["session_id"] if sessions else None,
        )
    active_session_id = ensure_search_session(runtime, st.session_state.active_session_id)

    with st.sidebar:
        st.header("Indexing")
        uploads = st.file_uploader(
            "Upload images",
            accept_multiple_files=True,
            type=["jpg", "jpeg", "png", "webp"],
        )
        if st.button("Queue Uploaded Images", use_container_width=True):
            try:
                job_id = enqueue_uploaded_files(runtime, uploads or [])
                st.success(f"Queued upload job {job_id[:8]}.")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

        folder_path = st.text_input(
            "Or index a folder",
            placeholder="/Users/me/Pictures/Family",
        )
        if st.button("Queue Folder Index", use_container_width=True):
            try:
                job_id = enqueue_folder_index(runtime, folder_path)
                st.success(f"Queued folder job {job_id[:8]}.")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

        st.divider()
        session_options = {
            f"{item['title']} ({item['session_id'][:8]})": item["session_id"]
            for item in dashboard["sessions"]
        }
        if session_options:
            label = st.selectbox("Search session", list(session_options.keys()))
            st.session_state.active_session_id = session_options[label]
            active_session_id = st.session_state.active_session_id
        if st.button("New Session", use_container_width=True):
            st.session_state.active_session_id = ensure_search_session(runtime, None)
            st.rerun()

    session_record = runtime.metadata_store.get_search_session(active_session_id)
    last_response = st.session_state.get("last_search_response")

    search_tab, jobs_tab, people_tab = st.tabs(["Search", "Jobs & Cost", "People Clusters"])

    with search_tab:
        st.markdown('<div class="section-panel">', unsafe_allow_html=True)
        st.subheader("Search the photo library")
        with st.form("search-form", clear_on_submit=False):
            query = st.text_input(
                "Describe what you want to find",
                placeholder="family photo with the Toyota truck near the lake in 2021",
            )
            submitted = st.form_submit_button("Run LangGraph Search")
        if submitted:
            try:
                response = start_search(runtime, active_session_id, query)
                st.session_state.last_search_response = response
                last_response = response
            except Exception as exc:
                st.error(str(exc))

        if session_record and session_record.get("pending_thread_id"):
            st.markdown("### Clarification required")
            st.write(session_record.get("pending_question") or "The graph is waiting for clarification.")
            with st.form("clarify-form", clear_on_submit=True):
                answer = st.text_input("Clarification answer", placeholder="Truck")
                clarify = st.form_submit_button("Resume Search")
            if clarify:
                try:
                    response = resume_search(runtime, active_session_id, answer)
                    st.session_state.last_search_response = response
                    last_response = response
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))

        render_search_results(last_response)

        recent_turns = runtime.metadata_store.get_recent_turn_context(active_session_id, limit=5)
        if recent_turns:
            with st.expander("Recent session memory", expanded=False):
                for turn in recent_turns:
                    st.write(f"{turn['created_at']} | {turn['status']} | {turn['query_text']}")
        st.markdown("</div>", unsafe_allow_html=True)

    with jobs_tab:
        st.markdown('<div class="section-panel">', unsafe_allow_html=True)
        st.subheader("Background indexing and cost telemetry")
        stats = dashboard["stats"]
        cost_cols = st.columns(2)
        with cost_cols[0]:
            st.metric("Estimated embedding spend", f"${stats['estimated_embedding_cost_usd']}")
        with cost_cols[1]:
            st.metric("Estimated search spend", f"${stats['estimated_search_cost_usd']}")
        st.caption(
            "These are heuristic estimates based on indexed image count and completed search turns, "
            "not direct billing exports."
        )

        if dashboard["jobs"]:
            for job in dashboard["jobs"]:
                progress = 0.0
                total = max(int(job["total_items"]), 1)
                progress = int(job["completed_items"]) / total
                st.write(
                    f"{job['source_label']} | status={job['status']} | "
                    f"completed={job['completed_items']}/{job['total_items']} | failed={job['failed_items']}"
                )
                st.progress(progress)
                if job.get("last_error"):
                    st.caption(f"Last error: {job['last_error']}")
        else:
            st.info("No indexing jobs have been queued yet.")
        st.markdown("</div>", unsafe_allow_html=True)

    with people_tab:
        st.markdown('<div class="section-panel">', unsafe_allow_html=True)
        st.subheader("Face clusters and aliases")
        if not dashboard["face_clusters"]:
            st.info("No face clusters yet. Index photos with visible faces to populate this view.")
        else:
            for cluster in dashboard["face_clusters"]:
                cols = st.columns([1, 2, 1])
                with cols[0]:
                    if cluster.get("sample_face_path"):
                        st.image(cluster["sample_face_path"], use_container_width=True)
                with cols[1]:
                    st.write(f"Cluster {cluster['cluster_id'][:8]}")
                    st.caption(f"face_count={cluster['face_count']}")
                    alias_key = f"alias_{cluster['cluster_id']}"
                    alias_value = st.text_input(
                        "Alias / contact label",
                        value=cluster.get("alias_name") or "",
                        key=alias_key,
                    )
                with cols[2]:
                    if st.button("Save Alias", key=f"save_{cluster['cluster_id']}"):
                        rename_face_cluster(runtime, cluster["cluster_id"], alias_value)
                        st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
