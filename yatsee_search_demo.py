"""
Yatsee Civic Research Engine (Streamlit UI)

This module provides an interactive, local-first research interface for
searching civic meeting transcripts and AI-generated summaries.

Design constraints:
- Streamlit owns the entrypoint (no CLI main())
- Configuration is sourced exclusively from yatsee.toml and entity config.toml
- Heavy resources (models, DB connections) are cached
- Search logic is UI-agnostic and reusable
"""

# Standard library
import argparse
import math
import os
import re
from glob import glob
from typing import Any, Dict, List
import sys

# Third-party imports
import chromadb
import streamlit as st
import toml
from sentence_transformers import SentenceTransformer

parser = argparse.ArgumentParser(description="Yatsee Search")
parser.add_argument("-e", "--entity", help="Entity handle (e.g. example_entity_name)")
parser.add_argument("-c", "--config", default="yatsee.toml", help="Path to global yatsee.toml")
args = parser.parse_args()

entity = args.entity or None
if entity is None:
    err_msg = (
        "❌ No entity specified.\n"
        "This application requires an explicit entity to be defined.\n\n"
        "Usage:\n"
        "  streamlit run yatsee_search_demo.py -- -e your_entity_name"
    )
    print(err_msg, file=sys.stderr)  # CLI visibility
    st.error(err_msg)                # Streamlit UI
    st.stop()                        # Stop execution

# ============================================================================
# PAGE SETUP (UI-ONLY, NO LOGIC)
# ============================================================================

st.set_page_config(
    page_title="Yatsee: Civic Research Engine",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Global CSS tweaks to improve sidebar usability.
# These intentionally override Streamlit defaults.
st.markdown(
    """
    <style>
    section[data-testid="stSidebar"]
    [data-baseweb="select"]
    [role="option"],

    section[data-testid="stSidebar"]
    [data-baseweb="select"]
    [role="combobox"],

    section[data-testid="stSidebar"]
    [data-baseweb="select"]

    div {
        cursor: pointer !important;
        user-select: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================================
# CONFIG LOADING
# ============================================================================

def load_global_config(path: str) -> Dict[str, Any]:
    """
    Load the global YATSEE configuration file.

    This file defines:
    - System defaults
    - Model registry
    - Known entities and their metadata

    :param path: Path to yatsee.toml
    :return: Parsed configuration dictionary
    :raises FileNotFoundError: If config file does not exist
    :raises ValueError: If TOML parsing fails
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Global configuration file not found: {path}")

    try:
        return toml.load(path)
    except Exception as exc:
        raise ValueError(f"Failed to parse global config '{path}': {exc}") from exc


@st.cache_resource
def load_entity_config(global_cfg: Dict[str, Any], entity: str) -> Dict[str, Any]:
    """
    Load and merge entity-specific configuration with global defaults.

    Merge precedence (last wins):
    - system defaults
    - entity definition from yatsee.toml
    - entity-local settings from config.toml

    Cached to avoid repeated disk reads during UI interaction.

    :param global_cfg: Parsed global configuration
    :param entity: Entity handle (e.g. 'us_ca_fresno_city_council')
    :return: Merged entity configuration dictionary
    :raises KeyError: If entity is not defined
    :raises FileNotFoundError: If entity config.toml is missing
    """
    reserved_keys = {"settings", "meta"}

    entities_cfg = global_cfg.get("entities", {})
    if entity not in entities_cfg:
        raise KeyError(f"Entity '{entity}' not defined in global config")

    system_cfg = global_cfg.get("system", {})
    root_data_dir = os.path.abspath(system_cfg.get("root_data_dir", "./data"))
    local_path = os.path.join(root_data_dir, entity, "config.toml")
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"Local config for entity '{entity}' not found at: {local_path}")

    local_cfg = toml.load(local_path)

    merged = {
        "entity": entity,
        "root_data_dir": root_data_dir,
        **system_cfg,
        **entities_cfg.get(entity, {}),
    }

    for key, value in local_cfg.items():
        if key in reserved_keys:
            merged.update(value)
        else:
            merged[key] = value
    return merged


# ============================================================================
# CONFIG RESOLUTION (RUNTIME CONSTANTS)
# ============================================================================

try:
    global_cfg = load_global_config(args.config)
    entity_cfg = load_entity_config(global_cfg, entity)
except Exception as exc:
    st.error(f"❌ Configuration load failed: {exc}")
    st.stop()

data_path = entity_cfg.get("data_path")
if not data_path or not os.path.exists(data_path):
    st.error(f"❌ Invalid data path: {data_path}")
    st.stop()

# Paths derived strictly from config
CHROMA_PATH = os.path.join(data_path, "yatsee_db")
COLLECTION_NAME = "council_knowledge"

MODEL_NAME = (
    entity_cfg.get("embedding_model")
    or global_cfg.get("system", {}).get("default_embedding_model")
    or "BAAI/bge-small-en-v1.5"
)

SUMMARY_DIR = os.path.join(data_path, global_cfg["models"][entity_cfg["summarization_model"]]["append_dir"])
TRANSCRIPT_DIR = os.path.join(data_path, "normalized")
DOWNLOAD_TRACKER = os.path.join(data_path, "downloads", ".downloaded")

# UI tuning defaults
TOP_K = 10
SNIPPET_CHARS = 350
MIN_SNIPPET_LENGTH = 50


# ============================================================================
# CACHED RESOURCES (MODELS, DB, LOOKUPS)
# ============================================================================

@st.cache_resource
def load_model(embedding_model: str) -> SentenceTransformer:
    """
    Load and cache the sentence embedding model.

    Cached as a resource because:
    - Model loading is expensive
    - Model is reused across queries
    - Device placement (CPU/GPU) should be stable
    """
    with st.spinner(f"⚡ Loading AI Engine ({embedding_model})..."):
        return SentenceTransformer(embedding_model, device="cuda")


@st.cache_resource
def get_chroma_collection():
    """
    Load and cache the ChromaDB collection.

    Cached to avoid reconnecting or reloading metadata on every query.
    """
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_collection(name=COLLECTION_NAME)


@st.cache_data
def load_video_id_map(tracker_path: str = DOWNLOAD_TRACKER) -> Dict[str, str]:
    """
    Load YouTube ID case-sensitivity map from the download tracker.

    This allows reconstruction of canonical YouTube URLs even when
    filenames were normalized to lowercase on disk.
    """
    id_map: Dict[str, str] = {}

    if os.path.exists(tracker_path):
        try:
            with open(tracker_path, "r", encoding="utf-8") as f:
                for line in f:
                    real_id = line.strip()
                    if real_id and len(real_id) == 11:
                        id_map[real_id.lower()] = real_id
        except Exception:
            pass

    return id_map


# ============================================================================
# TEXT & METADATA UTILITIES
# ============================================================================

def load_text_file(path: str) -> str:
    """
    Load a UTF-8 text file from disk.

    :param path: File path
    :return: File contents or error placeholder
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return "Failed to load file."


def extract_context_from_filename(filename: str, extra_note: str = None) -> str:
    """
    Extract a human-readable label and date from a filename.

    Supports:
    - ISO date formats (YYYY-MM-DD)
    - US-style dates (MM-DD-YYYY)

    :param filename: Filename to parse
    :return: Tuple of (label, ISO date string)
    """
    base = os.path.basename(filename)
    base = base.replace(".txt", "").replace(".summary.md", "").replace(".punct", "")

    date_match = re.search(r"(\d{4}[-_]\d{1,2}[-_]\d{1,2})", base)
    is_iso = True

    if not date_match:
        date_match = re.search(r"(\d{1,2}[-_]\d{1,2}[-_]\d{2,4})", base)
        is_iso = False

    date_str = "0000-00-00"
    label_part = base

    if date_match:
        raw_date = date_match.group(1)
        label_part = label_part.replace(raw_date, "")
        parts = re.split(r"[-_]", raw_date)

        if is_iso:
            year, month, day = parts
        else:
            month, day, year = parts

        if len(year) == 2:
            year = "20" + year

        date_str = f"{year}-{int(month):02d}-{int(day):02d}"

    if "." in label_part:
        label_part = label_part.split(".", 1)[1]

    label = re.sub(r"\s+", " ", label_part.replace("_", " ").replace("-", " ")).strip().title()
    if not label:
        label = "Meeting Transcript"

    return f"{label} | {date_str}", date_str


def get_video_url(filename: str) -> str:
    """
    Reconstruct a YouTube URL from a transcript filename.

    Uses the ID map to restore original case sensitivity.
    """
    base = os.path.basename(filename)
    if "." in base:
        file_id = base.split(".")[0]
        if len(file_id) == 11:
            id_map = load_video_id_map()
            real_id = id_map.get(file_id.lower(), file_id)
            return f"https://www.youtube.com/watch?v={real_id}"
    return None


def calculate_text_stats(text: str) -> Dict[str, Any]:
    """
    Compute basic statistics for a block of text.

    Used for summary metadata display.
    """
    words = text.split()
    return {
        "words": len(words),
        "read_time": f"{math.ceil(len(words) / 230)} min",
        "money_count": text.count("$"),
    }


# ============================================================================
# SEARCH PIPELINE (EMBEDDING + QUERY + RERANK)
# ============================================================================

embedder = load_model(MODEL_NAME)


def get_vector(query: str) -> List[float]:
    """
    Generate an embedding vector for a search query.

    Uses the standard instruction prefix expected by BGE-style models
    to ensure embeddings are aligned with passage representations.

    :param query: User search query
    :return: Normalized embedding vector
    """
    instruction = "Represent this sentence for searching relevant passages: "
    return embedder.encode(
        instruction + query,
        normalize_embeddings=True,
    ).tolist()


def chroma_query(vector: List[float], n_results: int = 10, category: str = None,) -> List[Dict[str, Any]]:
    """
    Execute a vector similarity search against ChromaDB.

    Optionally filters by document category to allow separate
    summary vs transcript retrieval strategies.

    :param vector: Query embedding vector
    :param n_results: Number of results to return
    :param category: Optional metadata category filter
    :return: List of structured search results
    """
    collection = get_chroma_collection()
    where_clause = {"category": category} if category else {}

    raw = collection.query(
        query_embeddings=[vector],
        n_results=n_results,
        where=where_clause,
        include=["documents", "metadatas", "distances"],
    )

    results: List[Dict[str, Any]] = []

    if raw["ids"] and raw["ids"][0]:
        for i in range(len(raw["ids"][0])):
            results.append(
                {
                    "document": raw["documents"][0][i],
                    "metadata": raw["metadatas"][0][i],
                    "similarity": 1.0 - raw["distances"][0][i],
                }
            )

    return results


def literal_boost(results: List[Dict[str, Any]], query: str, boost: float = 0.2, demote: float = 0.5,) -> List[Dict[str, Any]]:
    """
    Boost results containing literal query matches.

    This acts as a relevance correction mechanism when embeddings
    alone surface semantically similar but contextually weak matches.

    :param results: Search results
    :param query: User query
    :param boost: Score increase for literal matches
    :param demote: Score multiplier for non-matches
    :return: Re-ranked results
    """
    q = query.lower()

    for r in results:
        text = r["document"].lower()
        base = r.get("similarity", 0.0)

        if q in text:
            r["score"] = base + boost
        else:
            r["score"] = base * demote

    return sorted(results, key=lambda r: r["score"], reverse=True)


def hybrid_rerank(results: List[Dict[str, Any]], query: str, alpha: float = 0.7,) -> List[Dict[str, Any]]:
    """
    Combine embedding similarity with literal presence scoring.

    This is a softer alternative to literal boosting and works
    better for partial or noisy matches.

    :param results: Search results
    :param query: User query
    :param alpha: Weight for embedding similarity
    :return: Re-ranked results
    """
    q = query.lower()

    for r in results:
        text = r["document"].lower()
        literal = 1.0 if q in text else 0.0
        r["score"] = alpha * r["similarity"] + (1 - alpha) * literal

    return sorted(results, key=lambda r: r["score"], reverse=True)


def highlight_snippet(text: str, query: str, max_chars: int = 300,) -> str:
    """
    Extract and highlight a snippet centered around the first query match.

    Guarantees keyword visibility where possible.

    :param text: Full document text
    :param query: Search query
    :param max_chars: Maximum snippet length
    :return: Highlighted snippet
    """
    q = query.lower()
    lower = text.lower()
    match = re.search(re.escape(q), lower)

    if not match:
        snippet = text[:max_chars].strip()
        return snippet + "..." if len(text) > max_chars else snippet

    start = max(0, match.start() - max_chars // 2)
    end = min(len(text), match.end() + max_chars // 2)
    snippet = text[start:end]

    if start > 0:
        snippet = "..." + snippet
    if end < len(text):
        snippet += "..."

    pattern = re.compile(re.escape(query), re.IGNORECASE)
    return pattern.sub(lambda m: f"**{m.group(0)}**", snippet)


def generate_snippet(text: str, query: str, max_chars: int = SNIPPET_CHARS, min_len: int = MIN_SNIPPET_LENGTH,) -> str:
    """
    Generate a guaranteed-visible snippet for display.

    Falls back to a leading excerpt if highlighting produces
    an undersized result.

    :param text: Document text
    :param query: User query
    :return: Snippet string
    """
    snippet = highlight_snippet(text, query, max_chars)
    if not snippet or len(snippet) < min_len:
        return text[:max_chars] + "..."
    return snippet


def aggregate_results(
    summary_results: List[Dict[str, Any]],
    transcript_results: List[Dict[str, Any]],
    query: str,
    boost_summary: float = 0.1,
) -> List[Dict[str, Any]]:
    """
    Merge, deduplicate, and score results from multiple sources.

    Summary documents receive a slight relevance boost to
    encourage higher-level context discovery.

    :param summary_results: Summary search hits
    :param transcript_results: Transcript search hits
    :param query: User query
    :return: Aggregated ranked results
    """
    combined: Dict[str, Dict[str, Any]] = {}

    for bucket in (summary_results, transcript_results):
        for r in bucket:
            meta = r["metadata"]
            score = r.get("score", r.get("similarity", 0.0))
            snippet = generate_snippet(r["document"], query)

            if meta.get("category") == "summary":
                key = f"SUMMARY_{meta.get('source')}"
                score += boost_summary
            else:
                key = f"TRANSCRIPT_{meta.get('source')}_{meta.get('chunk_index')}"

            if key not in combined:
                combined[key] = {
                    "heading": meta.get("source"),
                    "meta": meta,
                    "snippet": snippet,
                    "score": score,
                }

    return sorted(combined.values(), key=lambda r: r["score"], reverse=True)


def literal_fallback(query: str) -> List[Dict[str, Any]]:
    """
    Perform a brute-force literal search over all files.

    Used as a safety net when embeddings miss exact matches.
    """
    results: Dict[str, Dict[str, Any]] = {}

    files = (
        glob(os.path.join(TRANSCRIPT_DIR, "*.txt"))
        + glob(os.path.join(SUMMARY_DIR, "*.md"))
    )

    for path in files:
        if path.endswith(".punct.txt"):
            continue

        text = load_text_file(path)
        if query.lower() not in text.lower():
            continue

        fname = os.path.basename(path)
        snippet = generate_snippet(text, query)

        results[f"LITERAL_{fname}_{len(results)}"] = {
            "heading": fname,
            "meta": {
                "source": fname,
                "category": "summary" if fname.endswith(".md") else "raw_transcript",
            },
            "snippet": snippet,
            "score": 2.0,
        }

    return list(results.values())


def run_search_pipeline(query_text: str, rerank_strategy: str = "literal",) -> List[Dict[str, Any]]:
    """
    Orchestrate the full search pipeline.

    Steps:
    - Embed query
    - Vector search (summary + transcript)
    - Re-rank
    - Aggregate
    - Literal fallback
    """
    if not query_text:
        return []

    vector = get_vector(query_text)

    summaries = chroma_query(vector, n_results=5, category="summary")
    transcripts = chroma_query(vector, n_results=35, category="raw_transcript")

    if rerank_strategy == "hybrid":
        summaries = hybrid_rerank(summaries, query_text)
        transcripts = hybrid_rerank(transcripts, query_text)
    else:
        summaries = literal_boost(summaries, query_text)
        transcripts = literal_boost(transcripts, query_text)

    aggregated = aggregate_results(summaries, transcripts, query_text)
    aggregated.extend(literal_fallback(query_text))

    aggregated.sort(key=lambda r: r["score"], reverse=True)
    return aggregated[:TOP_K]


# ==========================================
# SEARCH INTERFACE INTEGRATION
# ==========================================
def run_search():
    """
    Trigger the search pipeline and persist results in session state.

    Intended to be called via Streamlit input callbacks.
    """
    query_text = st.session_state.search_input.strip()
    if not query_text:
        st.session_state.search_results = []
        return

    st.session_state.last_query = query_text
    st.session_state.viewing_transcript = None
    st.session_state.search_results = run_search_pipeline(
        query_text,
        rerank_strategy="literal",
    )


# ==========================================
# SESSION STATE INITIALIZATION
# ==========================================
if "search_results" not in st.session_state:
    st.session_state.search_results = []

if "last_query" not in st.session_state:
    st.session_state.last_query = None

if "selected_summary" not in st.session_state:
    st.session_state.selected_summary = "-- Select Meeting --"

if "viewing_transcript" not in st.session_state:
    st.session_state.viewing_transcript = None


# ==========================================
# UI: SIDEBAR
# ==========================================
with st.sidebar:
    st.header("🗂️ Summaries")

    # After parsing the CLI arg
    st.session_state.setdefault("current_entity", entity)

    # --- Summaries ---
    summary_paths = sorted(glob(os.path.join(SUMMARY_DIR, "*.summary.md")))
    summaries = []
    for path in summary_paths:
        label, date_str = extract_context_from_filename(path)
        summaries.append({
            "path": path,
            "filename": os.path.basename(path),
            "label": label,
            "date": date_str
        })
    summaries.sort(key=lambda x: x["date"], reverse=True)

    summary_labels = ["-- Select Meeting --"] + [s["label"] for s in summaries]
    try:
        idx = summary_labels.index(st.session_state.selected_summary)
    except ValueError:
        idx = 0

    selected_label = st.selectbox("Browse by Meeting:", summary_labels, index=idx)
    if selected_label != st.session_state.selected_summary:
        st.session_state.selected_summary = selected_label
        st.session_state.viewing_transcript = None
        st.session_state.search_results = []  # Clear search results
        st.rerun()

    # --- Flexible spacer ---
    # st.markdown("<br><br><br><br><br>", unsafe_allow_html=True)  # crude spacer, pushes status toward bottom

    # --- Flexible spacer ---
    # This creates a container that will take up the available vertical space
    with st.container():
        st.markdown(" ", unsafe_allow_html=True)  # A dummy element inside


    # --- System Status ---
    st.divider()
    st.caption("⚙️ **System Status**")
    st.caption(f"• **Current Entity:** {st.session_state.current_entity}")
    st.caption(f"• **Index:** {COLLECTION_NAME}")
    st.caption(f"• **Meetings Indexed:** {len(summaries)}")
    st.caption(f"• **Accelerator:** `{embedder.device}`")
    st.caption("v1.0 • Local-First")


# ==========================================
# UI: MAIN CONTENT
# ==========================================
st.title("🏛️ Yatsee Research Engine")
st.info(
    "⚠️ **Research Preview:** Summaries are AI-generated. "
    "Verify important details against the canonical transcript or recording."
)


# ==========================================
# VIEW A: FULL TRANSCRIPT READER
# ==========================================
if st.session_state.viewing_transcript:
    transcript_path = st.session_state.viewing_transcript
    fname = os.path.basename(transcript_path)
    pretty_name, _ = extract_context_from_filename(fname)

    c1, c2 = st.columns([6, 1])
    with c1:
        st.subheader(f"📄 Transcript: {pretty_name}")
        if st.session_state.last_query:
            st.caption(
                f"🔎 Highlight context for query: "
                f"**'{st.session_state.last_query}'**"
            )
    with c2:
        if st.button("❌ Close", type="secondary"):
            st.session_state.viewing_transcript = None
            st.rerun()

    full_text = load_text_file(transcript_path)
    st.code(full_text, language=None)

    st.divider()
    if st.button("⬅️ Back to Search Results"):
        st.session_state.viewing_transcript = None
        st.rerun()


# ==========================================
# VIEW B: SUMMARY READER
# ==========================================
elif st.session_state.selected_summary != "-- Select Meeting --":
    selected = next(
        (s for s in summaries if s["label"] == st.session_state.selected_summary), None)

    if selected:
        content = load_text_file(selected["path"])
        stats = calculate_text_stats(content)

        # --- Navigation / Actions ---
        c1, c2, c3, c4 = st.columns([1, 3, 1.5, 1.5])
        with c1:
            if st.button("⬅️ Search"):
                st.session_state.selected_summary = "-- Select Meeting --"
                st.rerun()

        with c3:
            video_url = get_video_url(selected["filename"])
            if video_url:
                st.link_button("📺 View Video", video_url)

        with c4:
            transcript_filename = selected["filename"].replace(".summary.md", ".txt")
            transcript_path = os.path.join(TRANSCRIPT_DIR, transcript_filename)
            if os.path.exists(transcript_path):
                if st.button("📄 Transcript", type="primary"):
                    st.session_state.viewing_transcript = transcript_path
                    st.rerun()

        st.divider()

        # --- Intelligence Dashboard ---
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("📅 Date", selected["date"])
        m2.metric("📝 Word Count", f"{stats['words']:,}")
        m3.metric("⏱️ Est. Read Time", stats["read_time"])
        m4.metric("💰 Money Mentions", stats["money_count"])

        st.divider()
        st.header(selected["label"])
        st.markdown(content)

        st.divider()
        if st.button("⬅️ Back to Search"):
            st.session_state.selected_summary = "-- Select Meeting --"
            st.rerun()


# ==========================================
# VIEW C: SEARCH INTERFACE (DEFAULT)
# ==========================================
else:
    st.markdown("### Search Archives")

    st.text_input(
        "Search",
        label_visibility="collapsed",
        placeholder="Search for 'police budget', 'zoning variance', 'Main St'...",
        key="search_input",
        on_change=run_search,
    )

    if st.session_state.search_results:
        st.divider()
        st.subheader(
            f"Found {len(st.session_state.search_results)} results "
            f"for '{st.session_state.last_query}'"
        )

for i, result in enumerate(st.session_state.search_results):
    meta = result["meta"]
    snippet = result["snippet"]

    pretty_heading, date = extract_context_from_filename(meta["source"])
    is_summary = meta.get("category") == "summary"
    type_label = "📝 [SUMMARY]" if is_summary else "📄 [TRANSCRIPT]"
    confidence = "High" if i < 3 else "Medium"

    with st.expander(f"{type_label} {pretty_heading}"):
        # --- Snippet & meta ---
        c_snip, c_info = st.columns([3, 1])
        with c_snip:
            st.markdown(f"...{snippet}...", unsafe_allow_html=True)
        with c_info:
            st.caption(f"**Date:** {date}")
            st.caption(f"**Relevance:** {confidence}")
            if meta.get("chunk_index") is not None:
                st.caption(f"**Chunk:** {meta['chunk_index']}")
            elif not is_summary:
                st.caption("**Match:** Literal Text")

        st.markdown("---")

        # --- Action Buttons ---
        btn_cols = st.columns(4)

        # Video button (if available)
        video_url = get_video_url(meta["source"])
        if video_url:
            btn_cols[3].link_button("📺 Video", video_url)

        # Determine the summary file corresponding to this snippet
        summary_filename = meta["source"].replace(".txt", ".summary.md") if not is_summary else meta["source"]
        matching_summary = next((s for s in summaries if s["filename"] == summary_filename), None)
        if matching_summary and btn_cols[0].button("Read Summary", key=f"btn_sum_{i}"):
            st.session_state.selected_summary = matching_summary["label"]
            st.session_state.search_results = []  # clear search results
            st.rerun()

        # Determine the transcript file corresponding to this snippet
        transcript_filename = meta["source"].replace(".summary.md", ".txt") if is_summary else meta["source"]
        transcript_path = os.path.join(TRANSCRIPT_DIR, transcript_filename)
        if os.path.exists(transcript_path) and btn_cols[1].button("View Transcript", key=f"btn_trans_{i}"):
            st.session_state.viewing_transcript = transcript_path
            st.session_state.search_results = []  # clear search results
            st.rerun()
