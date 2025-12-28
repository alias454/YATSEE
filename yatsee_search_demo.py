import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import os
import math
import re
from glob import glob

# --- CONFIG ---
CHROMA_PATH = "./yatsee_db"
COLLECTION_NAME = "council_knowledge"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
SUMMARY_DIR = "summary_nemo"
TRANSCRIPT_DIR = "normalized"
DOWNLOAD_TRACKER = "./downloads/.downloaded"  # Path to ID history
TOP_K = 10
SNIPPET_CHARS = 300
MIN_SNIPPET_LENGTH = 50

# --- SETUP PAGE ---
st.set_page_config(
    page_title="Yatsee: Civic Research",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --- LOAD LOCAL MODEL (Cached) ---
@st.cache_resource
def load_model():
    with st.spinner(f"⚡ Loading AI Engine ({MODEL_NAME})..."):
        return SentenceTransformer(MODEL_NAME, device="cuda")


embedder = load_model()


# --- LOAD ID MAP (Cached) ---
@st.cache_data
def load_video_id_map(tracker_path=DOWNLOAD_TRACKER):
    """
    Reads the download tracker to create a lookup for Real Case IDs.
    Maps lowercase filenames back to case-sensitive YouTube IDs.
    """
    id_map = {}
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


def extract_context_from_filename(filename, extra_note=None):
    base = os.path.basename(filename)
    base = base.replace(".txt", "").replace(".summary.md", "").replace(".punct", "")

    # Priority: YYYY-MM-DD (ISO)
    date_match = re.search(r"(\d{4}[-_]\d{1,2}[-_]\d{1,2})", base)
    is_iso = True

    # Fallback: MM-DD-YYYY
    if not date_match:
        date_match = re.search(r"(\d{1,2}[-_]\d{1,2}[-_]\d{2,4})", base)
        is_iso = False

    date_str = "0000-00-00"
    label_part = base

    if date_match:
        raw_date_str = date_match.group(1)
        label_part = label_part.replace(raw_date_str, "")

        parts = re.split(r'[-_]', raw_date_str)
        if is_iso:
            year, month, day = parts
        else:
            month, day, year = parts

        if len(year) == 2: year = "20" + year

        date_str = f"{year}-{int(month):02d}-{int(day):02d}"

    if '.' in label_part:
        label_part = label_part.split('.', 1)[1]

    label = label_part.replace("_", " ").replace("-", " ").strip()
    label = re.sub(r"\s+", " ", label).title()

    if not label: label = "Meeting Transcript"

    return f"{label} | {date_str}", date_str


def get_video_url(filename):
    """
    Reconstructs the YouTube URL using the ID map to restore case sensitivity.
    """
    base = os.path.basename(filename)
    if '.' in base:
        file_id = base.split('.')[0]
        if len(file_id) == 11:
            # Look up the real case-sensitive ID
            id_map = load_video_id_map()
            real_id = id_map.get(file_id.lower(), file_id)
            return f"https://www.youtube.com/watch?v={real_id}"
    return None


def calculate_text_stats(text):
    words = text.split()
    word_count = len(words)

    # Average reading speed = 230 wpm
    read_time_min = math.ceil(word_count / 230)

    # Count dollar signs as a proxy for financial density
    money_mentions = text.count('$')

    return {
        "words": word_count,
        "read_time": f"{read_time_min} min",
        "money_count": money_mentions
    }


def get_embedding(text):
    instruction = "Represent this sentence for searching relevant passages: "
    return embedder.encode(instruction + text, normalize_embeddings=True).tolist()


def highlight_snippet(text, query, radius=SNIPPET_CHARS):
    pattern = re.compile(re.escape(query), re.IGNORECASE)
    match = pattern.search(text)
    if match:
        start = max(0, match.start() - radius)
        end = min(len(text), match.end() + radius)
        snippet = text[start:end]
        snippet = pattern.sub(lambda m: f"**{m.group(0)}**", snippet)
        return snippet
    return None


def load_text_file(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except:
        return "Failed to load file."


# --- SEARCH ENGINE ---
def run_search():
    query_text = st.session_state.search_input
    if not query_text: return

    st.session_state.last_query = query_text
    st.session_state.viewing_transcript = None

    # Thresholds
    is_short_query = len(query_text.split()) < 3
    score_threshold = 0.50 if is_short_query else 0.65

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(name=COLLECTION_NAME)

    # Generate vector once
    vector = get_embedding(query_text)

    # --- STRATEGY: SPLIT QUERY ---
    # We force Chroma to look for summaries specifically, ensuring they aren't
    # crowded out by the massive volume of transcript chunks.

    # 1. Fetch Top Summaries (Guaranteed slots)
    results_sum = collection.query(
        query_embeddings=[vector],
        n_results=5,
        where={"category": "summary"},  # <--- Force Category
        include=["documents", "metadatas", "distances"]
    )

    # 2. Fetch Top Transcripts (The rest of the slots)
    results_trans = collection.query(
        query_embeddings=[vector],
        n_results=35,
        where={"category": "raw_transcript"},  # <--- Force Category
        include=["documents", "metadatas", "distances"]
    )

    unique_results_map = {}

    # Helper to process a result batch
    def process_batch(results):
        if results['ids'] and results['ids'][0]:
            for i, doc_id in enumerate(results['ids'][0]):
                doc_text = results['documents'][0][i]
                meta = results['metadatas'][0][i]
                distance = results['distances'][0][i]

                if distance > score_threshold: continue

                snippet = highlight_snippet(doc_text, query_text)
                if snippet is None: snippet = doc_text[:300] + "..."

                if meta.get("category") == "summary":
                    key = f"SUMMARY_{meta.get('source')}"
                    score_boost = 0.1  # Give summaries a tiny ranking boost
                else:
                    key = f"TRANSCRIPT_{meta.get('source')}_{meta.get('chunk_index')}"
                    score_boost = 0.0

                if key not in unique_results_map:
                    unique_results_map[key] = {
                        "heading": meta.get("source"),
                        "meta": meta,
                        "snippet": snippet,
                        "score": (1.0 - distance) + score_boost
                    }

    # Process both batches
    process_batch(results_sum)
    process_batch(results_trans)

    # --- LITERAL FALLBACK (Unchanged) ---
    all_files = glob(os.path.join(TRANSCRIPT_DIR, "*.txt")) + glob(os.path.join(SUMMARY_DIR, "*.md"))
    for filepath in all_files:
        try:
            filename = os.path.basename(filepath)
            if filename.endswith(".punct.txt"): continue

            is_summary = filename.endswith(".md")
            check_key = f"SUMMARY_{filename}" if is_summary else None

            if check_key and check_key in unique_results_map: continue

            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()

            if query_text.lower() in text.lower():
                snippet = highlight_snippet(text, query_text)
                if snippet:
                    entry = {
                        "heading": filename,
                        "meta": {"source": filename, "category": "summary" if is_summary else "raw_transcript"},
                        "snippet": snippet,
                        "score": 2.0
                    }
                    if is_summary:
                        unique_results_map[check_key] = entry
                    else:
                        unique_results_map[f"LITERAL_{filename}_{len(unique_results_map)}"] = entry
        except:
            continue

    # Sort and store
    final_list = list(unique_results_map.values())
    final_list.sort(key=lambda x: x["score"], reverse=True)
    st.session_state.search_results = final_list[:15]


# --- SESSION STATE INITIALIZATION ---
if "search_results" not in st.session_state: st.session_state.search_results = []
if "last_query" not in st.session_state: st.session_state.last_query = ""
if "selected_summary" not in st.session_state: st.session_state.selected_summary = "-- None --"
if "viewing_transcript" not in st.session_state: st.session_state.viewing_transcript = None

# ==========================================
# UI: SIDEBAR
# ==========================================
with st.sidebar:
    st.header("🗂️ Summaries")

    summary_paths = sorted(glob(os.path.join(SUMMARY_DIR, "*_final_summary.summary.md")))
    summaries = []
    for path in summary_paths:
        label, date_str = extract_context_from_filename(path)
        summaries.append({"path": path, "filename": os.path.basename(path), "label": label, "date": date_str})

    summaries.sort(key=lambda x: x["date"], reverse=True)
    summary_labels = ["-- Select Meeting --"] + [s["label"] for s in summaries]

    try:
        idx = summary_labels.index(st.session_state.selected_summary)
    except:
        idx = 0

    selected_label = st.selectbox("Browse by Meeting:", summary_labels, index=idx)

    if selected_label != st.session_state.selected_summary:
        st.session_state.selected_summary = selected_label
        st.session_state.viewing_transcript = None
        st.rerun()

    st.divider()
    st.caption("⚙️ **System Status**")
    st.caption(f"• **Index:** {COLLECTION_NAME}")
    st.caption(f"• **Documents:** {len(summaries)}")
    st.caption(f"• **Accelerator:** `{embedder.device}`")
    st.caption("v1.0 (Local-First)")

# ==========================================
# UI: MAIN CONTENT
# ==========================================

st.title("🏛️ Yatsee Research Engine")
st.info(
    "⚠️ **Research Preview:** Summaries are AI-generated. "
    "Always verify key details against the **Canonical Transcript** or video recording."
)

# --- VIEW A: FULL TRANSCRIPT READER ---
if st.session_state.viewing_transcript:
    fname = os.path.basename(st.session_state.viewing_transcript)
    pretty_name, _ = extract_context_from_filename(fname)

    c1, c2 = st.columns([6, 1])
    with c1:
        st.subheader(f"📄 Transcript: {pretty_name}")
        if st.session_state.last_query:
            st.caption(f"🔎 Highlight context for query: **'{st.session_state.last_query}'**")
    with c2:
        if st.button("❌ Close", type="secondary"):
            st.session_state.viewing_transcript = None
            st.rerun()

    full_text = load_text_file(st.session_state.viewing_transcript)
    st.code(full_text, language=None)

    st.divider()
    if st.button("Back to Search Results"):
        st.session_state.viewing_transcript = None
        st.rerun()

# --- VIEW B: SUMMARY READER ---
elif st.session_state.selected_summary != "-- Select Meeting --":
    selected = next((s for s in summaries if s["label"] == st.session_state.selected_summary), None)

    if selected:
        # Load content early so we can calculate stats
        content = load_text_file(selected["path"])
        stats = calculate_text_stats(content)

        # 1. Navigation / Actions Bar (RESTORED)
        c1, c2, c3, c4 = st.columns([1, 3, 1.5, 1.5])

        with c1:
            if st.button("⬅️ Search", key="back_to_search"):
                st.session_state.selected_summary = "-- Select Meeting --"
                st.rerun()

        # (c2 is empty spacing)

        with c3:
            video_url = get_video_url(selected["filename"])
            if video_url:
                st.link_button("📺 View Video", video_url)

        with c4:
            transcript_filename = selected["filename"].replace("_final_summary.summary.md", ".txt")
            transcript_path = os.path.join(TRANSCRIPT_DIR, transcript_filename)
            if os.path.exists(transcript_path):
                if st.button("📄 Transcript", type="primary"):
                    st.session_state.viewing_transcript = transcript_path
                    st.rerun()

        st.divider()

        # 2. THE INTELLIGENCE DASHBOARD (NEW)
        # Displays high-level metrics before you read
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("📅 Date", selected['date'])
        m2.metric("📝 Word Count", f"{stats['words']:,}")
        m3.metric("⏱️ Est. Time", stats['read_time'])
        m4.metric("💰 Money Mentions", stats['money_count'])

        st.divider()

        # 3. Content Display
        st.header(selected['label'])
        st.markdown(content)

        # 4. Footer
        st.divider()
        if st.button("⬅️ Back to Search", key="back_to_search_footer"):
            st.session_state.selected_summary = "-- Select Meeting --"
            st.rerun()

# --- VIEW C: SEARCH INTERFACE (Default) ---
else:
    st.markdown("### Search Archives")
    st.text_input("Search", label_visibility="collapsed",
                  placeholder="Search for 'police budget', 'zoning variance', 'Main St'...", key="search_input",
                  on_change=run_search)

    if st.session_state.search_results:
        st.divider()
        st.subheader(f"Found {len(st.session_state.search_results)} results for '{st.session_state.last_query}'")

        for i, result in enumerate(st.session_state.search_results):
            meta = result["meta"]
            snippet = result["snippet"]
            score = result.get("score", 0)

            # 1. Format the Title
            pretty_heading, date = extract_context_from_filename(meta['source'])

            # 2. Determine Text Label (The Change)
            if meta.get("category") == "summary":
                type_label = "📝 [SUMMARY]"
            else:
                type_label = "📄 [TRANSCRIPT]"

            # 3. Calculate Confidence
            confidence = "High" if score > 0.6 else "Medium"

            # 4. Render Expander with Text Label
            with st.expander(f"{type_label} {pretty_heading}", expanded=False):
                c1, c2 = st.columns([3, 1])
                with c1:
                    st.markdown(f"**Snippet:** ...{snippet}...", unsafe_allow_html=True)
                with c2:
                    st.caption(f"**Date:** {date}")
                    st.caption(f"**Relevance:** {confidence}")

                    # Logic to handle missing chunk numbers (Literal matches)
                    chunk_id = meta.get('chunk_index')
                    if chunk_id is not None:
                        st.caption(f"**Chunk:** {chunk_id}")
                    elif meta.get("category") == "raw_transcript":
                        st.caption("**Match:** Literal Text")

                st.markdown("---")
                cols = st.columns(4)

                # Video Link
                video_url = get_video_url(meta["source"])
                if video_url:
                    cols[3].link_button("📺 Video", video_url)

                # Action Buttons
                if meta.get("category") == "summary":
                    matching = next((s for s in summaries if s["filename"] == meta["source"]), None)
                    if matching:
                        if cols[0].button(f"Read Summary", key=f"btn_sum_{i}"):
                            st.session_state.selected_summary = matching["label"]
                            st.rerun()

                elif meta.get("category") == "raw_transcript":
                    corresponding_summary = meta["source"].replace(".txt", "_final_summary.summary.md")
                    matching = next((s for s in summaries if s["filename"] == corresponding_summary), None)

                    if matching:
                        if cols[0].button(f"View Summary", key=f"btn_trans_sum_{i}"):
                            st.session_state.selected_summary = matching["label"]
                            st.rerun()

                    transcript_path = os.path.join(TRANSCRIPT_DIR, meta['source'])
                    if os.path.exists(transcript_path):
                        if cols[1].button(f"View Transcript", key=f"btn_trans_raw_{i}"):
                            st.session_state.viewing_transcript = transcript_path
                            st.rerun()