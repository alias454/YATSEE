import os
import re
from glob import glob
from tqdm import tqdm
import chromadb
from sentence_transformers import SentenceTransformer
import spacy

# --- CONFIGURATION ---
CHROMA_PATH = "./yatsee_db"
COLLECTION_NAME = "council_knowledge"
SUMMARY_DIR = "summary_nemo"
TRANSCRIPT_DIR = "normalized"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
DOWNLOAD_TRACKER = "./downloads/.downloaded"  # Path to your ID source

# --- CHUNKING ---
TRANSCRIPT_CHUNK_SIZE = 150
TRANSCRIPT_OVERLAP = 25
SUMMARY_CHUNK_SIZE = 300
SUMMARY_OVERLAP = 50

# --- LOAD MODELS ---
print(f"⚡ Loading {MODEL_NAME} to GPU...")
embedder = SentenceTransformer(MODEL_NAME, device="cuda")

print("⚡ Loading SpaCy for Entity Extraction...")
try:
    nlp = spacy.load("en_core_web_md")
except OSError:
    print("⚠️  'en_core_web_md' not found. Using 'sm' (install 'md' for better NER).")
    nlp = spacy.load("en_core_web_sm")


# -------------------------
# Utility functions
# -------------------------

def load_video_id_map(tracker_path):
    """
    Reads the download tracker file to map LowercaseID -> RealID.
    This ensures our DB links work even if filenames are lowercase.
    """
    id_map = {}
    if os.path.exists(tracker_path):
        try:
            with open(tracker_path, "r", encoding="utf-8") as f:
                for line in f:
                    real_id = line.strip()
                    if len(real_id) == 11:
                        id_map[real_id.lower()] = real_id
        except Exception as e:
            print(f"⚠️ Warning: Could not read download tracker: {e}")
    return id_map


def get_embedding(text):
    return embedder.encode(text, normalize_embeddings=True).tolist()


def split_text_window(text, chunk_size, overlap):
    words = text.split()
    chunks = []
    if len(words) <= chunk_size:
        return [" ".join(words)]
    for i in range(0, len(words), chunk_size - overlap):
        chunk = words[i:i + chunk_size]
        if len(chunk) > 10:
            chunks.append(" ".join(chunk))
    return chunks


def parse_metadata(filename, category, id_map):
    """
    Extracts metadata AND injects the persistent Video URL.
    """
    meta = {
        "source": filename,
        "category": category,
        "type": "general",
        "date": "unknown",
        "video_url": ""  # Default empty
    }

    # 1. Type extraction
    lower = filename.lower()
    if "finance" in lower or "budget" in lower:
        meta["type"] = "finance"
    elif "zoning" in lower:
        meta["type"] = "zoning"
    elif "council" in lower:
        meta["type"] = "council"

    # 2. Date extraction
    date_match = re.search(r"(\d{4}[-_]\d{1,2}[-_]\d{1,2})", filename)
    if not date_match:
        date_match = re.search(r"(\d{1,2}[-_]\d{1,2}[-_]\d{2,4})", filename)
    if date_match:
        meta["date"] = date_match.group(1).replace("_", "-")

    # 3. Video URL Injection (The New Feature)
    if '.' in filename:
        file_id_prefix = filename.split('.')[0]
        if len(file_id_prefix) == 11:
            # Recover Case Sensitivity using the map
            real_id = id_map.get(file_id_prefix.lower(), file_id_prefix)
            meta["video_id"] = real_id
            meta["video_url"] = f"https://www.youtube.com/watch?v={real_id}"

    return meta


# -------------------------
# Main ingestion
# -------------------------
def main():
    print("⚠️  REMINDER: Manually delete './yatsee_db' for a fresh index!")
    print(f"🚜 Stage 7: Indexing with {MODEL_NAME}")

    # Load the ID map once
    id_map = load_video_id_map(DOWNLOAD_TRACKER)
    print(f"🗺️  Loaded {len(id_map)} video IDs from tracker.")

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    # ---------
    # Summaries
    # ---------
    summary_files = glob(os.path.join(SUMMARY_DIR, "*_final_summary.summary.md"))
    print(f"📚 Indexing {len(summary_files)} summaries")

    for path in tqdm(summary_files, desc="Summaries"):
        filename = os.path.basename(path)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        if not text: continue

        chunks = split_text_window(text, SUMMARY_CHUNK_SIZE, SUMMARY_OVERLAP)

        # Pass id_map to metadata parser
        base_meta = parse_metadata(filename, "summary", id_map)

        ids = []
        docs = []
        metas = []

        for i, chunk in enumerate(chunks):
            ids.append(f"SUM_{filename}_{i}")
            docs.append(chunk)
            chunk_meta = base_meta.copy()
            chunk_meta["chunk_index"] = i
            metas.append(chunk_meta)

        if docs:
            embeddings_list = embedder.encode(docs, normalize_embeddings=True, show_progress_bar=False).tolist()
            collection.upsert(ids=ids, documents=docs, embeddings=embeddings_list, metadatas=metas)

    # -------------
    # Transcripts
    # -------------
    transcript_files = glob(os.path.join(TRANSCRIPT_DIR, "*.txt"))
    print(f"🎙️ Indexing {len(transcript_files)} transcripts")

    for path in tqdm(transcript_files, desc="Transcripts"):
        filename = os.path.basename(path)
        if filename.endswith(".punct.txt"): continue

        with open(path, "r", encoding="utf-8") as f:
            full_text = f.read().strip()
        if not full_text: continue

        chunks = split_text_window(full_text, TRANSCRIPT_CHUNK_SIZE, TRANSCRIPT_OVERLAP)

        # Pass id_map to metadata parser
        base_meta = parse_metadata(filename, "raw_transcript", id_map)

        ids = []
        docs = []
        metas = []

        for i, chunk in enumerate(chunks):
            # NER Extraction
            doc = nlp(chunk)
            orgs = list(set([e.text for e in doc.ents if e.label_ == "ORG"]))
            money = list(set([e.text for e in doc.ents if e.label_ == "MONEY"]))
            persons = list(set([e.text for e in doc.ents if e.label_ == "PERSON"]))

            ids.append(f"RAW_{filename}_{i}")
            docs.append(f"Transcript excerpt: {chunk}")

            chunk_meta = base_meta.copy()
            chunk_meta["chunk_index"] = i

            if orgs: chunk_meta["orgs"] = ", ".join(orgs)
            if money: chunk_meta["money"] = ", ".join(money)
            if persons: chunk_meta["persons"] = ", ".join(persons)

            metas.append(chunk_meta)

        if docs:
            embeddings_list = embedder.encode(docs, normalize_embeddings=True, show_progress_bar=False).tolist()
            collection.upsert(ids=ids, documents=docs, embeddings=embeddings_list, metadatas=metas)

    print(f"\n✅ Indexing complete. DB at {CHROMA_PATH}")


if __name__ == "__main__":
    main()