#!/usr/bin/env python3
"""
YATSEE Vector Indexer

This script creates a vector index for city council transcripts and summaries
using ChromaDB and SentenceTransformers. It supports entity-specific configuration,
chunked indexing, and NER metadata extraction.

Features:
- Load global and entity-specific TOML configs
- Split text into sentence-based chunks with overlap
- Extract metadata including finance, council, and zoning types
- Named Entity Recognition for organizations, money, and persons
- Index both summaries and transcripts into ChromaDB
"""

# Standard library
import argparse
import os
import re
import sys
from glob import glob
from typing import Any, Dict, List, Tuple

# Third-party imports
import chromadb
import spacy
import toml
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def load_global_config(path: str) -> Dict[str, Any]:
    """
    Load the global YATSEE configuration file.

    :param path: Path to the global TOML config file
    :return: Parsed global configuration as a dictionary
    :raises FileNotFoundError: If the file does not exist
    :raises ValueError: If the TOML cannot be parsed
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Global configuration file not found: {path}")
    try:
        return toml.load(path)
    except Exception as exc:
        raise ValueError(f"Failed to parse global config '{path}': {exc}") from exc


def load_entity_config(global_cfg: Dict[str, Any], entity: str) -> Dict[str, Any]:
    """
    Load and merge entity-specific configuration with global defaults.

    :param global_cfg: Global configuration dictionary
    :param entity: Entity handle to load (e.g., 'us_ca_fresno_city_council')
    :return: Merged entity configuration dictionary
    :raises KeyError: If entity is not defined in global config
    :raises FileNotFoundError: If local entity config is missing
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


def load_video_id_map(id_map_path: str) -> Dict[str, str]:
    """
    Load a mapping of video IDs from a download tracker file.

    :param id_map_path: Path to the tracker file
    :return: Dictionary mapping lowercase IDs to real IDs
    """
    id_map = {}
    if os.path.exists(id_map_path):
        with open(id_map_path, "r", encoding="utf-8") as f:
            for line in f:
                real_id = line.strip()
                if len(real_id) == 11:
                    id_map[real_id.lower()] = real_id
    return id_map


def split_text_sentences(text: str, nlp_model, chunk_size: int = 150, overlap: int = 25) -> List[str]:
    """
    Split text into sentence-based chunks with optional overlap.

    :param text: Input text to split
    :param nlp_model: SpaCy NLP model for sentence parsing
    :param chunk_size: Maximum number of words per chunk
    :param overlap: Number of overlapping words between chunks
    :return: List of text chunks
    """
    doc = nlp_model(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    chunks = []
    current_chunk = []
    current_len = 0

    for sent in sentences:
        sent_words = sent.split()
        sent_len = len(sent_words)

        if current_len + sent_len > chunk_size:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            overlap_words = current_chunk[-overlap:] if overlap and len(current_chunk) >= overlap else current_chunk
            current_chunk = overlap_words + sent_words
            current_len = len(current_chunk)
        else:
            current_chunk.extend(sent_words)
            current_len += sent_len

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def parse_metadata(filename: str, category: str, id_map: Dict[str, str]) -> Dict[str, Any]:
    """
    Parse metadata from a filename.

    :param filename: Name of the file
    :param category: Category of the file ('summary' or 'raw_transcript')
    :param id_map: Video ID mapping dictionary
    :return: Metadata dictionary
    """
    meta = {
        "source": filename,
        "category": category,
        "type": "general",
        "date": "unknown",
        "video_url": "",
    }

    lower = filename.lower()
    if "finance" in lower or "budget" in lower:
        meta["type"] = "finance"
    elif "zoning" in lower:
        meta["type"] = "zoning"
    elif "council" in lower:
        meta["type"] = "council"

    date_match = re.search(r"(\d{4}[-_]\d{1,2}[-_]\d{1,2})", filename)
    if not date_match:
        date_match = re.search(r"(\d{1,2}[-_]\d{1,2}[-_]\d{2,4})", filename)
    if date_match:
        meta["date"] = date_match.group(1).replace("_", "-")

    if "." in filename:
        file_id_prefix = filename.split(".")[0]
        if len(file_id_prefix) == 11:
            real_id = id_map.get(file_id_prefix.lower(), file_id_prefix)
            meta["video_id"] = real_id
            meta["video_url"] = f"https://www.youtube.com/watch?v={real_id}"

    return meta


def index_documents(
    files: List[str],
    nlp_model,
    embedder,
    collection,
    id_map: Dict[str, str],
    chunk_size: int,
    overlap: int,
    category: str,
    prepend_text: str = "",
) -> int:
    """
    Index a list of text files into ChromaDB.

    :param files: List of file paths
    :param nlp_model: SpaCy model for sentence splitting and NER
    :param embedder: SentenceTransformer model for embeddings
    :param collection: ChromaDB collection
    :param id_map: Video ID map for metadata
    :param chunk_size: Max words per chunk
    :param overlap: Overlap words between chunks
    :param category: 'summary' or 'raw_transcript'
    :param prepend_text: Optional text prefix for each chunk (e.g., "Transcript excerpt:")
    :return: Number of documents indexed
    """
    total_indexed = 0

    for path in tqdm(files, desc=f"Indexing {category}"):
        filename = os.path.basename(path)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        if not text:
            continue

        chunks = split_text_sentences(text, nlp_model, chunk_size, overlap)
        base_meta = parse_metadata(filename, category, id_map)

        ids, docs, metas = [], [], []
        for i, chunk in enumerate(chunks):
            doc_meta = base_meta.copy()
            doc_meta["chunk_index"] = i

            if category == "raw_transcript":
                ner_doc = nlp_model(chunk)
                orgs = list(set([e.text for e in ner_doc.ents if e.label_ == "ORG"]))
                money = list(set([e.text for e in ner_doc.ents if e.label_ == "MONEY"]))
                persons = list(set([e.text for e in ner_doc.ents if e.label_ == "PERSON"]))
                if orgs: doc_meta["orgs"] = ", ".join(orgs)
                if money: doc_meta["money"] = ", ".join(money)
                if persons: doc_meta["persons"] = ", ".join(persons)

            ids.append(f"{category.upper()}_{filename}_{i}")
            docs.append(f"{prepend_text}{chunk}" if prepend_text else chunk)
            metas.append(doc_meta)

        if docs:
            embeddings = embedder.encode(docs, normalize_embeddings=True, show_progress_bar=False).tolist()
            collection.upsert(ids=ids, documents=docs, embeddings=embeddings, metadatas=metas)
            total_indexed += len(docs)

    return total_indexed


def main() -> int:
    """
    Main CLI entry point for YATSEE indexer.

    :return: Exit code (0 = success, 1 = failure)
    """
    parser = argparse.ArgumentParser(description="Yatsee Indexer")
    parser.add_argument("-e", "--entity", help="Entity handle to process")
    parser.add_argument("-c", "--config", default="yatsee.toml", help="Path to global yatsee.toml")
    parser.add_argument("-m", "--model", help="SentenceTransformer model name")
    parser.add_argument("--summary_chunk", type=int, default=300)
    parser.add_argument("--summary_overlap", type=int, default=50)
    parser.add_argument("--transcript_chunk", type=int, default=150)
    parser.add_argument("--transcript_overlap", type=int, default=25)
    parser.add_argument("--force", action="store_true", help="Force reindexing even if DB exists")
    args = parser.parse_args()

    # Determine input/output paths
    entity_cfg = {}
    if args.entity:
        # Load entity config
        try:
            global_cfg = load_global_config(args.config)
            entity_cfg = load_entity_config(global_cfg, args.entity)
        except Exception as e:
            print(f"❌ Config load failed: {e}", file=sys.stderr)
            return 1

    data_path = entity_cfg.get("data_path")
    if not data_path or not os.path.exists(data_path):
        print(f"❌ Invalid data path: {data_path}", file=sys.stderr)
        return 1

    chroma_path = os.path.join(data_path, "yatsee_db")
    summary_dir = os.path.join(data_path, global_cfg["models"][entity_cfg["summarization_model"]]["append_dir"])
    transcript_dir = os.path.join(data_path, "normalized")
    download_tracker = os.path.join(data_path, "downloads", ".downloaded")

    if not os.path.exists(summary_dir):
        print(f"❌ Summary directory does not exist: {summary_dir}", file=sys.stderr)
        return 1
    if not os.path.exists(transcript_dir):
        print(f"❌ Transcript directory does not exist: {transcript_dir}", file=sys.stderr)
        return 1

    print(f"⚡ Loading SentenceTransformer model {args.model or 'BAAI/bge-small-en-v1.5'}...")
    embedder = SentenceTransformer(args.model or "BAAI/bge-small-en-v1.5", device="cuda")

    print(f"⚡ Loading SpaCy model {entity_cfg.get('default_sentence_model', 'en_core_web_md')}...")
    try:
        nlp_model = spacy.load(entity_cfg.get("default_sentence_model", "en_core_web_md"))
    except OSError:
        print("⚠️ SpaCy model not found, falling back to en_core_web_sm")
        nlp_model = spacy.load("en_core_web_sm")

    id_map = load_video_id_map(download_tracker)
    print(f"🗺️ Loaded {len(id_map)} video IDs")

    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_or_create_collection(name="council_knowledge", metadata={"hnsw:space": "cosine"})

    # Ingest summaries
    summary_files = glob(os.path.join(summary_dir, "*_final_summary.summary.md"))
    print(f"📚 Indexing {len(summary_files)} summaries")

    num_summaries = index_documents(
        summary_files, nlp_model, embedder, collection, id_map,
        args.summary_chunk, args.summary_overlap, category="summary"
    )

    # Ingest transcripts
    transcript_files = [
        f for f in glob(os.path.join(transcript_dir, "*.txt"))
        if not f.endswith(".punct.txt")
    ]
    print(f"🎙️ Indexing {len(transcript_files)} transcripts")

    num_transcripts = index_documents(
        transcript_files, nlp_model, embedder, collection, id_map,
        args.transcript_chunk, args.transcript_overlap,
        category="raw_transcript", prepend_text="Transcript excerpt: "
    )

    print(f"\n✅ Indexing complete: {num_summaries} summaries, {num_transcripts} transcripts")
    print(f"DB located at: {chroma_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
