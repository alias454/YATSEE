#!/usr/bin/env python3
"""
yatsee_indexer.py

Stage 6 of the YATSEE pipeline: index city council transcripts and summary
documents into a vector database for semantic search and analysis.

Inputs:
  - Summary files (.md) located in the entity summarization directory
  - Transcript files (.txt) located in the normalized transcript directory
  - Optional segment JSONL files (.segments.jsonl) for timestamp alignment
  - Global and entity-specific TOML configurations

Outputs:
  - Upserts document chunks into a ChromaDB collection with:
      * Embedded vector representation
      * Metadata including source, category, type, date, video_id, chunk index
      * Named Entity Recognition fields (ORG, MONEY, PERSON) for transcripts
      * Optional timestamp alignment to segments

Key Features:
  - Config-driven: merges entity-specific settings with global defaults
  - Sentence-aware chunking with configurable chunk size and overlap
  - Prepend optional text to chunks (e.g., "Transcript excerpt: ")
  - Batch embedding via SentenceTransformer for efficiency
  - Vectorized segment alignment using cosine similarity
  - Skips empty files/chunks and handles missing segment files gracefully
  - CLI interface for entity selection, model specification, and indexing options

Dependencies:
  - Python 3 standard libraries: os, sys, json, re, argparse, logging
  - Third-party: chromadb, numpy, spacy, toml, sentence_transformers, tqdm

Example Usage:
  ./yatsee_indexer.py -e us_ca_fresno_city_council --config yatsee.toml \
      --model BAAI/bge-small-en-v1.5 --spacy-model en_core_web_md \
      --summary_chunk 300 --summary_overlap 50 \
      --transcript_chunk 150 --transcript_overlap 25
"""

# Standard library
import argparse
import gc
import json
import logging
import os
import re
import sys
import textwrap
from typing import Any, Dict, List, Tuple

# Third-party imports
import chromadb
import numpy as np
import spacy
import torch
import toml
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def logger_setup(cfg: dict) -> logging.Logger:
    """
    Configure the root logger using settings from a configuration dictionary.

    Looks for the following keys in `cfg`:
      - "log_level": Logging level as a string (e.g., "INFO", "DEBUG"). Defaults to "INFO".
      - "log_format": Logging format string. Defaults to "%(asctime)s %(levelname)s %(name)s: %(message)s".

    Initializes basic logging configuration and returns a logger instance
    for the calling module.

    :param cfg: Dictionary containing logging configuration
    :return: Configured logger instance
    """
    log_level = cfg.get("log_level", "INFO")
    log_format = cfg.get("log_format", "%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.basicConfig(format=log_format, level=log_level)
    return logging.getLogger(__name__)


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


def discover_files(input_path: str, supported_exts, exclude_suffix: str = None) -> List[str]:
    """
    Collect files from a directory or single file based on allowed extensions.

    :param input_path: Path to directory or file
    :param supported_exts: Supported file extensions tuple (e.g., ".vtt, .txt")
    :param exclude_suffix: Optional suffix to exclude from results
    :return: Sorted list of valid file paths
    :raises FileNotFoundError: If path does not exist
    :raises ValueError: If single file has unsupported extension
    """
    files: List[str] = []

    if os.path.isdir(input_path):
        for f in os.listdir(input_path):
            full = os.path.join(input_path, f)
            if (
                os.path.isfile(full)
                and f.lower().endswith(supported_exts)
                and (exclude_suffix is None or not f.lower().endswith(exclude_suffix))
            ):
                files.append(full)
    elif os.path.isfile(input_path):
        if input_path.lower().endswith(supported_exts) and (
            exclude_suffix is None or not input_path.lower().endswith(exclude_suffix)
        ):
            files.append(input_path)
        else:
            raise ValueError(f"Unsupported file extension or excluded: {os.path.basename(input_path)}")
    else:
        raise FileNotFoundError(f"Path not found: {input_path}")

    return sorted(files)


def clear_gpu_cache() -> None:
    """
    Clear PyTorch GPU cache and trigger garbage collection.

    Useful to reduce memory pressure when transcribing large audio files on CUDA devices.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


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


def index_summaries(files: list, nlp_model, embedder, collection, id_map: dict, chunk_size: int, overlap: int, prepend_text: str = "") -> int:
    """
    Index a list of summary text files into ChromaDB with sentence-based chunking.

    Each file is split into chunks based on sentence boundaries, optionally prepended with
    custom text, and embedded using the provided embedding model. Metadata for each chunk
    includes the original source file and a chunk index.

    Notes:
    - Empty files or empty chunks are skipped.
    - Embeddings are computed in batch for efficiency.
    - No segment alignment or NER is performed for summaries.

    :param files: List of file paths containing summary text
    :param nlp_model: SpaCy model used for sentence tokenization
    :param embedder: SentenceTransformer model used for embeddings
    :param collection: ChromaDB collection where documents will be upserted
    :param id_map: Dictionary mapping video IDs or sources for metadata enrichment
    :param chunk_size: Maximum number of words per chunk
    :param overlap: Number of words to overlap between consecutive chunks
    :param prepend_text: Optional text to prepend to each chunk
    :return: Total number of chunks indexed
    """
    total_indexed = 0

    for path in tqdm(files, desc="Indexing summaries", file=sys.stdout, disable=not sys.stdout.isatty()):
        filename = os.path.basename(path)
        base_meta = parse_metadata(filename, "summary", id_map)
        ids, docs, metas = [], [], []

        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        if not text:
            continue

        chunks = split_text_sentences(text, nlp_model, chunk_size, overlap)

        for i, chunk in enumerate(chunks):
            doc_meta = base_meta.copy()
            doc_meta["chunk_index"] = i

            ids.append(f"SUMMARY_{filename}_{i}")
            docs.append(f"{prepend_text}{chunk}" if prepend_text else chunk)
            metas.append(doc_meta)

        if docs:
            embeddings = embedder.encode(docs, normalize_embeddings=True, show_progress_bar=False).tolist()
            collection.upsert(ids=ids, documents=docs, embeddings=embeddings, metadatas=metas)
            total_indexed += len(docs)

    return total_indexed


def index_transcripts(files: list, nlp_model, embedder, collection, id_map: dict, chunk_size: int, overlap: int, prepend_text: str = "", segments_dir: str = None) -> int:
    """
   Index transcript files into ChromaDB with top-ranked segment alignment.

    - Splits transcript into overlapping chunks
    - Computes embeddings for chunks
    - Aligns chunks to precomputed segment embeddings using top match
    - Adds NER metadata for chunks > 10 words
    - Inserts into ChromaDB collection

    :param files: List of transcript file paths
    :param nlp_model: SpaCy model for sentence splitting and NER
    :param embedder: SentenceTransformer model for embeddings
    :param collection: ChromaDB collection for storage
    :param id_map: Video ID mapping dictionary for metadata enrichment
    :param chunk_size: Maximum number of words per chunk
    :param overlap: Number of words to overlap between chunks
    :param prepend_text: Optional text to prepend to each chunk
    :param segments_dir: Directory containing *_segments.jsonl files for timestamp alignment
    :return: Total number of chunks indexed
    """
    total_indexed = 0

    for path in tqdm(files, desc="Indexing transcripts", file=sys.stdout, disable=not sys.stdout.isatty()):
        filename = os.path.basename(path)
        base_meta = parse_metadata(filename, "raw_transcript", id_map)
        ids, docs, metas = [], [], []

        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        if not text:
            continue

        chunks = split_text_sentences(text, nlp_model, chunk_size, overlap)
        if not chunks:
            continue

        chunk_embeddings = embedder.encode(chunks, normalize_embeddings=True, show_progress_bar=False)
        chunk_embs_arr = np.array(chunk_embeddings, dtype=np.float32)
        chunk_norms = np.linalg.norm(chunk_embs_arr, axis=1, keepdims=True)
        chunk_norms[chunk_norms == 0] = 1e-10

        segment_list = []
        if segments_dir:
            stem = os.path.splitext(filename)[0]
            seg_path = os.path.join(segments_dir, f"{stem}.segments.jsonl")
            if os.path.exists(seg_path):
                with open(seg_path, "r", encoding="utf-8") as f:
                    for line in f:
                        row = json.loads(line)
                        seg_emb = row.get("embedding")
                        if seg_emb is not None:
                            segment_list.append({"embedding": seg_emb, "ts": row.get("start_time")})

        if segment_list:
            seg_embs = np.array([s["embedding"] for s in segment_list], dtype=np.float32)
            seg_norms = np.linalg.norm(seg_embs, axis=1)
            seg_norms[seg_norms == 0] = 1e-10
            similarity_matrix = np.dot(chunk_embs_arr, seg_embs.T) / (chunk_norms * seg_norms)
            # Pick top match for every chunk
            best_indices = np.argmax(similarity_matrix, axis=1)
        else:
            best_indices = [None] * len(chunks)

        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue

            doc_meta = base_meta.copy()
            doc_meta["chunk_index"] = i
            # Always assign TS from top-ranked segment if available
            doc_meta["ts"] = segment_list[best_indices[i]]["ts"] if best_indices[i] is not None else None

            if len(chunk.split()) > 10:
                ner_doc = nlp_model(chunk)
                orgs = {e.text for e in ner_doc.ents if e.label_ == "ORG"}
                money = {e.text for e in ner_doc.ents if e.label_ == "MONEY"}
                persons = {e.text for e in ner_doc.ents if e.label_ == "PERSON"}
                if orgs: doc_meta["orgs"] = ", ".join(orgs)
                if money: doc_meta["money"] = ", ".join(money)
                if persons: doc_meta["persons"] = ", ".join(persons)

            ids.append(f"TRANSCRIPT_{filename}_{i}")
            docs.append(chunk if not prepend_text else f"{prepend_text}{chunk}")
            metas.append(doc_meta)

        if docs:
            collection.upsert(ids=ids, documents=docs, embeddings=chunk_embs_arr[:len(docs)], metadatas=metas)
            total_indexed += len(docs)

    return total_indexed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Index city council summaries and transcripts into a vector database using ChromaDB and SentenceTransformers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Requirements:
              - Python 3.10+
              - chromadb
              - numpy
              - spacy
              - toml
              - sentence_transformers
              - tqdm
              - os, sys, json, re, argparse, logging (standard library)

            Usage Examples:
              python yatsee_indexer.py -e defined_entity
              python yatsee_indexer.py --model BAAI/bge-small-en-v1.5 --spacy-model en_core_web_md
              python yatsee_indexer.py -e defined_entity --summary_chunk 500 --summary_overlap 50
              python yatsee_indexer.py --transcript_chunk 150 --transcript_overlap 25 --force
        """)
    )
    parser.add_argument(
        "-d", "--device",
        choices=["auto", "cuda", "cpu", "mps"],
        default="auto",
        help="Device for model execution: 'cuda' for NVIDIA GPU, 'mps' for Apple Silicon Support, 'cpu' for compatibility. Default is 'auto'"
    )
    parser.add_argument("-e", "--entity", help="Entity handle to process")
    parser.add_argument("-c", "--config", default="yatsee.toml", help="Path to global yatsee.toml")
    parser.add_argument("-m", "--model", help="SentenceTransformer model name for embeddings")
    parser.add_argument("-s", "--spacy-model", type=str, help="spaCy model to use for sentence splitting and NER (e.g., en_core_web_md)")
    parser.add_argument("--summary_chunk", type=int, default=300, help="Maximum number of words per summary chunk")
    parser.add_argument("--summary_overlap", type=int, default=50, help="Number of words to overlap between summary chunks")
    parser.add_argument("--transcript_chunk", type=int, default=150, help="Maximum number of words per transcript chunk")
    parser.add_argument("--transcript_overlap", type=int, default=25, help="Number of words to overlap between transcript chunks")
    parser.add_argument("--force", action="store_true", help="Force reindexing even if the database already exists")
    args = parser.parse_args()

    # Determine input/output paths
    entity_cfg = {}
    if args.entity:
        # Load entity config
        try:
            global_cfg = load_global_config(args.config)
            entity_cfg = load_entity_config(global_cfg, args.entity)
        except Exception as e:
            logging.error("Config load failed: %s", e)
            return 1

    # Set up custom logger
    logger = logger_setup(global_cfg.get("system", {}))

    # -----------------------------------------
    # Validate device for SentenceTransformer
    # -----------------------------------------
    if torch.cuda.is_available() and args.device in ["auto", "cuda"]:
        logger.debug("Cleared GPU cache")
        clear_gpu_cache()
        device = "cuda"
    elif torch.backends.mps.is_available() and args.device in ["auto", "mps"]:
        # Apple Silicon Support
        device = "mps"
    else:
        if args.device == "cuda":
            logger.warning("CUDA requested but not available, falling back to CPU.")
        if args.device == "mps":
            logger.warning("MPS requested but not available, falling back to CPU.")
        device = "cpu"

    # Only create the jsonl if using embeddings
    embedding_model = (
        args.model
        or entity_cfg.get("embedding_model")
        or global_cfg.get("system", {}).get("default_embedding_model", "BAAI/bge-small-en-v1.5")
    )
    logger.info("Loading SentenceTransformer model %s...", embedding_model)
    embedder = SentenceTransformer(embedding_model, device=device)

    # Load spaCy model
    spacy_model_name = (
        args.spacy_model
        or entity_cfg.get("sentence_model")
        or global_cfg.get("system", {}).get("default_sentence_model", "en_core_web_sm")
    )
    if not spacy_model_name:
        logger.error("No spaCy model specified from CLI, entity config, or system config.")
        return 1

    try:
        spacy_model = spacy.load(spacy_model_name)
        logger.info("Using spaCy model: %s", spacy_model_name)
    except OSError:
        logger.error("spaCy model '%s' not found. Install with: python -m spacy download %s",spacy_model_name, spacy_model_name)
        return 1

    download_tracker = os.path.join(entity_cfg["data_path"], "downloads", ".downloaded")
    id_map = load_video_id_map(download_tracker)
    logger.info("Loaded %d video IDs", len(id_map))

    chroma_path = os.path.join(entity_cfg["data_path"], "yatsee_db")
    # Disable anonymous telemetry https://docs.trychroma.com/docs/overview/oss#telemetry
    client = chromadb.PersistentClient(path=chroma_path, settings=Settings(anonymized_telemetry=False))
    collection = client.get_or_create_collection(name="council_knowledge", metadata={"hnsw:space": "cosine"})

    # ------------------
    # Ingest summaries
    # ------------------
    # Determine input/output directory based on entity or CLI override
    summary_dir = os.path.join(entity_cfg["data_path"], global_cfg["models"][entity_cfg["summarization_model"]]["append_dir"])
    summary_files = discover_files(summary_dir,".md")
    if not summary_files:
        logger.error("Summary directory does not exist: %s", summary_dir)
        return 1

    logger.info("Indexing %d summaries", len(summary_files))
    num_summaries = index_summaries(
        summary_files,
        spacy_model,
        embedder,
        collection,
        id_map,
        chunk_size=args.summary_chunk,
        overlap=args.summary_overlap
    )

    # ------------------
    # Ingest transcripts
    # ------------------
    # Find embeddings to be used for ts alignment if exist
    segments_dir = os.path.join(entity_cfg["data_path"], f"transcripts_{entity_cfg.get('transcription_model', 'small')}")
    segment_files = discover_files(segments_dir, ".jsonl")
    if not segment_files:
        segments_dir = ""
        logger.warning("No JSONL segments found in %s", segments_dir)

    # Determine input/output directory based on entity or CLI override
    transcript_dir = os.path.join(entity_cfg["data_path"], "normalized")
    transcript_files = discover_files(transcript_dir, ".txt")
    if not transcript_files:
        logger.error("Transcript directory does not exist: %s", transcript_dir)
        return 1

    logger.info("Indexing %d transcripts", len(transcript_files))
    num_transcripts = index_transcripts(
        transcript_files,
        spacy_model,
        embedder,
        collection,
        id_map,
        chunk_size=args.transcript_chunk,
        overlap=args.transcript_overlap,
        prepend_text="Transcript excerpt: ",
        segments_dir=segments_dir
    )

    logger.info("Indexing complete: %d summaries, %d transcripts", num_summaries, num_transcripts)
    logger.info("DB located at: %s", chroma_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
