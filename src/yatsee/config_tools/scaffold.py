"""
Entity scaffold generation for YATSEE.

This module reads the global registry and materializes local entity directories
and comment-rich config.toml scaffolds.

It intentionally avoids:
- overwriting existing config.toml files
- modifying the global registry
- inferring downstream content automatically
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import tomlkit
from tomlkit import comment, document, nl, table

from yatsee.core.errors import ConfigError, EntityNotFoundError
from yatsee.core.paths import get_entity_config_path, get_entity_dir, get_root_data_dir


def ensure_directory(path: str) -> Tuple[bool, str]:
    """
    Ensure a directory exists on disk.

    :param path: Directory path
    :return: Tuple of (created, status message)
    :raises ConfigError: If directory creation fails
    """
    if os.path.isdir(path):
        return False, f"Directory already exists: {path}"

    try:
        os.makedirs(path, exist_ok=True)
    except OSError as exc:
        raise ConfigError(f"Failed to create directory '{path}': {exc}") from exc

    return True, f"Created directory: {path}"


def build_entity_skeleton(entity: str, inputs: List[str], system_cfg: Dict[str, Any]) -> tomlkit.TOMLDocument:
    """
    Build a comment-rich entity config.toml skeleton using TOMLKit.

    Civic entities receive divisions/titles/people/replacements scaffolds.
    Non-civic media entities receive participant/alias scaffolds.

    :param entity: Entity handle
    :param inputs: Declared inputs such as ['youtube']
    :param system_cfg: Global system config dictionary
    :return: TOML document ready to write to disk
    """
    doc = document()

    settings_tbl = table()
    settings_tbl.add("entity_type", "unknown")
    settings_tbl.add("entity_level", "unknown")
    settings_tbl.add("location", "")
    settings_tbl.add("data_path", os.path.join(system_cfg.get("root_data_dir", "./data"), entity))
    settings_tbl.add(comment('summarization_model = "override default if needed"'))
    settings_tbl.add(comment('transcription_model = "override default if needed"'))
    settings_tbl.add(comment('diarization_model = "override default if needed"'))
    settings_tbl.add(comment('sentence_model = "override default if needed"'))
    settings_tbl.add(comment('embedding_model = "override default if needed"'))
    settings_tbl.add("notes", "")
    doc.add("settings", settings_tbl)

    normalized_inputs = [item.lower() for item in inputs]
    if "youtube" in normalized_inputs:
        sources_tbl = table()
        youtube_tbl = table()
        youtube_tbl.add("youtube_path", "")
        youtube_tbl.add("enabled", True)
        sources_tbl.add("youtube", youtube_tbl)
        doc.add("sources", sources_tbl)

    if "city_council" in entity.lower() or "county_board" in entity.lower():
        settings_tbl["entity_type"] = "city_council"
        settings_tbl["entity_level"] = "city"

        doc.add(nl())
        doc.add(comment("Hotword aliases are split into simple name parts (e.g. first name, last name, nickname)."))
        doc.add(comment("Avoid full phrases to keep hotword list concise and reduce input size pressure."))
        doc.add(comment("Faster-Whisper expects comma-separated phrases without weights."))

        doc.add(nl())
        doc.add(comment("Use 'divisions' as the normalized key for wards, districts, precincts, etc."))

        divisions_tbl = table()
        divisions_tbl.add("type", "wards")
        divisions_tbl.add("names", [])
        doc.add("divisions", divisions_tbl)

        titles_tbl = table()
        titles_tbl.add(comment("intentionally empty"))
        doc.add("titles", titles_tbl)

        people_tbl = table()
        people_tbl.add(comment("intentionally empty"))
        doc.add("people", people_tbl)

        replacements_tbl = table()
        replacements_tbl.add(comment("Format: 'Bad Spelling' = 'Correct Spelling'"))
        doc.add("replacements", replacements_tbl)
    else:
        settings_tbl["entity_type"] = "online_channel"
        settings_tbl["entity_level"] = "standard"

        participants_tbl = table()
        participants_tbl.add(comment("populated later via discovery"))
        doc.add("participants", participants_tbl)

        aliases_tbl = table()
        aliases_tbl.add(comment("optional"))
        doc.add("aliases", aliases_tbl)

    return doc


def create_entity_config(global_cfg: Dict[str, Any], entity: str) -> str:
    """
    Create a local config.toml for a single entity if missing.

    :param global_cfg: Global configuration dictionary
    :param entity: Entity handle
    :return: Status message
    :raises EntityNotFoundError: If entity is not in the global registry
    :raises ConfigError: If file writing fails
    """
    entities = global_cfg.get("entities", {})
    if entity not in entities:
        raise EntityNotFoundError(f"Entity '{entity}' not defined in global config")

    entity_cfg = entities[entity]
    entity_dir = get_entity_dir(global_cfg, entity)
    ensure_directory(entity_dir)

    config_path = get_entity_config_path(global_cfg, entity)
    if os.path.isfile(config_path):
        return f"Skipped {entity}: config.toml already exists"

    system_cfg = global_cfg.get("system", {})
    doc = build_entity_skeleton(
        entity=entity,
        inputs=entity_cfg.get("inputs", []),
        system_cfg=system_cfg,
    )

    try:
        with open(config_path, "w", encoding="utf-8") as handle:
            handle.write(tomlkit.dumps(doc))
    except OSError as exc:
        raise ConfigError(f"Failed to write entity config '{config_path}': {exc}") from exc

    return f"Created config.toml for {entity}"


def build_entity_structure(global_cfg: Dict[str, Any], entity: str | None = None) -> List[str]:
    """
    Create local entity directories and config.toml scaffolds.

    If an entity is provided, only that entity is initialized.
    Otherwise, all registered entities are processed.

    :param global_cfg: Global configuration dictionary
    :param entity: Optional entity handle
    :return: List of status messages
    """
    messages: List[str] = []

    root_dir = get_root_data_dir(global_cfg)
    _, msg = ensure_directory(root_dir)
    messages.append(msg)

    entities = global_cfg.get("entities", {})
    if entity is not None:
        if entity not in entities:
            raise EntityNotFoundError(f"Entity '{entity}' not defined in global config")
        target_entities = [entity]
    else:
        target_entities = sorted(entities.keys())

    for handle in target_entities:
        _, dir_msg = ensure_directory(os.path.join(root_dir, handle))
        messages.append(f"{handle}: {dir_msg}")

        try:
            cfg_msg = create_entity_config(global_cfg, handle)
            messages.append(f"{handle}: {cfg_msg}")
        except Exception as exc:
            messages.append(f"{handle}: Failed to initialize config - {exc}")

    return messages