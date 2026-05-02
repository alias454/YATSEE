"""
Entity registry management for YATSEE.

This module manages the global entity registry stored in yatsee.toml.
It separates safe registry operations from destructive filesystem purging.

Command intent is split intentionally:

- remove: remove only the registry entry from yatsee.toml
- purge: remove the registry entry and delete the entity data directory

This keeps routine config cleanup safe while still allowing explicit removal
of on-disk artifacts when requested.
"""

from __future__ import annotations

import os
import re
import shutil
from typing import Any, Dict, List

from yatsee.core.config import remove_entity_registry_entry, upsert_entity_registry_entry
from yatsee.core.errors import ConfigError, EntityNotFoundError, ValidationError
from yatsee.core.paths import get_entity_dir, get_root_data_dir


def list_entities(global_cfg: Dict[str, Any]) -> List[str]:
    """
    Return sorted entity handles from the global registry.

    :param global_cfg: Global configuration dictionary
    :return: Sorted list of entity handles
    """
    return sorted(global_cfg.get("entities", {}).keys())


def validate_entity_handle(entity: str) -> str:
    """
    Validate an entity handle against the project's current naming rules.

    Allowed characters:
    - lowercase letters
    - numbers
    - underscores

    :param entity: Proposed entity handle
    :return: Normalized entity handle
    :raises ValidationError: If the handle is invalid
    """
    entity = entity.strip()
    if not re.fullmatch(r"[a-z0-9_]+", entity):
        raise ValidationError(
            "Invalid entity handle. Only lowercase letters, numbers, and underscores are allowed."
        )
    return entity


def ensure_root_data_dir(global_cfg: Dict[str, Any]) -> str:
    """
    Ensure the configured root data directory exists.

    :param global_cfg: Global configuration dictionary
    :return: Absolute path to the root data directory
    :raises ConfigError: If directory creation fails
    """
    root_dir = get_root_data_dir(global_cfg)
    try:
        os.makedirs(root_dir, exist_ok=True)
    except OSError as exc:
        raise ConfigError(f"Failed to create root data directory '{root_dir}': {exc}") from exc
    return root_dir


def create_entity_directory(global_cfg: Dict[str, Any], entity: str) -> str:
    """
    Ensure the top-level entity directory exists under the root data dir.

    :param global_cfg: Global configuration dictionary
    :param entity: Entity handle
    :return: Path to the entity directory
    :raises ConfigError: If directory creation fails
    """
    ensure_root_data_dir(global_cfg)
    entity_dir = get_entity_dir(global_cfg, entity)
    try:
        os.makedirs(entity_dir, exist_ok=True)
    except OSError as exc:
        raise ConfigError(f"Failed to create entity directory '{entity_dir}': {exc}") from exc
    return entity_dir


def remove_entity_directory(global_cfg: Dict[str, Any], entity: str) -> str:
    """
    Remove the top-level entity directory and all contents.

    :param global_cfg: Global configuration dictionary
    :param entity: Entity handle
    :return: Removed path
    :raises ConfigError: If deletion fails
    """
    entity_dir = get_entity_dir(global_cfg, entity)
    try:
        if os.path.isdir(entity_dir):
            shutil.rmtree(entity_dir)
    except OSError as exc:
        raise ConfigError(f"Failed to remove entity directory '{entity_dir}': {exc}") from exc
    return entity_dir


def inspect_entity_storage(global_cfg: Dict[str, Any], entity: str) -> Dict[str, Any]:
    """
    Inspect the on-disk state of an entity directory without changing anything.

    This is used to support dry-run previews for destructive operations.

    :param global_cfg: Global configuration dictionary
    :param entity: Entity handle
    :return: Dictionary describing directory existence and basic contents
    """
    entity_dir = get_entity_dir(global_cfg, entity)
    exists = os.path.isdir(entity_dir)

    file_count = 0
    dir_count = 0
    config_exists = False

    if exists:
        for root, dirs, files in os.walk(entity_dir):
            dir_count += len(dirs)
            file_count += len(files)

        config_exists = os.path.isfile(os.path.join(entity_dir, "config.toml"))

    return {
        "entity": entity,
        "entity_dir": entity_dir,
        "exists": exists,
        "config_exists": config_exists,
        "file_count": file_count,
        "dir_count": dir_count,
    }


def add_entity(
    global_cfg: Dict[str, Any],
    config_path: str,
    display_name: str,
    entity: str,
    base: str = "",
    inputs: List[str] | None = None,
    create_dir: bool = True,
) -> Dict[str, str]:
    """
    Add a new entity to the global registry and persist it without flattening the file.

    This updates yatsee.toml by editing only the [entities.<handle>] subtree through
    TOMLKit. That preserves the rest of the file layout far better than dumping the
    entire config from a plain Python dictionary.

    :param global_cfg: Global configuration dictionary
    :param config_path: Path to yatsee.toml
    :param display_name: Human-friendly display name
    :param entity: Entity handle
    :param base: Optional namespace/base value
    :param inputs: Optional list of declared inputs
    :param create_dir: Create top-level entity directory if True
    :return: Summary of actions performed
    :raises ValidationError: If inputs are invalid or entity already exists
    """
    entity = validate_entity_handle(entity)
    display_name = display_name.strip()

    if not display_name:
        raise ValidationError("Display name is required.")

    entities = global_cfg.setdefault("entities", {})
    if entity in entities:
        raise ValidationError(f"Entity '{entity}' already exists.")

    normalized_inputs = [item.strip().lower() for item in (inputs or []) if item.strip()]
    base = base.strip().rstrip(".")

    upsert_entity_registry_entry(
        config_path=config_path,
        display_name=display_name,
        entity=entity,
        base=base,
        inputs=normalized_inputs,
    )

    entities[entity] = {
        "display_name": display_name,
        "base": base,
        "entity": entity,
        "inputs": normalized_inputs,
    }

    entity_dir = ""
    if create_dir:
        entity_dir = create_entity_directory(global_cfg, entity)

    return {
        "entity": entity,
        "entity_dir": entity_dir,
        "message": f"Added entity '{entity}'",
    }


def remove_entity(global_cfg: Dict[str, Any], config_path: str, entity: str) -> Dict[str, str]:
    """
    Remove an entity from the global registry only.

    This is intentionally non-destructive to on-disk entity data. It updates the
    registry but leaves the entity directory and all artifacts intact.

    :param global_cfg: Global configuration dictionary
    :param config_path: Path to yatsee.toml
    :param entity: Entity handle
    :return: Summary of actions performed
    :raises EntityNotFoundError: If the entity does not exist
    """
    entities = global_cfg.get("entities", {})
    if entity not in entities:
        raise EntityNotFoundError(f"Entity '{entity}' does not exist.")

    remove_entity_registry_entry(config_path, entity)
    del entities[entity]

    return {
        "entity": entity,
        "message": f"Removed entity '{entity}' from registry",
    }


def purge_entity(
    global_cfg: Dict[str, Any],
    config_path: str,
    entity: str,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Purge an entity from both the registry and the local filesystem.

    This is intentionally destructive and therefore separated from the normal
    registry removal command. A dry-run mode is provided so operators can inspect
    the exact blast radius before deletion.

    :param global_cfg: Global configuration dictionary
    :param config_path: Path to yatsee.toml
    :param entity: Entity handle
    :param dry_run: Preview actions without modifying config or filesystem
    :return: Summary of actions performed or that would be performed
    :raises EntityNotFoundError: If the entity does not exist
    """
    entities = global_cfg.get("entities", {})
    if entity not in entities:
        raise EntityNotFoundError(f"Entity '{entity}' does not exist.")

    storage = inspect_entity_storage(global_cfg, entity)

    if dry_run:
        return {
            "entity": entity,
            "dry_run": True,
            "registry_entry_exists": True,
            "entity_dir": storage["entity_dir"],
            "entity_dir_exists": storage["exists"],
            "config_exists": storage["config_exists"],
            "file_count": storage["file_count"],
            "dir_count": storage["dir_count"],
            "message": f"Dry run: would purge entity '{entity}'",
        }

    remove_entity_registry_entry(config_path, entity)
    del entities[entity]
    removed_path = remove_entity_directory(global_cfg, entity)

    return {
        "entity": entity,
        "dry_run": False,
        "registry_entry_exists": True,
        "entity_dir": removed_path,
        "entity_dir_exists": storage["exists"],
        "config_exists": storage["config_exists"],
        "file_count": storage["file_count"],
        "dir_count": storage["dir_count"],
        "message": f"Purged entity '{entity}'",
    }