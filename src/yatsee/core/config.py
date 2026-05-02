"""
Configuration loading and runtime resolution for YATSEE.

This module preserves the current runtime merge behavior used by existing
pipeline scripts while adding TOMLKit-based document editing helpers for
global registry changes. TOMLKit is used for writes so comments and general
file structure are preserved as much as possible.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional

import toml
import tomlkit
from tomlkit.items import Table

from yatsee.core.errors import ConfigNotFoundError, ConfigError, EntityNotFoundError
from yatsee.core.paths import get_entity_config_path, get_root_data_dir

GLOBAL_CONFIG_PATH = "yatsee.toml"
RESERVED_LOCAL_KEYS = {"settings", "meta"}


def load_global_config(path: str = GLOBAL_CONFIG_PATH) -> Dict[str, Any]:
    """
    Load the global YATSEE configuration file as a plain dictionary.

    This is used for runtime resolution and validation logic. TOMLKit document
    editing is handled by separate helpers to preserve human-authored layout
    when modifying yatsee.toml.

    :param path: Path to the global yatsee.toml
    :return: Parsed global configuration dictionary
    :raises ConfigNotFoundError: If the file does not exist
    :raises ConfigError: If TOML parsing fails
    """
    try:
        return toml.load(path)
    except FileNotFoundError as exc:
        raise ConfigNotFoundError(f"Global configuration file not found: {path}") from exc
    except Exception as exc:
        raise ConfigError(f"Failed to parse global config '{path}': {exc}") from exc


def load_global_config_document(path: str = GLOBAL_CONFIG_PATH) -> tomlkit.TOMLDocument:
    """
    Load the global YATSEE configuration as a TOMLKit document.

    This preserves comments, formatting intent, and table structure so that
    registry edits can update the existing file without flattening it into
    plain TOML output.

    :param path: Path to the global yatsee.toml
    :return: Parsed TOMLKit document
    :raises ConfigNotFoundError: If the file does not exist
    :raises ConfigError: If TOML parsing fails
    """
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return tomlkit.parse(handle.read())
    except FileNotFoundError as exc:
        raise ConfigNotFoundError(f"Global configuration file not found: {path}") from exc
    except Exception as exc:
        raise ConfigError(f"Failed to parse global config document '{path}': {exc}") from exc


def save_global_config_document(doc: tomlkit.TOMLDocument, path: str = GLOBAL_CONFIG_PATH) -> None:
    """
    Save a TOMLKit global configuration document to disk.

    :param doc: TOMLKit document to persist
    :param path: Destination path for yatsee.toml
    :raises ConfigError: If writing fails
    """
    try:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(tomlkit.dumps(doc))
    except Exception as exc:
        raise ConfigError(f"Failed to write global config '{path}': {exc}") from exc


def list_entities(global_cfg: Dict[str, Any]) -> List[str]:
    """
    Return sorted entity handles from the global registry.

    :param global_cfg: Global configuration dictionary
    :return: Sorted list of entity handles
    """
    return sorted(global_cfg.get("entities", {}).keys())


def get_entity_registry_entry(global_cfg: Dict[str, Any], entity: str) -> Dict[str, Any]:
    """
    Fetch a single entity record from the global registry.

    :param global_cfg: Global configuration dictionary
    :param entity: Entity handle
    :return: Entity registry dictionary
    :raises EntityNotFoundError: If the entity does not exist
    """
    entities = global_cfg.get("entities", {})
    if entity not in entities:
        raise EntityNotFoundError(f"Entity '{entity}' not defined in global config")
    return entities[entity]


def upsert_entity_registry_entry(
    config_path: str,
    display_name: str,
    entity: str,
    base: str = "",
    inputs: Optional[List[str]] = None,
) -> None:
    """
    Insert or replace a single entity entry inside the existing global config document.

    This edits only the [entities.<handle>] subtree and preserves the rest of the
    human-authored TOML file structure as much as TOMLKit allows.

    :param config_path: Path to yatsee.toml
    :param display_name: Human-friendly display name
    :param entity: Entity handle
    :param base: Optional base namespace
    :param inputs: Optional list of declared inputs
    :raises ConfigError: If the document cannot be updated
    """
    doc = load_global_config_document(config_path)

    if "entities" not in doc:
        doc["entities"] = tomlkit.table()

    entities_table = doc["entities"]
    entity_table = tomlkit.table()
    entity_table["display_name"] = display_name
    entity_table["base"] = base
    entity_table["entity"] = entity
    entity_table["inputs"] = inputs or []

    entities_table[entity] = entity_table
    save_global_config_document(doc, config_path)


def remove_entity_registry_entry(config_path: str, entity: str) -> None:
    """
    Remove a single entity entry from the existing global config document.

    :param config_path: Path to yatsee.toml
    :param entity: Entity handle
    :raises EntityNotFoundError: If the entity is not present
    :raises ConfigError: If the document cannot be updated
    """
    doc = load_global_config_document(config_path)

    if "entities" not in doc or entity not in doc["entities"]:
        raise EntityNotFoundError(f"Entity '{entity}' does not exist.")

    del doc["entities"][entity]
    save_global_config_document(doc, config_path)


def load_local_entity_config(global_cfg: Dict[str, Any], entity: str) -> Dict[str, Any]:
    """
    Load an entity's local config.toml file.

    :param global_cfg: Global configuration dictionary
    :param entity: Entity handle
    :return: Parsed local entity configuration dictionary
    :raises ConfigNotFoundError: If local config.toml is missing
    :raises ConfigError: If TOML parsing fails
    """
    local_path = get_entity_config_path(global_cfg, entity)

    try:
        return toml.load(local_path)
    except FileNotFoundError as exc:
        raise ConfigNotFoundError(
            f"Local config for entity '{entity}' not found at: {local_path}"
        ) from exc
    except Exception as exc:
        raise ConfigError(f"Failed to parse local config '{local_path}': {exc}") from exc


def load_entity_config(global_cfg: Dict[str, Any], entity: str) -> Dict[str, Any]:
    """
    Load and merge entity-specific configuration with global defaults.

    This preserves the current merge contract used by the existing pipeline
    scripts. Reserved local sections ('settings', 'meta') are flattened into
    the merged runtime config. Other local sections remain nested.

    :param global_cfg: Global configuration dictionary
    :param entity: Entity handle
    :return: Flattened merged runtime configuration dictionary
    :raises EntityNotFoundError: If the entity is not defined
    :raises ConfigNotFoundError: If local config.toml is missing
    :raises ConfigError: If parsing fails
    """
    entity_registry = get_entity_registry_entry(global_cfg, entity)
    local_cfg = load_local_entity_config(global_cfg, entity)

    system_cfg = global_cfg.get("system", {})
    root_data_dir = get_root_data_dir(global_cfg)

    merged: Dict[str, Any] = {
        "entity": entity,
        "root_data_dir": root_data_dir,
        **deepcopy(system_cfg),
        **deepcopy(entity_registry),
    }

    for key, value in local_cfg.items():
        if key in RESERVED_LOCAL_KEYS and isinstance(value, dict):
            merged.update(deepcopy(value))
        else:
            merged[key] = deepcopy(value)

    return merged


def resolve_runtime_config(
    global_config_path: str = GLOBAL_CONFIG_PATH,
    entity: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Resolve runtime configuration for either global-only or entity-aware usage.

    If no entity is provided, this returns only the parsed global config.
    If an entity is provided, this returns the merged runtime config used by
    stage scripts.

    :param global_config_path: Path to global yatsee.toml
    :param entity: Optional entity handle
    :return: Resolved configuration dictionary
    """
    global_cfg = load_global_config(global_config_path)
    if entity is None:
        return global_cfg
    return load_entity_config(global_cfg, entity)