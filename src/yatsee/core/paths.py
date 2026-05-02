"""
Path helpers for YATSEE.

This module centralizes filesystem path construction so CLI commands and later
pipeline stages do not each re-implement path logic.
"""

from __future__ import annotations

import os
from typing import Any, Dict


def get_root_data_dir(cfg: Dict[str, Any]) -> str:
    """
    Resolve the absolute root data directory from configuration.

    :param cfg: Configuration dictionary containing system settings
    :return: Absolute path to the root data directory
    """
    root = cfg.get("system", {}).get("root_data_dir", "./data")
    return os.path.abspath(root)


def get_entity_dir(cfg: Dict[str, Any], entity: str) -> str:
    """
    Resolve the absolute directory for a specific entity.

    :param cfg: Global configuration dictionary
    :param entity: Entity handle
    :return: Absolute path to the entity directory
    """
    return os.path.join(get_root_data_dir(cfg), entity)


def get_entity_config_path(cfg: Dict[str, Any], entity: str) -> str:
    """
    Resolve the absolute path to an entity's local config.toml file.

    :param cfg: Global configuration dictionary
    :param entity: Entity handle
    :return: Absolute path to the entity config.toml
    """
    return os.path.join(get_entity_dir(cfg, entity), "config.toml")