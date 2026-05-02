"""
Merged runtime configuration inspection for YATSEE.

This module exposes the resolved runtime view that later CLI commands and
pipeline stages will consume.
"""

from __future__ import annotations

from typing import Any, Dict

from yatsee.core.config import resolve_runtime_config


def resolve_config(global_config_path: str, entity: str | None = None) -> Dict[str, Any]:
    """
    Resolve the effective runtime configuration.

    :param global_config_path: Path to global yatsee.toml
    :param entity: Optional entity handle
    :return: Resolved configuration dictionary
    """
    return resolve_runtime_config(global_config_path=global_config_path, entity=entity)