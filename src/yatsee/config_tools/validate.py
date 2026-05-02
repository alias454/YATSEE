"""
Validation helpers for YATSEE configuration.

This module intentionally starts with structural validation only. Deep semantic
validation can be added later once the package CLI is stable.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

from yatsee.core.config import get_entity_registry_entry
from yatsee.core.errors import ValidationError
from yatsee.core.paths import get_entity_config_path


def validate_global_config(global_cfg: Dict[str, Any]) -> List[str]:
    """
    Validate the structural minimum of the global configuration.

    :param global_cfg: Global configuration dictionary
    :return: List of success messages
    :raises ValidationError: If required sections are missing
    """
    messages: List[str] = []

    if "system" not in global_cfg:
        raise ValidationError("Global config is missing required [system] section.")
    messages.append("Found [system] section.")

    if "entities" not in global_cfg:
        raise ValidationError("Global config is missing required [entities] section.")
    messages.append("Found [entities] section.")

    return messages


def validate_entity_config(global_cfg: Dict[str, Any], entity: str) -> List[str]:
    """
    Validate the minimum structural contract for an entity.

    :param global_cfg: Global configuration dictionary
    :param entity: Entity handle
    :return: List of success messages
    :raises ValidationError: If required files or fields are missing
    """
    messages: List[str] = []

    entity_registry = get_entity_registry_entry(global_cfg, entity)
    messages.append(f"Found entity '{entity}' in global registry.")

    config_path = get_entity_config_path(global_cfg, entity)
    if not os.path.isfile(config_path):
        raise ValidationError(f"Missing local entity config: {config_path}")
    messages.append(f"Found local entity config: {config_path}")

    required_registry_keys = {"display_name", "entity", "inputs"}
    missing_registry = sorted(required_registry_keys - set(entity_registry.keys()))
    if missing_registry:
        raise ValidationError(
            f"Entity '{entity}' is missing required registry keys: {', '.join(missing_registry)}"
        )
    messages.append("Entity registry contains required keys.")

    return messages