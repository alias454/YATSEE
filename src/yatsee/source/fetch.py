"""
Entity-driven source fetch orchestration for YATSEE.

This module inspects configured entity inputs and dispatches to supported
source adapters. The initial implementation supports YouTube only.
"""

from __future__ import annotations

from typing import Any, Dict, List

from yatsee.core.config import load_entity_config, load_global_config
from yatsee.core.errors import ValidationError
from yatsee.source.youtube import run_youtube_fetch


def run_source_fetch_for_entity(
    global_config_path: str,
    entity: str,
    source_name: str | None = None,
    output_dir: str | None = None,
    date_after: str | None = None,
    date_before: str | None = None,
    make_playlist: bool = False,
) -> Dict[str, Any]:
    """
    Fetch source artifacts for an entity by dispatching configured adapters.

    If source_name is provided, only that source is dispatched.
    Otherwise, the entity's declared inputs are used as the dispatch list.

    :param global_config_path: Path to global yatsee.toml
    :param entity: Entity handle
    :param source_name: Optional specific source to run
    :param output_dir: Optional output directory override
    :param date_after: Optional lower date bound in YYYYMMDD
    :param date_before: Optional upper date bound in YYYYMMDD
    :param make_playlist: Rebuild playlist cache and exit
    :return: Summary dictionary
    :raises ValidationError: If no supported source can be dispatched
    """
    global_cfg = load_global_config(global_config_path)
    entity_cfg = load_entity_config(global_cfg, entity)

    declared_inputs = entity_cfg.get("inputs", [])
    if not isinstance(declared_inputs, list):
        raise ValidationError(f"Entity '{entity}' has invalid 'inputs'; expected a list.")

    requested_sources: List[str]
    if source_name:
        requested_sources = [source_name.strip().lower()]
    else:
        requested_sources = [str(item).strip().lower() for item in declared_inputs if str(item).strip()]

    if not requested_sources:
        raise ValidationError(f"Entity '{entity}' declares no usable inputs.")

    messages: List[str] = []
    adapter_results: List[Dict[str, Any]] = []

    for source in requested_sources:
        if source == "youtube":
            result = run_youtube_fetch(
                global_config_path=global_config_path,
                entity=entity,
                output_dir=output_dir,
                date_after=date_after,
                date_before=date_before,
                make_playlist=make_playlist,
            )
            adapter_results.append(result)
            messages.extend(result["messages"])
            continue

        raise ValidationError(
            f"Input type '{source}' is declared for entity '{entity}' but no source adapter is installed."
        )

    return {
        "entity": entity,
        "sources_run": [item["source_type"] for item in adapter_results],
        "results": adapter_results,
        "messages": messages,
    }