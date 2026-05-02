"""
Tracker file helpers for YATSEE.

Many pipeline stages persist a simple newline-delimited tracker file
to avoid reprocessing previously handled artifacts.
"""

from __future__ import annotations

import os
from typing import Set

from yatsee.core.errors import ConfigError


def load_tracker_set(tracker_path: str) -> Set[str]:
    """
    Load a newline-delimited tracker file into a set.

    Missing files are treated as empty state.

    :param tracker_path: Path to the tracker file
    :return: Set of non-empty tracker values
    """
    if not os.path.exists(tracker_path):
        return set()

    try:
        with open(tracker_path, "r", encoding="utf-8") as handle:
            return {line.strip() for line in handle if line.strip()}
    except OSError:
        return set()


def append_tracker_value(tracker_path: str, value: str) -> None:
    """
    Append a value to a newline-delimited tracker file.

    :param tracker_path: Path to the tracker file
    :param value: Value to append
    :raises ConfigError: If writing fails
    """
    try:
        with open(tracker_path, "a", encoding="utf-8") as handle:
            handle.write(f"{value}\n")
    except OSError as exc:
        raise ConfigError(f"Failed to update tracker '{tracker_path}'") from exc