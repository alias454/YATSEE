"""
Filesystem discovery helpers for YATSEE.

This module centralizes the common pattern of accepting either a directory
or a single file and returning a normalized list of supported files.
"""

from __future__ import annotations

import os
from typing import List, Tuple

from yatsee.core.errors import ValidationError


def discover_files(input_path: str, supported_exts: Tuple[str, ...]) -> List[str]:
    """
    Collect supported files from a directory or a single file.

    :param input_path: Directory path or single file path
    :param supported_exts: Tuple of allowed file extensions
    :return: Sorted list of matching file paths
    :raises FileNotFoundError: If input_path does not exist
    :raises ValidationError: If a single file has an unsupported extension
    """
    files: List[str] = []

    if os.path.isdir(input_path):
        for entry in os.listdir(input_path):
            full = os.path.join(input_path, entry)
            if os.path.isfile(full) and entry.lower().endswith(supported_exts):
                files.append(full)
        return sorted(files)

    if os.path.isfile(input_path):
        if input_path.lower().endswith(supported_exts):
            return [input_path]
        raise ValidationError(f"Unsupported file extension: {os.path.splitext(input_path)[1]}")

    raise FileNotFoundError(f"Path not found: {input_path}")