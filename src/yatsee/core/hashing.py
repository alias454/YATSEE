"""
Hashing helpers for YATSEE.

This module centralizes the file hashing used by multiple pipeline stages
for idempotency and change tracking.
"""

from __future__ import annotations

import hashlib

from yatsee.core.errors import ConfigError


def compute_sha256(path: str) -> str:
    """
    Compute the SHA-256 hash of a file.

    :param path: Path to the file
    :return: Hex-encoded SHA-256 hash string
    :raises ConfigError: If the file cannot be read
    """
    try:
        hasher = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except OSError as exc:
        raise ConfigError(f"Failed to hash file: {path}") from exc