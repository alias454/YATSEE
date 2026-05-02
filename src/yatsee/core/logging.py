"""
Logging helpers for YATSEE.

This module provides a single logger setup path so future commands can share
consistent log formatting and verbosity behavior.
"""

from __future__ import annotations

import logging
from typing import Any, Dict


def setup_logging(cfg: Dict[str, Any], verbose: bool = False, quiet: bool = False) -> logging.Logger:
    """
    Configure project logging from system config and optional CLI verbosity flags.

    CLI flags are allowed to override config defaults because operator intent
    should win for interactive commands.

    :param cfg: Configuration dictionary containing system settings
    :param verbose: Enable debug-level logging
    :param quiet: Reduce logging output to warnings/errors only
    :return: Configured module logger
    """
    system_cfg = cfg.get("system", {})
    log_level = str(system_cfg.get("log_level", "INFO")).upper()
    log_format = system_cfg.get(
        "log_format",
        "%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if verbose:
        log_level = "DEBUG"
    elif quiet:
        log_level = "WARNING"

    logging.basicConfig(level=getattr(logging, log_level, logging.INFO), format=log_format)
    return logging.getLogger("yatsee")