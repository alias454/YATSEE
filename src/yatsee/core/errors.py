"""
Project-specific exceptions for YATSEE.

These exceptions make failure modes explicit and keep CLI error handling simple.
"""


class YatseeError(Exception):
    """
    Base exception for YATSEE-specific failures.
    """


class ConfigError(YatseeError):
    """
    Raised when configuration files are missing, malformed, or inconsistent.
    """


class ConfigNotFoundError(ConfigError):
    """
    Raised when a required configuration file cannot be found.
    """


class EntityNotFoundError(ConfigError):
    """
    Raised when an entity handle is not present in the global registry.
    """


class ValidationError(YatseeError):
    """
    Raised when configuration validation fails.
    """