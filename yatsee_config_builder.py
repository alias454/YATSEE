"""
YATSEE Entity Configuration Scaffold Generator

This tool reads the global YATSEE registry (yatsee.toml) and materializes
per-entity directory structures and config.toml files.

What it does:
- Creates one directory per registered entity under the root data path
- Generates a minimally populated, comment-rich config.toml for each entity
- Preserves structure, ordering, and explanatory comments using tomlkit
- Differentiates civic entities (e.g. city councils) from media/online entities
- Produces safe, non-destructive scaffolds intended for manual refinement

What it deliberately does NOT do:
- Does not overwrite existing config.toml files
- Does not enforce required fields or validation
- Does not infer people, titles, or divisions automatically
- Does not modify the global registry

Default behavior is read-only. No files or directories are created unless
--create is explicitly provided.

This script exists to make entity setup boring, predictable, and reversible.
"""

# Standard library
import argparse
import os
import sys
from typing import Any, Dict, List, Tuple

# Third-party imports
import toml
import tomlkit
from tomlkit import comment, document, nl, table

GLOBAL_CONFIG_PATH = "yatsee.toml"


def load_global_config(path: str = GLOBAL_CONFIG_PATH) -> Dict[str, Any]:
    """
    Load the global YATSEE configuration file.

    :param path: Path to yatsee.toml
    :return: Parsed TOML configuration
    :raises FileNotFoundError: If the file does not exist
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Global configuration '{path}' not found")
    return toml.load(path)


def ensure_directory(path: str) -> Tuple[bool, str]:
    """
    Ensure a directory exists on disk.

    :param path: Directory path
    :return: Tuple of (created, status message)
    """
    if os.path.isdir(path):
        return False, f"Directory already exists: {path}"
    os.makedirs(path, exist_ok=True)
    return True, f"Created directory: {path}"


def build_entity_skeleton(entity: str, inputs: list[str], system_cfg: dict[str, Any]) -> tomlkit.TOMLDocument:
    """
    Build a TOMLKit document skeleton for an entity, preserving all
    scaffolds and explanatory comments.

    - Creates [settings] with entity metadata
    - Creates [sources] for all inputs
    - Includes scaffolds for divisions, titles, people, directors, staff, replacements
    - Preserves hotword guidance and comments
    - Media entities get participants, aliases, and sources scaffolds but no civic scaffolds.

    :param entity: Slug/handle of the entity
    :param inputs: Declared input sources (e.g., ["youtube"])
    :param system_cfg: Global system config for default paths/models
    :return: TOMLKit document ready to write to disk
    """
    doc = document()
    # --- Root 'settings' table ---
    settings_tbl = table()
    settings_tbl.add("entity_type", "unknown")
    settings_tbl.add("entity_level", "unknown")
    settings_tbl.add("location", "")
    settings_tbl.add("data_path", os.path.join(system_cfg.get("root_data_dir", "./data"), entity))
    settings_tbl.add(comment('summarization_model = "override default if needed"'))
    settings_tbl.add(comment('transcription_model = "override default if needed"'))
    settings_tbl.add(comment('diarization_model = "override default if needed"'))
    settings_tbl.add(comment('sentences_model = "override default if needed"'))
    settings_tbl.add(comment('embedding_model = "override default if needed"'))
    settings_tbl.add("notes", "")
    doc.add("settings", settings_tbl)

    # --- Sources table (universal for YouTube) ---
    if "youtube" in [i.lower() for i in inputs]:
        sources_tbl = table()
        youtube_tbl = table()
        youtube_tbl.add("youtube_path", "")
        youtube_tbl.add("enabled", True)
        sources_tbl.add("youtube", youtube_tbl)
        doc.add("sources", sources_tbl)

    # Determine if civic-style entity
    if "city_council" in entity.lower() or "county_board" in entity.lower():
        settings_tbl["entity_type"] = "city_council"
        settings_tbl["entity_level"] = "city"

        # --- Insert blank line and comments for hotwords ---
        doc.add(nl())
        doc.add(comment("Hotword aliases are split into simple name parts (e.g. first name, last name, nickname)."))
        doc.add(comment("Avoid full phrases to keep hotword list concise to avoid hitting input size limits."))
        doc.add(comment("Faster-Whisper expects comma-separated phrases (no weights)."))
        doc.add(comment("Example: 'Tom, Thomas, Smith, Mayor, City Manager, Jim, Johnson'"))

        doc.add(nl())
        doc.add(comment("Instead of a fixed key like 'wards', allow any division type with a normalized name 'divisions'."))
        doc.add(comment("Can be districts, parishes, precincts, wards etc"))

        # --- Divisions scaffold ---
        divisions = table()
        divisions.add("type", "wards")
        divisions.add("names", [])
        doc.add("divisions", divisions)

        # --- Titles scaffold ---
        titles = table()
        titles.add(comment("intentionally empty"))
        doc.add("titles", titles)

        # --- People scaffold ---
        people = table()
        people.add(comment("intentionally empty"))
        doc.add("people", people)

        # --- Replacements scaffold ---
        replacements = table()
        replacements.add(comment("Format: 'Bad Spelling' = 'Correct Spelling'"))
        doc.add("replacements", replacements)

    # --- Media/YouTube entity scaffolds (non-civic) ---
    else:
        settings_tbl["entity_type"] = "online_channel"
        settings_tbl["entity_level"] = "standard"

        # Participants scaffold
        participants = table()
        participants.add(comment("populated later via discovery"))
        doc.add("participants", participants)

        # Aliases scaffold
        aliases = table()
        aliases.add(comment("optional"))
        doc.add("aliases", aliases)

    return doc


def create_entity_config(global_config: Dict[str, Any], entity_handle: str, root_dir: str) -> str:
    """
    Create a local config.toml for a single entity.

    :param global_config: Parsed global YATSEE configuration
    :param entity_handle: Handle of the entity to initialize
    :param root_dir: Root directory for entity folders
    :return: Status message indicating creation or skip
    """
    entity_cfg = global_config["entities"][entity_handle]
    entity_dir = os.path.join(root_dir, entity_handle)
    ensure_directory(entity_dir)
    config_path = os.path.join(entity_dir, "config.toml")

    if os.path.isfile(config_path):
        return f"Skipped {entity_handle}: config.toml already exists"

    system_cfg = global_config.get("system", {})
    doc = build_entity_skeleton(
        entity=entity_handle,
        inputs=entity_cfg.get("inputs", []),
        system_cfg=system_cfg
    )

    with open(config_path, "w", encoding="utf-8") as fh:
        fh.write(tomlkit.dumps(doc))

    return f"Created config.toml for {entity_handle}"


def build_entity_structure(global_config: Dict[str, Any]) -> List[str]:
    """
    Iterate over all entities in the global config and generate local
    directories and config.toml scaffolds.

    :param global_config: Parsed global YATSEE configuration
    :return: List of status messages for each entity
    """
    messages: List[str] = []
    system_cfg = global_config.get("system", {})
    root_data_dir = system_cfg.get("root_data_dir", "./data")

    _, msg = ensure_directory(root_data_dir)
    messages.append(msg)

    entities = global_config.get("entities", {})
    for handle in sorted(entities.keys()):
        _, msg = ensure_directory(os.path.join(root_data_dir, handle))
        messages.append(f"{handle}: {msg}")

        try:
            msg = create_entity_config(global_config, handle, root_data_dir)
            messages.append(f"{handle}: {msg}")
        except Exception as exc:
            messages.append(f"{handle}: ❌ Failed to initialize config - {exc}")

    return messages


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate per-entity directory structures and config.toml scaffolds "
            "from the global YATSEE registry (yatsee.toml). By default, runs in "
            "read-only mode and lists registered entities."
        )
    )
    parser.add_argument("--create", action="store_true", help="Create entity directories and scaffold config.toml files")
    args = parser.parse_args()

    try:
        global_config = load_global_config()
    except FileNotFoundError as exc:
        print(f"❌ {exc}", file=sys.stderr)
        return 1

    if not args.create:
        entities = global_config.get("entities", {})
        print("Registered Entities:")
        for name in sorted(entities.keys()):
            print(f"- {name}")
        return 0

    try:
        messages = build_entity_structure(global_config)
    except Exception as exc:
        print(f"❌ Failed: {exc}", file=sys.stderr)
        return 1

    for msg in messages:
        print(msg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
