"""
Speaker detection and context helpers for YATSEE intelligence jobs.
"""

from __future__ import annotations

import re
from typing import Dict, List, Set


def _clean_string(value: str) -> str:
    """
    Normalize a string used in speaker permutation generation.

    This keeps whitespace cleanup explicit and avoids noisy IDE type warnings
    when chained string operations appear inside set literals or f-strings.

    :param value: Input string
    :return: Stripped string
    """
    return str(value).strip()


def _join_nonempty_parts(*parts: str) -> str:
    """
    Join non-empty string parts with single spaces.

    This is used for title/name combinations so normalization happens in one
    place instead of being repeated inline.

    :param parts: String fragments to combine
    :return: Joined and normalized string
    """
    cleaned_parts: List[str] = []

    for part in parts:
        cleaned = _clean_string(part)
        if cleaned:
            cleaned_parts.append(cleaned)

    return " ".join(cleaned_parts)


def _build_name_variant_pattern(name_variant: str) -> re.Pattern[str]:
    """
    Build a whole-name regex pattern for a single name variant.

    The variant is escaped before compilation so matching stays literal rather
    than interpreting punctuation or other characters as regex syntax.

    :param name_variant: Name variant to match
    :return: Compiled regex pattern
    """
    pattern_text = r"\b" + re.escape(name_variant.lower()) + r"\b"
    return re.compile(pattern_text)


def build_name_permutations(data_config: dict) -> dict:
    """
    Build detectable name variants for configured people.

    Variants include aliases, first name, last name, full name, and title-based
    expansions derived from the role/title configuration. The result is used to
    scan transcript chunks for likely speaker references and inject supporting
    context into prompts.

    :param data_config: Configuration containing ``people`` and ``titles`` mappings
    :return: Mapping of canonical person keys to sorted name variants
    """
    people_by_role = data_config.get("people", {})
    titles_by_role = data_config.get("titles", {})

    if not isinstance(people_by_role, dict):
        return {}

    if not isinstance(titles_by_role, dict):
        titles_by_role = {}

    name_permutations: Dict[str, List[str]] = {}

    for role_key, people in people_by_role.items():
        if not isinstance(people, dict):
            continue

        titles_for_role = titles_by_role.get(role_key, [])
        if not isinstance(titles_for_role, list):
            titles_for_role = []

        for person_key, known_aliases in people.items():
            if not isinstance(person_key, str):
                continue

            clean_person_key = _clean_string(person_key)
            if not clean_person_key:
                continue

            if not isinstance(known_aliases, list):
                known_aliases = []

            try:
                first_name, last_name = clean_person_key.split("_", 1)
            except ValueError:
                first_name = clean_person_key
                last_name = ""

            full_name = _join_nonempty_parts(first_name, last_name)

            permutations = {
                _clean_string(alias)
                for alias in known_aliases
                if isinstance(alias, str) and _clean_string(alias)
            }

            permutations.update(
                {
                    _clean_string(first_name),
                    _clean_string(last_name),
                    _clean_string(full_name),
                }
            )
            permutations.discard("")

            for title in titles_for_role:
                if not isinstance(title, str):
                    continue

                clean_title = _clean_string(title)
                if not clean_title:
                    continue

                permutations.add(_join_nonempty_parts(clean_title, last_name))
                permutations.add(_join_nonempty_parts(clean_title, full_name))

                if " " in clean_title:
                    permutations.add(
                        _join_nonempty_parts(clean_title, "Director", last_name)
                    )
                    permutations.add(
                        _join_nonempty_parts(clean_title, "Director", full_name)
                    )

            name_permutations[clean_person_key] = sorted(permutations)

    return name_permutations


def scan_transcript_for_names(
    transcript_text: str,
    name_permutations: Dict[str, List[str]],
) -> Dict[str, Set[str]]:
    """
    Scan transcript text for configured name variants.

    Matching is case-insensitive and uses word boundaries so short names do not
    match unrelated substrings inside other words.

    :param transcript_text: Transcript text to scan
    :param name_permutations: Canonical-to-variants mapping
    :return: Canonical-to-matched-variants mapping
    """
    found_matches: Dict[str, Set[str]] = {}
    lower_transcript = transcript_text.lower()

    for person_key, permutations in name_permutations.items():
        if not isinstance(permutations, list):
            continue

        matches: Set[str] = set()

        for name_variant in permutations:
            if not isinstance(name_variant, str):
                continue

            clean_variant = _clean_string(name_variant)
            if not clean_variant:
                continue

            pattern = _build_name_variant_pattern(clean_variant)
            if pattern.search(lower_transcript):
                matches.add(clean_variant)

        if matches:
            found_matches[person_key] = matches

    return found_matches


def generate_known_speakers_context(speaker_matches: Dict[str, Set[str]]) -> str:
    """
    Build prompt context describing detected speaker references in a chunk.

    The context string is designed to give downstream prompts lightweight
    grounding about which configured people were mentioned and where ambiguity
    may exist due to shorthand references such as first-name-only mentions.

    :param speaker_matches: Canonical speaker key to matched variants
    :return: Context string for prompt injection
    """
    if not speaker_matches:
        return ""

    descriptive_mentions: List[str] = []
    ambiguous_mentions: List[tuple[str, List[str]]] = []

    for canonical_name, matches in speaker_matches.items():
        sorted_matches = sorted(matches, key=len, reverse=True)
        best_variant = sorted_matches[0]
        descriptive_mentions.append(best_variant)

        if len(matches) > 1:
            ambiguous_mentions.append((canonical_name, sorted_matches))

        first_name_only = [
            match
            for match in matches
            if len(match.split()) == 1 and match.lower() not in canonical_name.lower()
        ]
        if first_name_only:
            ambiguous_mentions.append((canonical_name, first_name_only))

    mentions_string = ", ".join(descriptive_mentions)
    context_lines = [f"**Speakers detected in this segment: {mentions_string}.**"]

    if ambiguous_mentions:
        context_lines.append("*Some mentions are ambiguous or used multiple name variants:*")
        for canonical_name, variants in ambiguous_mentions:
            readable_name = canonical_name.replace("_", " ")
            context_lines.append(f"  - {readable_name}: {', '.join(variants)}")

    context_lines.append("*Mentions may refer to multiple individuals if only first names are used.*")
    return "\n".join(context_lines)