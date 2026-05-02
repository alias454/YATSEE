"""
Per-transcript summary orchestration for YATSEE intelligence jobs.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from yatsee.core.errors import ConfigError
from yatsee.intel.chunking import chunk_text
from yatsee.intel.classifier import classify_transcript
from yatsee.intel.llm import llm_generate_text
from yatsee.intel.prompts import resolve_prompt_ids
from yatsee.intel.speakers import (
    build_name_permutations,
    generate_known_speakers_context,
    scan_transcript_for_names,
)
from yatsee.intel.writers import write_chunk_file, write_summary_file
from yatsee.providers.pricing import build_pricing_summary
from yatsee.providers.tokenization import estimate_token_count

logger = logging.getLogger(__name__)


def _resolve_density_threshold(pass_num: int) -> int:
    """
    Resolve the density threshold used for a summarization pass.

    Earlier passes work on larger raw transcript text, while later passes work
    on increasingly compressed material. The threshold is relaxed across passes
    so density-aware chunking does not become overly aggressive once the text
    has already been reduced.

    :param pass_num: One-based summarization pass number
    :return: Density threshold for chunking
    """
    return [25, 75, 500][min(pass_num - 1, 2)]


def _validate_resolved_prompts(
    prompts: Dict[str, str],
    prompt_ids: Dict[str, str],
) -> None:
    """
    Validate that resolved prompt identifiers exist in the loaded prompt set.

    Prompt routing should fail early and clearly if a route points at a missing
    prompt. This avoids vague downstream failures during chunk or final-pass
    generation.

    :param prompts: Loaded prompt text map
    :param prompt_ids: Resolved first/multi/final prompt IDs
    :raises ConfigError: If any resolved prompt is missing
    """
    for pass_label, prompt_id in prompt_ids.items():
        if prompt_id not in prompts:
            available = ", ".join(sorted(prompts.keys()))
            raise ConfigError(
                f"Resolved {pass_label} prompt '{prompt_id}' is not present in the loaded prompt bundle. "
                f"Available prompts: {available}"
            )


def summarize_one_transcript(
    *,
    session: Any,
    llm_provider: str,
    llm_provider_url: str,
    llm_api_key: str | None,
    llm_allow_remote: bool,
    llm_allow_insecure_http: bool,
    llm_allow_custom_executable: bool,
    model: str,
    num_ctx: int,
    transcript: str,
    base_name: str,
    context: str,
    prompt_bundle: Dict[str, Any],
    entity_cfg: Dict[str, Any],
    run_cfg: Dict[str, Any],
    output_dir: str,
) -> Dict[str, Any]:
    """
    Summarize one transcript through the configured multi-pass intelligence flow.

    The transcript may be classified first to select a prompt route. The working
    text is then chunked and summarized across one or more passes until it is
    small enough for final synthesis or the maximum pass count is reached.
    Speaker context is derived from entity configuration and injected into chunk
    prompts to improve continuity for named participants.

    Token counts are tracked for visibility and optional reference pricing.
    Final output may be written to disk, and chunk-level debug output can also
    be emitted when enabled.

    :param session: Shared requests session
    :param llm_provider: Configured provider name
    :param llm_provider_url: Provider base URL or CLI executable target
    :param llm_api_key: Optional provider API key
    :param llm_allow_remote: Whether remote non-local targets are allowed for local HTTP providers
    :param llm_allow_insecure_http: Whether plain HTTP is allowed for hosted providers
    :param llm_allow_custom_executable: Whether custom CLI executable targets are allowed
    :param model: Model name used for generation
    :param num_ctx: Requested context window
    :param transcript: Transcript text
    :param base_name: Transcript basename used for output naming
    :param context: Human-readable transcript context
    :param prompt_bundle: Loaded prompt bundle and routing metadata
    :param entity_cfg: Resolved entity configuration
    :param run_cfg: Runtime configuration for the intelligence stage
    :param output_dir: Directory for final outputs
    :return: Summary result metadata and generated output
    :raises ProviderError: If provider execution fails
    :raises ConfigError: If resolved prompt configuration is invalid
    """
    prompts = prompt_bundle["prompts"]
    prompt_lookup = prompt_bundle["prompt_router"]
    classifier_prompt_template = prompt_bundle["classifier_prompt"]
    classifier_types = prompt_bundle["classifier_types"]
    density_keywords_cfg = prompt_bundle["density_keywords"]

    token_usage = 0
    output_token_usage = 0

    meeting_type = "general"
    if not run_cfg["disable_auto_classification"]:
        classifier_prompt = classifier_prompt_template.format(
            context=context,
            text=transcript[:2000],
        )
        meeting_type = classify_transcript(
            session=session,
            llm_provider=llm_provider,
            llm_provider_url=llm_provider_url,
            llm_api_key=llm_api_key,
            model=model,
            prompt=classifier_prompt,
            num_ctx=num_ctx,
            allowed_labels=set(classifier_types.get("allowed", [])),
            llm_allow_remote=llm_allow_remote,
            llm_allow_insecure_http=llm_allow_insecure_http,
            llm_allow_custom_executable=llm_allow_custom_executable,
        )

        token_usage += estimate_token_count(classifier_prompt)
        output_token_usage += estimate_token_count(meeting_type)
        logger.info("Auto-detected meeting type: %s", meeting_type)

    prompt_ids = resolve_prompt_ids(
        meeting_type=meeting_type,
        prompt_lookup=prompt_lookup,
        first_override=run_cfg["first_prompt"],
        multi_override=run_cfg["second_prompt"],
        final_override=run_cfg["final_prompt"],
    )
    _validate_resolved_prompts(prompts, prompt_ids)

    known_speakers_permutations = build_name_permutations(entity_cfg)

    pass_num = 1
    max_pass = run_cfg["max_pass"]
    summary = ""
    working_text = transcript

    while pass_num <= max_pass:
        prompt_type = prompt_ids["first"] if pass_num == 1 else prompt_ids["multi"]
        density_threshold = _resolve_density_threshold(pass_num)
        density_keywords = set(density_keywords_cfg.get("keywords", []))

        chunks = chunk_text(
            text=working_text,
            chunk_style=run_cfg["chunk_style"],
            max_tokens=run_cfg["max_tokens"],
            density_keywords=density_keywords,
            density_threshold=density_threshold,
        )

        chunk_summaries = []
        for chunk_index, chunk in enumerate(chunks, start=1):
            logger.info(
                "Pass %d processing chunk %d/%d with prompt [%s]",
                pass_num,
                chunk_index,
                len(chunks),
                prompt_type,
            )

            speaker_matches = scan_transcript_for_names(
                chunk,
                known_speakers_permutations,
            )
            speaker_context = generate_known_speakers_context(speaker_matches)

            prompt_template = prompts[prompt_type]
            chunk_prompt = prompt_template.format(
                context=context or "No context provided.",
                text=f"{speaker_context}\n\n{chunk}",
            )

            chunk_summary = llm_generate_text(
                session=session,
                provider_name=llm_provider,
                llm_provider_url=llm_provider_url,
                api_key=llm_api_key,
                model=model,
                prompt=chunk_prompt,
                num_ctx=num_ctx,
                allow_remote=llm_allow_remote,
                allow_insecure_http=llm_allow_insecure_http,
                allow_custom_executable=llm_allow_custom_executable,
            )
            chunk_summaries.append(chunk_summary)

            if run_cfg["enable_chunk_writer"]:
                chunk_file = write_chunk_file(
                    chunk_text=chunk_summary,
                    output_dir=output_dir,
                    meeting_type=meeting_type,
                    base_name=base_name,
                    pass_num=pass_num,
                    chunk_id=chunk_index,
                )
                logger.info(
                    "Wrote %d chunk tokens to %s",
                    estimate_token_count(chunk_summary),
                    chunk_file,
                )

            token_usage += estimate_token_count(chunk_prompt)
            output_token_usage += estimate_token_count(chunk_summary)

        working_text = "\n\n".join(chunk_summaries)
        logger.debug("Joined transcript tokens %d", estimate_token_count(working_text))

        # Final synthesis starts once the intermediate text has been reduced to
        # a manageable size, collapsed to a single chunk, or the run has reached
        # its configured pass limit. A single-pass run always performs final
        # synthesis so the function returns a complete summary.
        should_finalize = False

        if pass_num >= 2:
            should_finalize = (
                estimate_token_count(working_text) <= run_cfg["max_tokens"]
                or len(chunk_summaries) == 1
                or pass_num == max_pass
            )
        elif max_pass == 1:
            should_finalize = True

        if should_finalize:
            final_prompt_id = prompt_ids["final"]
            logger.info("Processing final summary with prompt [%s]", final_prompt_id)

            final_prompt_template = prompts[final_prompt_id]
            final_prompt = final_prompt_template.format(
                context=context or "No context provided.",
                text=working_text,
            )

            summary = llm_generate_text(
                session=session,
                provider_name=llm_provider,
                llm_provider_url=llm_provider_url,
                api_key=llm_api_key,
                model=model,
                prompt=final_prompt,
                num_ctx=num_ctx,
                allow_remote=llm_allow_remote,
                allow_insecure_http=llm_allow_insecure_http,
                allow_custom_executable=llm_allow_custom_executable,
            )

            token_usage += estimate_token_count(final_prompt)
            output_token_usage += estimate_token_count(summary)
            break

        pass_num += 1

    total_tokens = token_usage + output_token_usage

    pricing = build_pricing_summary(
        show_pricing=bool(run_cfg.get("show_pricing", False)),
        actual_provider=llm_provider,
        actual_model=model,
        pricing_provider=run_cfg.get("pricing_provider"),
        pricing_model=run_cfg.get("pricing_model"),
        input_tokens=token_usage,
        output_tokens=output_token_usage,
    )

    logger.info(
        "Token usage: tokIn=%d tokOut=%d total=%d",
        token_usage,
        output_token_usage,
        total_tokens,
    )

    if pricing["enabled"]:
        if pricing["estimated_cost"] is None:
            logger.info(
                "Estimated reference cost unavailable for provider=%s model=%s",
                pricing["reference_provider"],
                pricing["reference_model"],
            )
        else:
            logger.info(
                "Estimated reference cost using %s/%s: $%.4f",
                pricing["reference_provider"],
                pricing["reference_model"],
                pricing["estimated_cost"],
            )

    output_path = ""
    if run_cfg["job_profile"] == "civic":
        output_path = write_summary_file(
            summary,
            base_name,
            output_dir,
            run_cfg["output_format"],
        )
        logger.info(
            "Final summary: %d tokens written to: %s",
            estimate_token_count(summary),
            output_path,
        )

    return {
        "base_name": base_name,
        "meeting_type": meeting_type,
        "summary": summary,
        "output_path": output_path,
        "input_tokens": token_usage,
        "output_tokens": output_token_usage,
        "total_tokens": total_tokens,
        "pricing": pricing,
    }