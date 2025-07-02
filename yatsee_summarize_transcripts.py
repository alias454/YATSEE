#!/usr/bin/env python3
"""
YATSEE Transcript Summarizer
----------------------------

This tool summarizes transcripts (e.g., city council, committee meetings) using a local LLM via Ollama.

usage examples:
  python yatsee_summarize_transcripts.py --model llama3 -i council_meeting_2025_06_01 --context "City Council Meeting - June 2025"
  python yatsee_summarize_transcripts.py --model mistral -i firehall_meeting_2025_05 --context "Fire Hall Proposal Discussion"
  python yatsee_summarize_transcripts.py --model mistral -i firehall_meeting_2025_05 --output-format markdown

requirements:
  - Ollama running locally: https://ollama.com
  - A supported model pulled locally (e.g. `ollama pull mistral`, `ollama pull llama3`)

example models:
  - `mistral`: small, fast, reliable (recommended for quick or resource-limited use)
  - `llama3`: slower, but better at reasoning and context tracking (ideal for complex transcripts)

features:
  - Auto-classifies meeting type using a short snippet of transcript + fallback regex based on filename
  - Dynamically selects appropriate summarization prompts based on classification (e.g. city council, finance committee, etc.)
  - Optionally override classification with manual prompt selection (--first-prompt, --second-prompt, --final-prompt)
  - Summarizes long transcripts in chunks with automatic merging and optional multi-pass refinement
  - Supports multi-pass summarization up to a defined depth (--max-passes)
  - Works with streamed, local models via Ollama’s `/api/generate` endpoint
  - Modular prompt system for various summary styles (overview, action items, detailed, exhaustive, final pass, etc.)

arguments:
  -i / --input-dir         Directory or file path of transcripts to summarize
  --model                  Model to use (e.g., llama3, mistral)
  --context                Optional human-readable meeting name, fallback is derived from filename
  --output-dir             Directory to save summaries (default: ./summary/)
  --output-format          'markdown' (default) or 'yaml'
  --prompt                 Optional manual prompt ID override for first pass
  --second-pass-prompt     Optional manual prompt ID for second summarization pass
  --max-words              Approximate word count per chunk (default: 3500)
  --max-passes             Maximum summarization passes to perform (default: 3)
  --disable-auto-classification  Disable automatic prompt selection based on meeting type

outputs:
  - Saves each chunk summary as: `<filename>_chunkN.yaml|md`
  - Saves final merged summary as: `<filename>_final_summary.yaml|md`
  - All output written to `--output-dir` (default: ./summary)

prompt system:
  - Built-in prompt variants include: overview, action_items, detailed, more_detailed, most_detailed, final_pass_detailed
  - Classification-aware mapping chooses best prompt variants based on detected meeting type
  - Supports context-driven adaptation of summaries (via --context or filename fallback)

notes:
  - Designed for transcripts with structured civic dialogue: councils, committees, town halls
  - Gender-neutral output style with summarization emphasis on motions, votes, decisions, and speaker intent
  - Designed to be local-first and privacy-preserving — no cloud APIs required

TODO: Entity Normalization & Community Feedback Loop
"""

import argparse
import os
import re
import sys
import toml
import json
import requests
import textwrap
from typing import List, Dict, Set

OLLAMA_SERVER_URL = "http://localhost:11434"

# PROMPT_LOOKUP structure that maps meeting types to the recommended first and second pass prompts
# Get IDs from the PROMPTS dictionary.
PROMPT_LOOKUP = {
    "city_council": {
        "first": "most_detailed",
        "multi": "structure_preserving",
        "final": "civic_scan"
},
    "finance_committee": {
        "first": "finance_committee_detailed",
        "multi": "structure_preserving",
        "final": "finance_committee_final"
    },
    "committee_of_the_whole": {
        "first": "committee_meeting_detailed",
        "multi": "structure_preserving",
        "final": "committee_meeting_final"
    },
    "zoning_committee": {
        "first": "more_detailed",
        "multi": "structure_preserving",
        "final": "committee_meeting_final"
    },
    "general": {
        "first": "more_detailed",
        "multi": "structure_preserving",
        "final": "final_pass_detailed"
    }
}

# Define PROMPTS dictionary to hold different prompt templates
PROMPTS = {
    "overview": (
        "You are an assistant that produces structured meeting minutes from transcripts.\n\n"
        "Context: {context}\n\n"
        "Transcript:\n{text}\n\n"
        "Produce a concise, high-level meeting summary from the transcript.\n"
        "Use they/them pronouns unless gender is explicit.\n"
        "Group related content; exclude non-substantive or empty sections.\n"
        "Avoid repeating headers.\n"
        "Use plain numbers, not Roman numerals."
    ),
    "action_items": (
        "You are an assistant that produces structured meeting minutes from transcripts.\n\n"
        "Context: {context}\n\n"
        "Transcript:\n{text}\n\n"
        "Extract all action items, motions, decisions, and follow-ups from the transcript.\n"
        "Use they/them pronouns unless gender is explicit.\n"
        "Focus on actionable content only.\n"
        "Skip commentary, informal remarks, and empty sections.\n"
        "Avoid repeating headers.\n"
        "Use plain numbers, not Roman numerals."
    ),
    "roll_call_only": (
        "You are an assistant that produces structured meeting minutes from transcripts.\n\n"
        "Context: {context}\n\n"
        "Transcript:\n{text}\n\n"
        "Extract attendance from transcript.\n"
        "List 'Present' and 'Absent' names.\n"
        "Infer attendance if no roll call.\n"
        "Exclude titles unless needed.\n"
        "Skip irrelevant commentary and empty sections.\n"
        "Avoid repeating headers.\n"
        "Use plain numbers, not Roman numerals."
    ),
    "vote_log_only": (
        "You are an assistant that produces structured meeting minutes from transcripts.\n\n"
        "Context: {context}\n\n"
        "Transcript:\n{text}\n\n"
        "List motions, proposers, seconders, and vote results.\n"
        "Format: 'Motion to [action], made by [name], seconded by [name] – Passed [X–Y]'.\n"
        "Include individual votes if roll call.\n"
        "Skip procedural or unrelated comments.\n"
        "Omit empty sections.\n"
        "Avoid repeating headers.\n"
        "Use plain numbers, not Roman numerals."
    ),
    "finance_meeting_summary": (
        "You produce structured minutes of finance or budget-related meetings.\n\n"
        "Context: {context}\n\n"
        "Transcript:\n{text}\n\n"
        "Use they/them pronouns unless gender is explicit.\n\n"
        "Summarize financial topics in clear sections:\n"
        "- Budget proposals, allocations, transfers, amendments\n"
        "- Capital improvement fund discussions\n"
        "- Grant approvals, expenses, revenue reports\n"
        "- Voting records and finance-related motions\n"
        "- Include a summary header with attendees, location, date\n"
        "- Exclude non-financial discussions unless relevant\n"
        "- Omit empty or non-substantive sections (e.g., 'none,' 'N/A')\n"
        "- Avoid repeating headers (attendees, location, date)\n"
        "- Use plain numbers, not Roman numerals."
    ),
    "finance_committee_detailed": (
        "You produce detailed, structured minutes of finance or budget committee meetings from transcripts.\n\n"
        "Context: {context}\n\n"
        "Transcript:\n{text}\n\n"
        "Use they/them pronouns unless gender is explicit.\n\n"
        "Focus on fiscal matters and oversight. Include:\n"
        "- Committee name, attendees, location, date\n"
        "- Budget discussions, allocations, audits, expenditures\n"
        "- Proposed budget amendments, rationale, vote outcomes\n"
        "- Motions, approvals (who made/seconded, vote tallies)\n"
        "- Grants, contracts, loans, financial forecasts\n"
        "- Member concerns (risk, liabilities, transparency)\n"
        "- Recommendations to city council or others\n"
        "- Clearly label sections: Budget Amendments, Audit Reports, Approvals, Follow-ups\n"
        "- Maintain chronological flow\n"
        "- Group related items (e.g., multiple grants)\n"
        "- Include extended debate or dissent if relevant\n"
        "- Exclude 'N/A' or irrelevant procedural chatter\n"
        "- Omit empty or non-substantive sections (e.g., 'none', 'N/A')\n"
        "- Avoid repeating headers (attendees, location, date)\n"
        "- Use plain numbers, not Roman numerals."
    ),
    "finance_committee_final": (
        "You are a civic assistant generating detailed markdown summaries from finance or budget-related meetings.\n\n"
        "Context: {context}\n\n"
        "Summaries:\n{text}\n\n"
        "Instructions:\n"
        "- Merge the content into a single, well-structured summary organized by topic or department.\n"
        "- Highlight all budget amendments, proposed cuts/increases, reallocations, and any funding shifts.\n"
        "- Clearly list each grant, program, or revenue source mentioned. Include names, amounts, match requirements, and intended use.\n"
        "- If a fund or budget line is being adjusted, note the original and revised amounts if mentioned.\n"
        "- Attribute concerns or proposals to specific officials when clearly named; otherwise use general labels like 'a committee member said...'.\n"
        "- Include all formal motions and votes, but recognize that many discussions may not result in a vote — document those discussions anyway.\n"
        "- Avoid repeating boilerplate like roll call or approval of prior minutes unless notable.\n"
        "- Use markdown headers to separate topics (e.g., 'Grants and Funding', 'Budget Amendments', 'Departmental Requests', etc).\n"
        "- Maintain a neutral and professional tone throughout.\n"
        "- Do not collapse multiple topics into a single vague bullet — retain detail even for small funding items."
    ),
    "committee_meeting_summary": (
        "You are an assistant producing structured summaries of committee meetings.\n\n"
        "Context: {context}\n\n"
        "Transcript:\n{text}\n\n"
        "Use gender-neutral pronouns (they/them) unless a speaker’s gender is explicit.\n\n"
        "Summarize the meeting by topic or subcommittee focus:\n"
        "- Identify presenters, topics, motions, and recommendations.\n"
        "- Include outcomes, follow-ups, and any items forwarded to city council.\n"
        "- Briefly note informal or public comments only if relevant.\n"
        "- Begin with a summary of attendees, location, and date.\n"
        "- Omit sections with no substantive content (e.g., 'none', 'N/A').\n"
        "- Avoid repeating headers (attendees, location, date) within the meeting.\n"
        "- Use plain numbers (e.g., '2') instead of Roman numerals."
    ),
    "committee_meeting_detailed": (
        "You are an assistant producing structured minutes of committee meetings from transcripts.\n\n"
        "Context: {context}\n\n"
        "Transcript:\n{text}\n\n"
        "Use gender-neutral pronouns (they/them) unless a speaker’s gender is explicit.\n\n"
        "Provide a detailed summary organized by agenda item or topic.\n"
        "- Include committee name, attendees, location, and date.\n"
        "- Cover key discussion points and speakers.\n"
        "- Document motions made, seconded, and outcomes.\n"
        "- Include votes with counts or roll call results when available.\n"
        "- Highlight notable quotes influencing decisions.\n"
        "- Note recommendations to higher bodies (e.g., city council).\n"
        "- List follow-ups or pending reviews.\n"
        "- Briefly summarize informal remarks or public comments only if relevant.\n"
        "- Use clear, plain headers by topic or agenda.\n"
        "- Preserve event sequence and meeting flow.\n"
        "- Exclude 'N/A', empty, or irrelevant sections.\n"
        "- Avoid repeating headers (attendees, location, date) within the same meeting.\n"
        "- Use plain numbers, not Roman numerals."
    ),
    "committee_meeting_final": (
        "You are an assistant that synthesizes committee meeting summaries into a unified, readable markdown report.\n\n"
        "Context: {context}\n\n"
        "Summaries:\n{text}\n\n"
        "Instructions:\n"
        "- Organize by **agenda item or major discussion topic**, not speaker order.\n"
        "- Include **motions and votes**, but only if clearly described — avoid fabricating counts.\n"
        "- Reflect **disagreements, concerns, or points of confusion** clearly and neutrally.\n"
        "- If attendees raise issues without resolution, group them under 'Discussion Points' or 'Open Questions'.\n"
        "- Do **not** repeat metadata (date, location) more than once.\n"
        "- Preserve key comments **only** if they clarify issues, provide insight, or explain a stance.\n"
        "- Remove redundancy across chunks; combine repeated information.\n"
        "- Use markdown formatting: clear headers, bullet points, and clean lists.\n"
        "- Avoid robotic phrasing or repeating 'Alderman X said...' unless it adds value.\n"
        "- If a motion is described in multiple ways, preserve the **clearest or most complete version**."
    ),
    "detailed": (
        "You are an assistant that produces structured meeting minutes from transcripts.\n\n"
        "Context: {context}\n\n"
        "Transcript:\n{text}\n\n"
        "Use gender-neutral pronouns (they/them) unless a speaker’s gender is explicit.\n\n"
        "Provide a detailed, structured summary including agenda items, major topics, speakers, motions, and vote outcomes.\n"
        "- Briefly summarize prayers or informal remarks.\n"
        "- Group related discussions logically.\n"
        "- Exclude empty or non-substantive sections (e.g., 'none', 'N/A').\n"
        "- Avoid repeating headers (attendees, location, date) within the same meeting.\n"
        "- Use plain numbers instead of Roman numerals."
    ),
    "more_detailed": (
        "You are an assistant that produces structured meeting minutes from transcripts.\n\n"
        "Context: {context}\n\n"
        "Transcript:\n{text}\n\n"
        "Use gender-neutral pronouns (they/them) unless a speaker’s gender is explicit.\n\n"
        "Provide a detailed summary organized by agenda item or topic.\n"
        "- For each topic, include key speakers, main points, quotes, motions (with who made/seconded), vote counts, and follow-ups.\n"
        "- Begin with a meeting header: attendees, location, and date.\n"
        "- Preserve topic order and discussion flow.\n"
        "- Briefly summarize prayers or invocations.\n"
        "- Combine related discussions for clarity.\n"
        "- Exclude empty or non-substantive sections (e.g., 'none', 'N/A').\n"
        "- Avoid repeating headers within the same meeting.\n"
        "- Use plain numbers instead of Roman numerals."
    ),
    "most_detailed": (
        "You are a meeting summarizer producing clear, structured markdown summaries from transcripts.\n\n"
        "Context: {context}\n\n"
        "Transcript:\n{text}\n\n"
        "Instructions:\n"
        "- Summarize all important agenda items, discussions, and decisions.\n"
        "- Include motions, who made and seconded them, and vote outcomes. If a roll call vote is taken, list how each person voted.\n"
        "- Identify speakers by name or title *only if clearly stated*. Do not guess or combine similar names (e.g., 'Wayne' and 'Director Duckman') unless explicitly linked.\n"
        "- If a speaker is unnamed, refer to them generically (e.g., 'a council member said...').\n"
        "- Use quotes sparingly and only when they add clarity or emphasis. Otherwise, paraphrase naturally.\n"
        "- Group content under clear, natural section headers. Avoid copying transcript labels like 'Item Number Four.'\n"
        "- Maintain a neutral, civic tone. Avoid robotic phrasing or repetition.\n"
        "- Use plain numbers (e.g., '2' instead of 'II').\n"
        "- Avoid assuming gender. Use gendered pronouns only when unambiguous from the transcript."
    ),
    "structure_preserving": (
        "You are a meeting summarizer producing clear, structured markdown summaries from transcripts.\n\n"
        "Context: {context}\n\n"
        "Transcript:\n{text}\n\n"
        "Instructions:\n"
        "- Summarize all substantive agenda items, discussion points, and decisions without over-compressing.\n"
        "- Retain detail on motions, who made and seconded them, and vote outcomes. If a roll call vote is included, list individual votes.\n"
        "- Preserve attribution when names or titles are explicitly stated. Refer to unnamed speakers generically (e.g., 'a resident said...').\n"
        "- Group content under natural, informative section headers (e.g., 'Public Comments,' 'Budget Discussion'). Avoid generic item labels.\n"
        "- Capture meaningful quotes or paraphrased statements that reflect public sentiment or council debate.\n"
        "- Include resolution and ordinance numbers when mentioned and summarize what each one does.\n"
        "- Avoid unnecessary compression; this is an intermediate summary meant to preserve useful structure and fidelity for a later final pass.\n"
        "- Maintain a civic, neutral tone. Do not editorialize. Avoid assumptions about speaker identity or intent.\n"
        "- Use plain numbers (e.g., '2' instead of 'II').\n"
        "- Format output in clean, readable Markdown. Use bullet points or subheaders where helpful."
    ),
    "civic_scan": (
        "You are an assistant that produces concise, high-level civic meeting summaries from transcripts.\n\n"
        "**Context:** {context}\n\n"
        "**Transcript:**\n{text}\n\n"
        "**Instructions:**\n"
        "- Output must be formatted in **Markdown**.\n"
        "- Use **plain Arabic numerals (1, 2, 3...)** for section headers. **Do not use Roman numerals.**\n"
        "- Group content under clear, topical headers (e.g., Ordinance Changes, Public Funding, Planning and Zoning).\n"
        "- Focus on real civic impact: surface ordinance changes, grant activity, public funding, zoning decisions, infrastructure/budget items, emotionally charged issues, controversies, and unresolved matters.\n"
        "- Include **brief quotes** when they express public sentiment, concern, or disagreement.\n"
        "- Flag votes, funding amounts, appointments, and any item likely to impact residents or warrant follow-up.\n"
        "- Exclude filler or procedural commentary unless it affects transparency, accountability, or trust.\n"
        "- Use **they/them pronouns** unless gender is clearly stated.\n"
        "- Keep the summary skimmable in **2–3 minutes** for an engaged citizen audience.\n"
        "- Be consistent and avoid repeating section headers.\n"
    ),
    "final_pass_detailed": (
        "You are an assistant that synthesizes structured summaries of public meetings into a unified, readable markdown file.\n\n"
        "Context: {context}\n\n"
        "Summaries:\n{text}\n\n"
        "Instructions:\n"
        "- Combine all chunked summaries into a cohesive markdown file organized by agenda item or topic.\n"
        "- Retain all motions, including who made and seconded them, the vote result, and vote breakdowns if present.\n"
        "- Preserve all ordinance/resolution numbers, titles, subjects, amounts, contractors, and related agency names.\n"
        "- Avoid duplicating items in multiple sections — instead consolidate detailed discussion with motion outcome.\n"
        "- Include a dedicated 'Motions and Votes Summary' section at the end with a clear list of all actions taken.\n"
        "- Do not omit public comments; summarize each speaker’s topic, concern, or praise with clarity.\n"
        "- Accurately attribute remarks only when names or roles are clearly stated. Use 'a speaker' or 'a council member' otherwise.\n"
        "- If multiple individuals have the same last name, use title and full name to disambiguate (e.g., 'Mayor Jodi Miller' vs. 'Scott Miller').\n"
        "- Use markdown headings to match the meeting flow, and include bullet points for clarity where appropriate.\n"
        "- Keep a professional, neutral tone throughout — do not invent dialogue or fill in missing attribution.\n"
        "- Use plain numbers (e.g., '2'), avoid duplicate metadata (e.g., meeting date), and do not repeat sections.\n"
        "- **Do NOT include any internal notes, meta-commentary, explanations of how this summary was created, or references to the summarization process.**"
    )
}

CLASSIFIER_PROMPT = {
    "classifier": (
    "You are an assistant that analyzes meeting transcripts to determine the meeting type.\n\n"
    "Context: {context}\n\n"
    "Transcript:\n{text}\n\n"
    "Based on the content and structure of the meeting, classify it into **one** of the following categories:\n"
    "- city_council\n"
    "- finance_committee\n"
    "- committee_of_the_whole\n"
    "- zoning_committee\n"
    "- general\n\n"
    "Return only the classification value as plain text. Do not include explanations or additional commentary."
    )
}


def classify_meeting(session: requests.Session, model: str, prompt: str) -> str:
    """
    Classifies the type of meeting (e.g., city council, finance committee) using a local LLM via a streaming API.

    :param session: A preconfigured requests.Session object for efficient HTTP reuse.
    :param model: Name of the model to use (e.g., 'llama3').
    :param prompt: A formatted prompt string with placeholders for context and text.
    :return: A lowercase string representing the inferred meeting type, or 'general' as fallback.
    :raises RuntimeError: If HTTP request or JSON parsing fails.
    """
    payload = {"model": model, "prompt": prompt, "stream": True}
    try:
        response = session.post(f"{OLLAMA_SERVER_URL}/api/generate", json=payload, stream=True)
        response.raise_for_status()

        raw_reply = ""
        for line in response.iter_lines():
            if not line:
                continue
            try:
                data = json.loads(line.decode("utf-8"))
                raw_reply += data.get("response", "")
                if data.get("done", False):
                    break
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Failed to parse JSON from Ollama stream: {e}")

        label = raw_reply.strip().lower()

        # Accepted meeting types The model returns on of these based on the classifier prompt
        allowed_labels = {
            "city_council",
            "finance_committee",
            "committee_of_the_whole",
            "zoning_committee",
            "general"
        }

        label = label if label in allowed_labels else "general"
        return label

    except requests.RequestException as e:
        raise RuntimeError(f"Error during meeting classification request: {e}")


def extract_context_from_filename(filename, extra_note=None):
    """Extracts a human-friendly meeting context from a transcript filename."""
    # Just the basename (strip path)
    base = os.path.basename(filename)

    # Remove hash-like prefix if present
    base = re.sub(r"^[\w\-]+?\.", "", base)

    # Remove extension
    base = base.replace(".txt", "")

    # Normalize separators
    base = base.replace("_", " ").replace("-", " ")

    # Try to extract components
    meeting_match = re.match(r"(.*?) (\d{1,2}) (\d{1,2}) (\d{2,4})", base)
    if meeting_match:
        kind = meeting_match.group(1).strip().title()
        month = int(meeting_match.group(2))
        day = int(meeting_match.group(3))
        year = int(meeting_match.group(4))
        if year < 100:
            year += 2000

        date_str = f"{year:04d}-{month:02d}-{day:02d}"
        context = f"{kind} — {date_str}"
    else:
        context = base.title()

    if extra_note:
        context += f" ({extra_note.strip()})"

    return context


def estimate_token_count(text: str) -> int:
    """
    Estimate token count from a text string using an average ratio of 0.75 words per token.

    :param text: Input text
    :return: Estimated token count
    """
    return int(len(text.split()) / 0.75)


def calculate_cost_openai_gpt4o(input_tokens: int, output_tokens: int) -> float:
    """
    Estimate the cost of using OpenAI's GPT-4o model based on input and output token usage.

    GPT-4o Pricing:
      - $0.005 per 1K input tokens
      - $0.015 per 1K output tokens

    :param input_tokens: Estimated input token count
    :param output_tokens: Estimated output token count
    :return: Estimated cost in USD
    """
    input_cost = input_tokens / 1000 * 0.005
    output_cost = output_tokens / 1000 * 0.015
    return round(input_cost + output_cost, 4)


def filter_transcript_file(path_list: list[str]) -> list[str]:
    """
    Resolve which file to use for summarization: .out or .txt.
    .out files will override .txt files in precedence
    .out file extensions are preferred but fallback to .txt

    :param path_list: Base name of the meeting file (no extension)
    :return: List of resolved file paths
    """
    files = []

    for path in path_list:
        if path.lower().endswith(".out"):
            files.append(path)
        elif path.lower().endswith(".txt"):
            files.append(path)

    return files


def get_files_list(path: str) -> list[str]:
    """
    Collect a list of .txt and .out files from a directory or single file path.

    :param path: Path to .txt or .out files, or directory
    :return: List of valid txt file paths
    :raises FileNotFoundError: If no valid files found
    :raises ValueError: If unsupported file extension encountered
    """
    valid_extensions = (".txt", ".out")
    txt_files = []

    if os.path.isdir(path):
        for filename in os.listdir(path):
            full_path = os.path.join(path, filename)
            if os.path.isfile(full_path) and filename.lower().endswith(valid_extensions):
                txt_files.append(full_path)
        if not txt_files:
            raise FileNotFoundError(f"No valid .txt files found in directory: {path}")
    elif os.path.isfile(path):
        if path.lower().endswith(valid_extensions):
            txt_files.append(path)
        else:
            raise ValueError(f"Unsupported file extension: {os.path.splitext(path)[1]}")
    else:
        raise FileNotFoundError(f"Input path not found: {path}")

    return txt_files


def prepare_text_chunk(text: str, max_tokens: int = 2500, overlap_tokens: int = 400) -> List[str]:
    """
    Split text into overlapping chunks by token count using a fixed sliding window.

    :param text: Raw text input
    :param max_tokens: Number of tokens per chunk
    :param overlap_tokens: Number of tokens to overlap between chunks
    :return: List of text chunks (strings)
    """

    tokens = text.split()
    total_tokens = len(tokens)

    if total_tokens <= max_tokens:
        return [text]

    chunks = []
    start = 0

    while start < total_tokens:
        end = min(start + max_tokens, total_tokens)
        chunk_tokens = tokens[start:end]
        chunk_text = " ".join(chunk_tokens)
        chunks.append(chunk_text)

        if end == total_tokens:
            break

        # Slide window by max_tokens - overlap_tokens to keep overlap
        start += max_tokens - overlap_tokens
        if start < 0:
            start = 0

    return chunks


def prepare_text_chunk_sentence(text: str, max_tokens: int = 3000) -> list[str]:
    """
    Chunk text roughly by sentences to stay within token limit.
    If the full text fits, returns it as a single chunk.

    :param text: The full transcript text
    :param max_tokens: Approximate max tokens per chunk (default 3000)
    :return: List of text chunks
    """
    # Fast path: skip chunking if the text fits in one chunk
    if int(len(text.split()) / 0.75) <= max_tokens:
        return [text]

    # Split text into sentences
    sentences = re.split(r'(?<=[.?!])\s+', text)

    chunks = []
    current_chunk = []
    current_word_count = 0

    for sentence in sentences:
        sentence_word_count = len(sentence.split())

        if current_word_count + sentence_word_count > max_tokens and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_word_count = 0

        current_chunk.append(sentence)
        current_word_count += sentence_word_count

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def summarize_transcript(session: requests.Session, model: str, prompt: str = "detailed") -> str:
    """
    Use a local Ollama LLM to summarize transcript text.
    Ollama returns streaming JSON objects, one per line

    :param session: Pass in connection session
    :param model: Name of the model to use (e.g., 'llama3', 'mistral')
    :param prompt: Type of prompt to use (key from PROMPTS)
    :return: The generated summary as a string
    :raises RuntimeError: On HTTP or JSON streaming errors
    """
    payload = {"model": model, "prompt": prompt}

    try:
        response = session.post(f"{OLLAMA_SERVER_URL}/api/generate", json=payload, stream=True)
        response.raise_for_status()

        summary = ""
        for line in response.iter_lines():
            if not line:
                continue
            try:
                data = json.loads(line.decode("utf-8"))
                summary += data.get("response", "")
                if data.get("done", False):
                    break
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Failed to parse JSON from Ollama stream: {e}")
        return summary

    except requests.RequestException as e:
        raise RuntimeError(f"Error during Ollama request: {e}")


def write_summary_file(summary: str, basename: str, output_dir: str, fmt: str = "yaml") -> bool:
    """
    Write the summary to the given output directory in either YAML or Markdown format.

    :param summary: The summary string
    :param basename: Original transcript base name (used for output filename)
    :param output_dir: Directory to write summary file into
    :param fmt: Output format: "markdown" (default) or "yaml"
    :return: True if file written successfully, False otherwise
    """
    try:
        filename = f"{basename}.summary.{ 'md' if fmt == 'markdown' else 'yaml' }"
        out_path = os.path.join(output_dir, filename)

        if fmt == "markdown":
            content = f"# Summary: {basename}\n\n{summary.strip()}\n"
        else:  # YAML default
            content = f"summary: |\n  " + summary.strip().replace("\n", "\n  ") + "\n"

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(content)

        return True

    except (OSError, IOError) as e:
        print(f"❌ Failed to write summary file: {e}")
    except Exception as e:
        print(f"❌ Unexpected error during write: {e}")

    return False


def write_chunk_files(chunks: list[str], output_dir: str, meeting_type: str, base_name: str) -> bool:
    """
    Save transcript chunks to disk under organized directory structure.

    :param chunks: List of text chunks to write
    :param output_dir: Base directory for output files
    :param meeting_type: Meeting category label (e.g., 'city_council')
    :param base_name: Base filename (without extension) for chunk files
    :return: True if all chunks saved successfully, False otherwise
    """
    try:
        chunks_path = os.path.join(output_dir, "chunks", meeting_type, base_name)
        os.makedirs(chunks_path, exist_ok=True)

        for idx, chunk_text in enumerate(chunks):
            chunk_filename = f"{base_name}_part{idx:02d}.txt"
            chunk_path = os.path.join(chunks_path, chunk_filename)
            with open(chunk_path, "w", encoding="utf-8") as f:
                f.write(chunk_text)
        return True
    except Exception as e:
        print(f"Error saving chunks: {e}")
        return False


def generate_known_speakers_context(speaker_matches: dict) -> List[str] | str:
    """
    Generate a short markdown-friendly context note from speaker match data.

    :param speaker_matches: Dict mapping canonical person ID to set of matching name variants
    :return: A string with a sentence noting detected speaker mentions in this chunk
    """
    if not speaker_matches:
        return ""

    descriptive_mentions = []
    ambiguous_hits = []

    for canonical_name, matches in speaker_matches.items():
        sorted_matches = sorted(matches, key=len, reverse=True)
        best_variant = sorted_matches[0]
        descriptive_mentions.append(best_variant)

        # Check if multiple name forms were used
        if len(matches) > 1:
            ambiguous_hits.append((canonical_name, sorted_matches))

    mentions_string = ", ".join(descriptive_mentions)
    context_lines = [f"*Speakers detected in this segment include: {mentions_string}."]

    if ambiguous_hits:
        context_lines.append("Some individuals were referenced using multiple name variants, e.g.:")
        for name, variants in ambiguous_hits:
            readable = name.replace("_", " ")
            context_lines.append(f"  - {readable}: {', '.join(variants)}")

    context_lines.append("Mentions of partial or ambiguous names may refer to multiple individuals.*")
    return "\n".join(context_lines)


def scan_transcript_for_names(transcript_text: str, name_permutations: Dict[str, List[str]]) -> Dict[str, Set[str]]:
    """
    Scan a transcript and return all matched name/title variants from the known list of permutations.

    :param transcript_text: Raw string of the transcript text
    :param name_permutations: Dictionary mapping person keys (e.g., 'Rachel_Simmons') to list of known name/title permutations
    :return: Dictionary mapping matched person keys to the set of matched strings found in the transcript
    """
    found_matches = {}

    # Normalize the transcript once (for case-insensitive matching)
    lower_transcript = transcript_text.lower()

    for person_key, permutations in name_permutations.items():
        matches = set()

        for name_variant in permutations:
            # Simple word boundary match to avoid partial hits (e.g., "Rob" inside "Robotics")
            pattern = r'\b' + re.escape(name_variant.lower()) + r'\b'
            if re.search(pattern, lower_transcript):
                matches.add(name_variant)

        if matches:
            found_matches[person_key] = matches

    return found_matches


def build_name_permutations(data_config: dict) -> dict:
    """
    Generate all possible name and title permutations for known individuals in a city council config.

    :param data_config: Subsection of a parsed TOML config under a specific entity (e.g., city_council)
    :return: Dictionary mapping person identifiers (e.g., 'Rachel_Sampson') to a sorted list of name/title variants
    :raises KeyError: If required keys like 'people' or 'titles' are missing in the config
    """
    people_by_role = data_config.get("people", {})
    titles_by_role = data_config.get("titles", {})

    name_permutations = {}

    for role_key, people in people_by_role.items():
        titles_for_role = titles_by_role.get(role_key, [])

        for person_key, known_aliases in people.items():
            # Extract full name from the key (e.g., 'Rachel_Sampson')
            try:
                first_name, last_name = person_key.split("_", 1)
            except ValueError:
                # Handle unexpected name formats gracefully
                first_name = person_key
                last_name = ""
            full_name = f"{first_name} {last_name}".strip()

            # Start with known aliases + inferred names
            permutations = set(known_aliases)
            permutations.update({first_name, last_name, full_name})

            # Add title permutations
            for title in titles_for_role:
                # Add standard "Title Lastname" and "Title Fullname" forms
                permutations.add(f"{title} {last_name}".strip())
                permutations.add(f"{title} {full_name}".strip())

                # If the title contains multiple words (e.g., "Information Technology"), include rephrased forms
                if " " in title:
                    permutations.add(f"{title} Director {last_name}".strip())
                    permutations.add(f"{title} Director {full_name}".strip())

            name_permutations[person_key] = sorted(permutations)

    return name_permutations


def get_nested_config(config: dict, path: str) -> dict:
    """
    Given a dot-separated path (e.g. 'country.US.state.IL.city_council'),
    traverse the config dictionary to return the nested sub-config.
    """
    for part in path.split("."):
        config = config[part]
    return config


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Summarize city council, committee, or other civic meeting transcripts using a local Ollama language model.

        This tool reads transcript files (either a single file or an entire directory), auto-classifies the meeting 
        type (e.g., city_council, finance_committee), and summarizes using a multi-pass pipeline with structured prompts.

        Features:
          - Automatic classification of meeting type using transcript context + fallback filename heuristics
          - Dynamic prompt selection per meeting type (e.g., final_pass_detailed for council, financial_summary for finance)
          - Manual override of prompt type(s) for custom workflows
          - Recursive summarization of long transcripts via intelligent chunking and reprocessing
          - Multi-pass refinement up to a configurable depth (default: 3)
          - Optional disabling of auto classification with `--disable-auto-classification`
          - Output in either YAML or Markdown format
          - Token usage tracking with cost estimation (for reference if used with paid APIs in future)

        Example usage:
          python yatsee_summarize_transcripts.py --model llama3 -i normalized/ --context "City Council - June 2025"
          python yatsee_summarize_transcripts.py -m mistral -i transcripts/ -o summaries/ --output-format markdown
          python yatsee_summarize_transcripts.py -m gemma:2b -i finance/ --disable-auto-classification --prompt detailed

        Input:
          - Single file or directory of `.txt` or `.out` files
          - Context string (optional) used to enhance prompt relevance
          - Models must be locally pulled via `ollama pull` (e.g., `llama3`, `mistral`)

        Output:
          - Individual chunk summaries saved with `_chunkN` suffix
          - Final combined summary saved as `_final_summary.md` or `.yaml`
          - Written to output directory (`--output-dir`, default: ./summary)

        This tool is privacy-respecting, fully local, and optimized for transparency, civic clarity, and reproducibility.
        """)
    )

    parser.add_argument("-m", "--model", help="Model name (e.g. 'llama3', 'mistral', gemma:2b)")
    parser.add_argument("-i", "--txt-input", help="Path to a transcript file or directory (supports .txt or .out)")
    parser.add_argument("-c", "--context", default="", help="Optional meeting context to guide summarization")
    parser.add_argument("-o", "--output-dir", default="summary", help="Directory to save the summary output(Default: summary)")
    parser.add_argument("-f", "--output-format", choices=["markdown", "yaml"], default="markdown", help="Summary output format")
    parser.add_argument("-w", "--max-words", type=int, default=3500, help="Word count above which transcript is chunked")
    parser.add_argument("-t", "--max-tokens", type=int, default=2500, help="Approximate max tokens per chunk")
    parser.add_argument("-p", "--max-pass", type=int, default=3, help="Max number of iterations for multi-pass refinement (Default: 3)")
    parser.add_argument("-d", "--disable-auto-classification", action="store_true", help="Disable auto classification. Make sure to set first and second pass prompts")
    parser.add_argument("--first-prompt", choices=PROMPTS.keys(), help="Prompt type to use for summarization (Only used when auto classification is disabled)")
    parser.add_argument("--second-prompt", choices=PROMPTS.keys(), help="Prompt type for second pass summarization of chunk summaries (Only used when auto classification is disabled)")
    parser.add_argument("--final-prompt", choices=PROMPTS.keys(), help="Prompt type for final pass summarization of chunk summaries (Only used when auto classification is disabled)")
    parser.add_argument("--print-prompts", action="store_true", help="Print all prompt templates and exit")
    args = parser.parse_args()

    # Get the base and entity keys
    config_file = "yatsee.toml"
    config = toml.load(config_file)
    base = config.get("base", "")
    entity = config.get("entity", "")

    full_key = base + entity

    # Load and extract the sub-config dynamically
    data_config = get_nested_config(config, full_key)

    # Generate permutations
    known_speakers_permutations = build_name_permutations(data_config)

    # Handle --print-prompts early and exit
    if args.print_prompts:
        print("Available prompt templates:\n")
        for key, prompt_text in PROMPTS.items():
            print(f"=== {key} ===\n{prompt_text}\n{'-'*40}\n")
        return 0

    # Check for passed in arg.model
    supported_models = ["mistral", "llama3", "gemma:2b"]
    if not args.model:
        print("❌ No model specified. Use --model to set one (e.g. mistral, llama3).", file=sys.stderr)
        return 1

    if args.model.lower() not in supported_models:
        print(f"❌ Unsupported model '{args.model}'. Supported models are: {', '.join(supported_models)}.", file=sys.stderr)
        return 1

    # Collect txt files from input path
    if not args.txt_input:
        print("❌ No input file or directory specified. Use --txt-input to set one.", file=sys.stderr)
        return 1

    try:
        file_list = get_files_list(args.txt_input)
        file_list = filter_transcript_file(file_list)
    except (FileNotFoundError, ValueError) as e:
        print(f"❌ {e}", file=sys.stderr)
        return 1

    # Determine output directory, default to the ./summary directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Open up a request object and setup the client connection
    # Create a global or shared session object once
    llm_session = requests.Session()

    # Optionally configure session headers, retries, timeouts here
    llm_session.headers.update({"Content-Type": "application/json"})

    for file_path in file_list:
        # Extract a friendly context string (e.g., meeting name and date) from the filename
        context = extract_context_from_filename(file_path, args.context)

        # Initialize counters for input and output tokens (used to estimate API cost)
        token_usage = 0
        output_token_usage = 0

        print(f"🔍 Using model: {args.model}")
        # Base name of the input file
        base_name = os.path.splitext(os.path.basename(file_path))[0]

        try:
            # Read the full transcript text from file
            with open(file_path, "r", encoding="utf-8") as f:
                transcript = f.read()

            meeting_type = "general"
            if not args.disable_auto_classification:
                # Try to auto classify the meeting type based off a small snippet of text
                classifier_prompt = CLASSIFIER_PROMPT["classifier"].format(context=context, text=transcript[:2000])
                meeting_type = classify_meeting(session=llm_session, model=args.model, prompt=classifier_prompt)

                # track token usage
                token_usage += estimate_token_count(classifier_prompt)
                output_token_usage += estimate_token_count(meeting_type)
                print(f"🧠 Auto-detected meeting type: {meeting_type}")

            first_pass_id = args.first_prompt or PROMPT_LOOKUP.get(meeting_type, PROMPT_LOOKUP["general"])["first"]
            multi_pass_id = args.second_prompt or PROMPT_LOOKUP.get(meeting_type, PROMPT_LOOKUP["general"])["multi"]
            final_pass_id = args.final_prompt or PROMPT_LOOKUP.get(meeting_type, PROMPT_LOOKUP["general"])["final"]

            print(f"🧩 Processing transcript: {base_name}...")

            # Loop to dynamically summarize text, possibly recursively summarizing summaries
            pass_num = 1  # Current pass counter
            while pass_num <= args.max_pass:
                # Select prompt type: first pass uses 'prompt_type_first', subsequent use 'prompt_type_second'
                prompt_type = first_pass_id if pass_num == 1 else multi_pass_id

                enable_chunk_writer = False
                if enable_chunk_writer:
                    # Prepare transcript for RAG if chunk writer enabled
                    # This will use half the max tokens per chunk with about 16.7% overlap
                    chunks = prepare_text_chunk(transcript, int(args.max_tokens / 2), int(args.max_tokens / 6))
                    write_chunk_files(chunks, output_dir, meeting_type, base_name)

                chunks_only = False
                if chunks_only:
                    break  # exit the while loop

                # Prepare transcript for summarization by chunking contents if necessary
                # If text too large to summarize in one call, break into chunks and summarize each chunk
                chunks = prepare_text_chunk(transcript, max_tokens=args.max_tokens)
                chunk_summaries = []

                count = 0
                for chunk in chunks:
                    count += 1
                    print(f"Pass {pass_num} processing chunk {count}/{len(chunks)} with [{prompt_type}]")
                    # Inject context back into the chunk when operating on large files
                    # The context is in the file but with each chunk the llm looses context so add some back
                    speaker_matches = scan_transcript_for_names(chunk, known_speakers_permutations)
                    speaker_context = generate_known_speakers_context(speaker_matches)
                    chunk = speaker_context + "\n\n" + chunk

                    # for person, hits in speaker_matches.items():
                    #     print(f"{person}: {sorted(hits)}")
                    # print("\n\n")

                    # print(speaker_context)
                    # print("\n")

                    # Summarize each chunk using the main prompt
                    prompt_template = PROMPTS.get(prompt_type, PROMPTS["detailed"])
                    chunk_prompt = prompt_template.format(context=context or "No context provided.", text=chunk)

                    chunk_summary = summarize_transcript(llm_session,args.model, chunk_prompt)
                    chunk_summaries.append(chunk_summary)

                    # Update token usage counts
                    token_usage += estimate_token_count(chunk)
                    output_token_usage += estimate_token_count(chunk_summary)
                # End for loop

                # Combine chunk summaries into a single string for final summarization
                summary = "\n\n".join(chunk_summaries)

                # If summary is small enough or this is the last pass, finalize and exit loop
                if estimate_token_count(summary) <= args.max_tokens or len(chunk_summaries) == 1 or pass_num == args.max_pass:
                    print(f"Processing final summary with [{final_pass_id}]")
                    prompt_template = PROMPTS.get(final_pass_id, PROMPTS["detailed"])
                    final_prompt = prompt_template.format(context=context or "No context provided.", text=summary)

                    transcript = summarize_transcript(llm_session, args.model, final_prompt)

                    # Update token usage counts
                    token_usage += estimate_token_count(transcript)
                    output_token_usage += estimate_token_count(transcript)
                    break

                # Increment pass number and repeat loop
                transcript = summary
                pass_num += 1
            # End while loop

            chunks_only = False
            if chunks_only:
                continue # continue to next file

            # After all passes or break condition, final_summary is stored in transcript variable
            final_summary = transcript

            # Calculate total tokens used and estimate API cost
            total_tokens = token_usage + output_token_usage
            estimated_cost = calculate_cost_openai_gpt4o(token_usage, output_token_usage)

            # Print token usage and estimated cost for transparency
            print(f"\n📊 Token usage:")
            print(f"  - tokIn: {token_usage} tokOut: {output_token_usage} - Total tokens: {total_tokens}")
            print(f"💰 Estimated GPT-4o API cost: ${estimated_cost:.4f}")

            # Write the final summary to an output file
            final_basename = f"{base_name}_final_summary"
            success = write_summary_file(final_summary, final_basename, output_dir, args.output_format)
            if success:
                print("\n📝 Final Summary:\n")
                print(final_summary)
                print(f"\n✅ Final summary written to: {os.path.join(output_dir, final_basename)}.{'md' if args.output_format == 'markdown' else 'yaml'}")
            else:
                print("❌ Failed to write final summary file.")

        except Exception as e:
            # Catch and report any errors reading file or processing
            print(f"❌ Error: {e}")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
