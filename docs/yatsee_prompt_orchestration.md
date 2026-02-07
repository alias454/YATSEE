# Prompt Orchestration Overview

This guide explains how YATSEE manages the intelligence layer of the system. Instead of sending a full transcript to a model and hoping for a usable summary, YATSEE uses a structured, multi-stage process to ensure accuracy, consistency, and domain-specific formatting.

---

## 1. Prompt Hierarchy

YATSEE resolves prompt instructions in a layered order. This allows global defaults, job-level customization, and entity-specific overrides to coexist without conflict.

Priority order:

1. **Entity-Specific Prompts**  
   `data/<entity>/prompts/<job_type>/prompts.toml`

2. **Global Job Defaults**  
   `prompts/<job_type>/prompts.toml`

3. **System Fallbacks**  
   Hardcoded defaults in the Python pipeline.

This structure allows each city, committee, or department to have its own customized “voice” or summary style without affecting any others.

### Prompt override layout example:
```
./prompts/                      # default prompts for all entities
  └── research/
      └── prompts.toml          # default prompts & routing for 'research' job type

./data/
  └── defined_entity/           # entity-specific data
      └── prompts/
          └── research/
              └── prompts.toml  # full override for defined_entity 'research' job type

./data/
  └── generic_entity/           # another entity with no override
      └── prompts/
          └── research/
              # no file, falls back to default in prompts/research/prompts.toml

**Behavior**:

  - Loader first checks `data/<entity>/prompts/<job_type>/prompts.toml`.  
  - If found → full override of defaults.  
  - If not found → fall back to `prompts/<job_type>/prompts.toml`.
```

---

## 2. Automated Classification (Prompt Router)

Before summarization begins, YATSEE performs a classification step. Using an initial snippet from the transcript plus context about the entity, it determines the type of meeting.

**Mechanism:**
- The model scans for domain indicators such as “Budget,” “Appropriation,” “Zoning,” “Variance,” etc.
- Once classified, the system references `[prompt_router]` in the configuration to select the correct prompt set for the meeting type.

**Outcome:**  
A batch of mixed files automatically routes to the appropriate templates such as finance, zoning, council, committee, or any custom meeting category.

### Config file routing/load order
```
Global TOML
    |
    +--> Entity handle
            |
            +--> Local config (hotwords, divisions, data_path)
                    |
                    +--> Pipeline stage (downloads, audio, transcripts)
```


---

## 3. Multi-Pass Summarization Workflow

Government transcripts routinely exceed the context window of most models. YATSEE processes them through a structured refinery pipeline that preserves detail across multiple passes.

### Pass 1: First-Pass  
The transcript is split into chunks. Each chunk is summarized using high-detail extraction prompts. This produces granular notes containing actions, motions, votes, dollar amounts, and speaker-specific content.

### Pass 2: Multi-Pass  
If the volume of notes is still too large, the system performs a consolidation pass. Summaries are grouped and compressed while preserving structure so no decisions or financial details are lost.

### Pass 3: Final-Pass  
The model takes all refined notes and produces a polished report with enforced sections such as:
- Decisions
- Motions and Votes
- Public Comments
- Recommendations
- Follow-up Items

---

## 4. Density-Based Chunking

Instead of cutting the transcript arbitrarily, YATSEE uses semantic hotspots to determine safe chunk boundaries.

### Mechanism
A configurable `density_keywords` list highlights areas involving:
- motions  
- seconds  
- votes  
- ordinances  
- resolutions  
- financial amounts  

Chunking attempts to keep these hotspots intact so context is not split across boundaries.

### Benefit
The model always sees the full action surrounding a meeting decision, improving accuracy and reducing hallucinations.

---

## 5. Job Types & Custom Behavior

The pipeline is driven by the `--job-type` argument. A job type defines:
- which prompt hierarchy to use  
- which extraction rules apply  
- what formatting is expected in the final output  

### Examples
- **summary**: Default meeting summarization workflow.  
- **research**: A custom job type with prompts optimized for legal, policy, or historical investigations.  
- **custom audits**: Any custom job type created simply by adding a folder with a new `prompts.toml`.

No Python modifications are required to introduce new specialized tasks. It is 100% config driven.

---

## Summary: How the Intelligence Layer Works

1. **Classify**  
   Identify the meeting type automatically using transcript cues.

2. **Chunk**  
   Split the transcript into size-appropriate sections while preserving semantic clusters like motions and votes.

3. **Extract**  
   Perform detailed extraction on each chunk for names, amounts, actions, decisions, and topic markers.

4. **Refine**  
   Consolidate hundreds of notes into coherent, structured summaries.

5. **Audit**  
   Produce a final, standardized Markdown report.

---

## Final Notes

The orchestration system enables YATSEE to process multi-hour meetings reliably. By combining classification, density-aware chunking, and multi-pass refinement, the pipeline scales to very large transcripts without losing detail or structure. The output is consistent across meeting types and remains customizable for each entity and job type.
