---
name: bioinformatics-benchmark-curator
description: Curate, normalize, validate, and generate awesome-list style benchmark pages for bioinformatics (or adjacent scientific ML domains). Use when building or maintaining benchmark catalogs from papers/repos, enforcing a consistent metadata schema, checking links, and rendering grouped Markdown sections like awesome-bioinformatics-benchmarks.
---

# Bioinformatics Benchmark Curator

## Overview
Use this skill to implement or maintain a benchmark curation agent that turns structured benchmark metadata into a high-quality awesome-list style README.
Prefer scripted generation/validation for deterministic formatting and quality checks.

## Quick Start Workflow
1. Collect benchmark items into `assets/sample_benchmarks.json` schema (or your own JSON file).
2. Validate data quality and links with:
   - `python scripts/validate_benchmarks.py --input <path>.json`
3. Render Markdown page with:
   - `python scripts/render_awesome_readme.py --input <path>.json --output README.generated.md`
4. Review grouped sections and edit taxonomy in `references/taxonomy.md` if needed.

## Required Entry Schema
Each benchmark item should include:
- `name`: Human-readable benchmark name.
- `task_category`: Top-level grouping (e.g., genomics, protein-structure, single-cell).
- `summary`: One-sentence purpose.
- `paper_url`: Canonical publication URL.
- `code_url`: Repository or project URL.
- `modality`: Data modality (sequence, structure, expression, multimodal).
- `license`: Dataset/benchmark license (or `unknown`).
- `year`: Integer release/publication year.

Optional fields:
- `dataset_url`, `leaderboard_url`, `metrics`, `notes`, `tags`.

See `references/entry-schema.md` for full details and examples.

## Curation Rules
- Use canonical URLs (`https://...`) and prefer stable landing pages.
- Keep summaries concise (one sentence; avoid marketing language).
- Group by `task_category`, then sort by `year` descending and `name` ascending.
- Reject duplicate names (case-insensitive) during validation.
- Preserve neutral wording and cite benchmark scope/limitations in `notes` if relevant.

## Resource Usage Guide
- Read `references/taxonomy.md` when deciding task groups and naming conventions.
- Use `scripts/validate_benchmarks.py` before rendering to catch missing fields and malformed links.
- Use `scripts/render_awesome_readme.py` to regenerate markdown after edits.
- Use `assets/awesome_readme_template.md` as the output structure baseline.

## Example Commands
```bash
python scripts/validate_benchmarks.py --input assets/sample_benchmarks.json
python scripts/render_awesome_readme.py \
  --input assets/sample_benchmarks.json \
  --output assets/sample_README.generated.md
```
