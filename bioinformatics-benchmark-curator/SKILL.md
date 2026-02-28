---
name: bioinformatics-benchmark-curator
description: Curate, normalize, validate, and generate awesome-list style benchmark pages for bioinformatics (or adjacent scientific ML domains). Use when building or maintaining benchmark catalogs from papers/repos, importing from existing awesome README pages, enforcing a metadata schema, checking links, and rendering grouped Markdown sections.
---

# Bioinformatics Benchmark Curator

Use this skill to build or maintain benchmark catalogs similar to `awesome-bioinformatics-benchmarks`.

## Workflow
1. (Optional) Bootstrap JSON from an existing awesome README:
   - `python scripts/import_from_awesome_readme.py --source <README.md or URL> --output assets/benchmarks.json`
2. Validate entries:
   - `python scripts/validate_benchmarks.py --input <path>.json`
3. Render the catalog page:
   - `python scripts/render_awesome_readme.py --input <path>.json --output README.generated.md`
4. Adjust categories and naming conventions with `references/taxonomy.md`.

## Required Fields per Item
- `name`, `task_category`, `summary`, `paper_url`, `code_url`, `modality`, `license`, `year`

Optional fields:
- `dataset_url`, `leaderboard_url`, `metrics`, `notes`, `tags`

## Rules
- URLs must start with `http://` or `https://`.
- Required string fields must be non-empty.
- `metrics` and `tags` must be lists of non-empty strings.
- Duplicate benchmark names (case-insensitive) are rejected.
- Render order: category ascending, then year descending, then name ascending.

## Resources
- Schema details: `references/entry-schema.md`
- Taxonomy guidance: `references/taxonomy.md`
- Template starter: `assets/awesome_readme_template.md`
