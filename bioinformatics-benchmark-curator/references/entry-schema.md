# Benchmark Entry Schema

## Required Fields
- `name` (string): Benchmark title.
- `task_category` (string): Section grouping key.
- `summary` (string): One-sentence description.
- `paper_url` (string): Publication/preprint URL.
- `code_url` (string): Source code URL.
- `modality` (string): Data modality.
- `license` (string): Distribution license or `unknown`.
- `year` (integer): Publication year.

## Optional Fields
- `dataset_url` (string)
- `leaderboard_url` (string)
- `metrics` (array of strings)
- `tags` (array of strings)
- `notes` (string)

## Example Entry
```json
{
  "name": "Genomics Long-Range Benchmark",
  "task_category": "genomics",
  "summary": "Evaluate sequence models on long-context regulatory prediction tasks.",
  "paper_url": "https://arxiv.org/abs/2401.00001",
  "code_url": "https://github.com/example/glrb",
  "dataset_url": "https://zenodo.org/records/1234567",
  "leaderboard_url": "https://paperswithcode.com/sota/example",
  "modality": "sequence",
  "license": "CC-BY-4.0",
  "metrics": ["AUROC", "AUPRC"],
  "tags": ["long-context", "regulatory"],
  "notes": "Contains train/dev/test splits for human and mouse.",
  "year": 2024
}
```
