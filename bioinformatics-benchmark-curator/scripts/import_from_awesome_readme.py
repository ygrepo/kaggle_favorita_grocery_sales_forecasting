#!/usr/bin/env python3
"""Import benchmark entries from an awesome-list style README."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from urllib.request import urlopen

BULLET_RE = re.compile(r"^-\s+\*\*(?P<name>.+?)\*\*\s*(?P<body>.+)$")
LINK_RE = re.compile(r"\[(?P<label>[^\]]+)\]\((?P<url>https?://[^)]+)\)")
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


def read_text(path_or_url: str) -> str:
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        with urlopen(path_or_url) as response:  # nosec: B310 - expected user-provided URL for curated source
            return response.read().decode("utf-8")
    return Path(path_or_url).read_text(encoding="utf-8")


def normalize_category(heading: str) -> str:
    cleaned = heading.strip().lower()
    cleaned = re.sub(r"[^a-z0-9\s-]", "", cleaned)
    return re.sub(r"\s+", "-", cleaned)


def clean_summary(body: str) -> str:
    no_links = LINK_RE.sub("", body)
    no_links = re.sub(r"\|", " ", no_links)
    no_links = re.sub(r"_\(tags:[^)]+\)_", "", no_links)
    no_links = re.sub(r"\(\d{4}\)", "", no_links)
    no_links = no_links.replace("—", " ")
    no_links = re.sub(r"\s+", " ", no_links)
    return no_links.strip(" :-") or "TBD"


def parse_markdown(markdown: str) -> list[dict]:
    items: list[dict] = []
    current_category = "uncategorized"

    for raw_line in markdown.splitlines():
        line = raw_line.strip()
        if line.startswith("### "):
            candidate = normalize_category(line[4:])
            if candidate:
                current_category = candidate
            continue
        if line.startswith("## "):
            candidate = normalize_category(line[3:])
            if candidate not in {"contents", "benchmarks-by-category"} and candidate:
                current_category = candidate
            continue
        if not line.startswith("- **"):
            continue

        match = BULLET_RE.match(line)
        if not match:
            continue

        name = match.group("name").strip()
        body = match.group("body").strip()
        links = {m.group("label").lower(): m.group("url") for m in LINK_RE.finditer(body)}
        summary = clean_summary(body)

        year_match = YEAR_RE.search(line)
        year = int(year_match.group()) if year_match else 2000

        items.append(
            {
                "name": name,
                "task_category": current_category,
                "summary": summary,
                "paper_url": links.get("paper", "https://example.com/paper"),
                "code_url": links.get("code", "https://example.com/code"),
                "modality": "unknown",
                "license": "unknown",
                "year": year,
            }
        )

    return items


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        required=True,
        help="README markdown URL or local file path (for example raw README.md from awesome-bioinformatics-benchmarks)",
    )
    parser.add_argument("--output", required=True, type=Path, help="Path for generated JSON")
    parser.add_argument("--title", default="Awesome Bioinformatics Benchmarks")
    parser.add_argument("--description", default="Imported benchmark entries from an awesome-style README.")
    args = parser.parse_args()

    markdown = read_text(args.source)
    payload = {
        "title": args.title,
        "description": args.description,
        "items": parse_markdown(markdown),
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Imported {len(payload['items'])} items into {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
