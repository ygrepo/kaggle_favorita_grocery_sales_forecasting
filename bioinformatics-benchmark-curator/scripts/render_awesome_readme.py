#!/usr/bin/env python3
"""Render an awesome-list style benchmark README from JSON metadata."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def slugify(text: str) -> str:
    return text.strip().lower().replace(" ", "-")


def render_item(item: dict) -> str:
    links = [f"[paper]({item['paper_url']})", f"[code]({item['code_url']})"]
    if item.get("dataset_url"):
        links.append(f"[dataset]({item['dataset_url']})")
    if item.get("leaderboard_url"):
        links.append(f"[leaderboard]({item['leaderboard_url']})")

    tail = f" — {' | '.join(links)}"
    return f"- **{item['name']}** ({item['year']}): {item['summary']}{tail}"


def render_markdown(data: dict) -> str:
    title = data.get("title", "Awesome Bioinformatics Benchmarks")
    description = data.get("description", "Curated benchmark resources.")
    items = data.get("items", [])

    grouped = defaultdict(list)
    for item in items:
        grouped[item["task_category"]].append(item)

    for category_items in grouped.values():
        category_items.sort(key=lambda x: (-x["year"], x["name"].lower()))

    categories = sorted(grouped.keys())
    lines: list[str] = [f"# {title}", "", description, "", "## Contents"]
    for category in categories:
        lines.append(f"- [{category}](#{slugify(category)})")

    lines.extend(["", "## Benchmarks by Category", ""])
    for category in categories:
        lines.append(f"### {category}")
        for item in grouped[category]:
            lines.append(render_item(item))
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Path to benchmark JSON")
    parser.add_argument("--output", required=True, type=Path, help="Output markdown path")
    args = parser.parse_args()

    data = load_json(args.input)
    output = render_markdown(data)
    args.output.write_text(output, encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
