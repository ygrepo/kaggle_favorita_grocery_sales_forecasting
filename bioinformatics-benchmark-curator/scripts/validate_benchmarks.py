#!/usr/bin/env python3
"""Validate benchmark entries for awesome-list style generation."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

REQUIRED_FIELDS = {
    "name": str,
    "task_category": str,
    "summary": str,
    "paper_url": str,
    "code_url": str,
    "modality": str,
    "license": str,
    "year": int,
}
URL_FIELDS = {"paper_url", "code_url", "dataset_url", "leaderboard_url"}
URL_RE = re.compile(r"^https?://")
LIST_OF_STR_FIELDS = {"metrics", "tags"}


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate(data: dict) -> list[str]:
    errors: list[str] = []
    items = data.get("items")
    if not isinstance(items, list):
        return ["top-level 'items' must be a list"]

    seen_names: set[str] = set()
    current_year = datetime.now().year + 1
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            errors.append(f"items[{i}] must be an object")
            continue

        for field, expected_type in REQUIRED_FIELDS.items():
            if field not in item:
                errors.append(f"items[{i}] missing required field: {field}")
                continue
            if not isinstance(item[field], expected_type):
                errors.append(
                    f"items[{i}].{field} must be {expected_type.__name__}, got {type(item[field]).__name__}"
                )
                continue

            if expected_type is str and not item[field].strip():
                errors.append(f"items[{i}].{field} must be a non-empty string")

        name = str(item.get("name", "")).strip().lower()
        if name:
            if name in seen_names:
                errors.append(f"items[{i}] duplicate name: {item.get('name')}")
            seen_names.add(name)

        year = item.get("year")
        if isinstance(year, int) and (year < 1990 or year > current_year):
            errors.append(f"items[{i}].year out of range: {year}")

        for url_field in URL_FIELDS:
            if url_field in item:
                value = item.get(url_field)
                if not isinstance(value, str) or not URL_RE.match(value):
                    errors.append(f"items[{i}].{url_field} must start with http:// or https://")

        for list_field in LIST_OF_STR_FIELDS:
            if list_field in item:
                value = item.get(list_field)
                if not isinstance(value, list) or any(not isinstance(v, str) or not v.strip() for v in value):
                    errors.append(f"items[{i}].{list_field} must be a list of non-empty strings")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Path to benchmark JSON")
    args = parser.parse_args()

    data = load_json(args.input)
    errors = validate(data)

    if errors:
        print("Validation failed:")
        for err in errors:
            print(f"- {err}")
        return 1

    print("Validation passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
