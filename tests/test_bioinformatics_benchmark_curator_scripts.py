import importlib.util
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "bioinformatics-benchmark-curator" / "scripts"
ASSETS_DIR = ROOT / "bioinformatics-benchmark-curator" / "assets"


def run(cmd):
    return subprocess.run(cmd, text=True, capture_output=True, check=False)


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_validate_sample_benchmarks_passes():
    result = run(
        [
            sys.executable,
            str(SCRIPT_DIR / "validate_benchmarks.py"),
            "--input",
            str(ASSETS_DIR / "sample_benchmarks.json"),
        ]
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Validation passed" in result.stdout


def test_render_generates_markdown(tmp_path):
    out = tmp_path / "README.generated.md"
    result = run(
        [
            sys.executable,
            str(SCRIPT_DIR / "render_awesome_readme.py"),
            "--input",
            str(ASSETS_DIR / "sample_benchmarks.json"),
            "--output",
            str(out),
        ]
    )
    assert result.returncode == 0, result.stdout + result.stderr
    content = out.read_text(encoding="utf-8")
    assert "# Awesome Bioinformatics Benchmarks" in content
    assert "### genomics" in content
    assert "tags:" in content


def test_validate_rejects_duplicate_names(tmp_path):
    bad = {
        "items": [
            {
                "name": "Dup",
                "task_category": "genomics",
                "summary": "A",
                "paper_url": "https://example.com/paper",
                "code_url": "https://example.com/code",
                "modality": "sequence",
                "license": "MIT",
                "year": 2023,
            },
            {
                "name": "dup",
                "task_category": "genomics",
                "summary": "B",
                "paper_url": "https://example.com/paper2",
                "code_url": "https://example.com/code2",
                "modality": "sequence",
                "license": "MIT",
                "year": 2024,
            },
        ]
    }
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(bad), encoding="utf-8")

    result = run(
        [sys.executable, str(SCRIPT_DIR / "validate_benchmarks.py"), "--input", str(path)]
    )
    assert result.returncode == 1
    assert "duplicate name" in result.stdout


def test_validate_rejects_empty_strings_and_bad_tags(tmp_path):
    bad = {
        "items": [
            {
                "name": " ",
                "task_category": "genomics",
                "summary": "A",
                "paper_url": "https://example.com/paper",
                "code_url": "https://example.com/code",
                "modality": "sequence",
                "license": "MIT",
                "year": 2023,
                "tags": ["ok", ""],
            }
        ]
    }
    path = tmp_path / "bad_fields.json"
    path.write_text(json.dumps(bad), encoding="utf-8")

    result = run(
        [sys.executable, str(SCRIPT_DIR / "validate_benchmarks.py"), "--input", str(path)]
    )
    assert result.returncode == 1
    assert "non-empty string" in result.stdout
    assert "list of non-empty strings" in result.stdout


def test_import_parser_extracts_items():
    module = _load_module(
        SCRIPT_DIR / "import_from_awesome_readme.py", "import_from_awesome_readme"
    )
    markdown = """
## Genomics
- **BenchA**: short summary (2024) — [paper](https://example.com/p) | [code](https://example.com/c)
"""
    items = module.parse_markdown(markdown)
    assert len(items) == 1
    assert items[0]["name"] == "BenchA"
    assert items[0]["task_category"] == "genomics"
    assert items[0]["year"] == 2024
