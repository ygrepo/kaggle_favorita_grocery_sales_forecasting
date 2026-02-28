import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "bioinformatics-benchmark-curator" / "scripts"
ASSETS_DIR = ROOT / "bioinformatics-benchmark-curator" / "assets"


def run(cmd):
    return subprocess.run(cmd, text=True, capture_output=True, check=False)


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
