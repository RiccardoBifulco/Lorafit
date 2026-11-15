import json
import subprocess
import sys
from pathlib import Path

from lora_fitcheck._demo import build_demo


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "lora_fitcheck", *args],
        capture_output=True,
        encoding="utf-8",
        check=False,
    )


def test_readme_demo() -> None:
    result = _run("demo")
    assert result.returncode == 0, result.stderr
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    assert result.stdout.strip() in readme


def test_json_exit_codes(tmp_path: Path) -> None:
    base, wrong, adapter = build_demo(tmp_path / "demo with spaces")
    for code, path, status in (
        (0, base, "match"),
        (1, wrong, "mismatch"),
        (2, tmp_path / "missing", "incomplete"),
    ):
        result = _run("check", str(path), str(adapter), "--format", "json")
        assert (result.returncode, result.stderr) == (code, "")
        report = json.loads(result.stdout)
        assert (report["schema_version"], report["status"]) == (1, status)
        if code == 1:
            assert report["findings"][0]["actual_shape"] == [8, 16]
            assert report["summary"]["metadata_bytes"] > 0


def test_malformed_config_has_actionable_error(tmp_path: Path) -> None:
    base, _, adapter = build_demo(tmp_path)
    (adapter / "adapter_config.json").write_text('{"r":2,"r":8}', encoding="utf-8")
    result = _run("check", str(base), str(adapter))
    assert (result.returncode, result.stderr) == (2, "")
    assert "invalid_config: invalid JSON: duplicate JSON key 'r'" in result.stdout


def test_invalid_argument_does_not_print_traceback() -> None:
    result = _run("check", "base", "adapter", "--max-findings", "0")
    assert result.returncode == 2
    assert "expected a positive integer" in result.stderr
    assert "Traceback" not in result.stderr
