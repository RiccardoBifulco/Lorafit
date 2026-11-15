"""Command line interface and report rendering."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

from lora_fitcheck._check import Report, check
from lora_fitcheck._demo import build_demo


def _as_json(report: Report) -> str:
    return json.dumps(
        {
            "schema_version": 1,
            "status": report.status,
            "summary": {
                "modules": report.modules,
                "matched": report.matched,
                "base_shards": report.base_shards,
                "base_shards_read": report.base_shards_read,
                "metadata_bytes": report.metadata_bytes,
            },
            "declared_base": report.declared_base,
            "findings": [asdict(finding) for finding in report.findings],
        },
        indent=2,
    )


def _as_text(report: Report, max_findings: int) -> str:
    verdict = {"match": "MATCH", "mismatch": "MISMATCH", "incomplete": "INCOMPLETE"}[report.status]
    lines = [
        f"{verdict}: {report.matched}/{report.modules} modules match; "
        f"read {report.base_shards_read}/{report.base_shards} base headers"
    ]
    if report.declared_base:
        lines.append(f"  declared base (unverified): {ascii(report.declared_base)}")
    for item in report.findings[:max_findings]:
        subject = f" {ascii(item.module)}:" if item.module is not None else ""
        path = f" [{ascii(item.path)}]" if item.path is not None else ""
        detail = item.detail.encode("ascii", "backslashreplace").decode("ascii")
        detail = "".join(c if c.isprintable() else f"\\x{ord(c):02x}" for c in detail)
        lines.append(f"  {item.code}:{subject} {detail}{path}")
    if len(report.findings) > max_findings:
        lines.append(
            f"  ... {len(report.findings) - max_findings} more; use --format json for all findings"
        )
    return "\n".join(lines)


def _positive(value: str) -> int:
    try:
        size = int(value)
    except ValueError:
        size = 0
    if size < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return size


def _demo() -> int:
    try:
        with TemporaryDirectory(prefix="lora-fitcheck-") as directory:
            base, wrong, adapter = build_demo(Path(directory))
            good, bad = check(base, adapter), check(wrong, adapter)
            print(f"$ lora-fitcheck check base adapter\n{_as_text(good, 12)}\n")
            print(f"$ lora-fitcheck check wrong-base adapter\n{_as_text(bad, 12)}")
            return 0 if (good.exit_code, bad.exit_code) == (0, 1) else 2
    except OSError as exc:
        print(f"lora-fitcheck: could not create demo files: {exc}", file=sys.stderr)
        return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="lora-fitcheck",
        description="Check the names, shapes and rank of a saved PEFT LoRA against a local base.",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {version('lora-fitcheck')}"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("check", help="check a local base and adapter")
    command.add_argument(
        "base", type=Path, help="model directory, .safetensors file or shard index"
    )
    command.add_argument("adapter", type=Path, help="PEFT adapter directory")
    command.add_argument("--format", choices=("text", "json"), default="text", help="output format")
    command.add_argument(
        "--max-findings", type=_positive, default=12, help="text finding limit (default: 12)"
    )
    commands.add_parser("demo", help="run a match and a mismatch on bundled synthetic examples")
    args = parser.parse_args(argv)
    if args.command == "demo":
        return _demo()
    report = check(args.base, args.adapter)
    print(_as_json(report) if args.format == "json" else _as_text(report, args.max_findings))
    return report.exit_code
