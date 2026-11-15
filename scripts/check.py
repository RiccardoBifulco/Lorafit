"""Run the same lint, typing, and test commands used by CI."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    checks = (("ruff", "check"), ("ruff", "format", "--check"), ("mypy",), ("pytest", "-q"))
    for args in checks:
        result = subprocess.run([sys.executable, "-m", *args], cwd=ROOT, check=False)
        if result.returncode:
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
