"""Build the sdist and wheel, then test the installed wheel in isolation."""

import argparse
import os
import shutil
import subprocess
import sys
import venv
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from zipfile import ZipFile

from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[1]


def verify(work: Path, output: Path | None) -> None:
    dist = work / "dist"
    subprocess.run([sys.executable, "-m", "build", "--outdir", str(dist)], cwd=ROOT, check=True)
    (wheel,) = dist.glob("*.whl")
    (sdist,) = dist.glob("*.tar.gz")
    with ZipFile(wheel) as archive:
        (metadata,) = (name for name in archive.namelist() if name.endswith(".dist-info/METADATA"))
        text = archive.read(metadata).decode("utf-8")
        headers = text.split("\n\n", 1)[0]
        requirements = [
            Requirement(line.removeprefix("Requires-Dist:").strip())
            for line in headers.splitlines()
            if line.startswith("Requires-Dist:")
        ]
        if any(str(requirement.marker) != 'extra == "dev"' for requirement in requirements):
            raise RuntimeError("the wheel unexpectedly declares a runtime dependency")
    environment = work / "clean-venv"
    venv.EnvBuilder(with_pip=True).create(environment)
    windows = os.name == "nt"
    scripts = environment / ("Scripts" if windows else "bin")
    python = scripts / ("python.exe" if windows else "python")
    cli = scripts / ("lora-fitcheck.exe" if windows else "lora-fitcheck")
    start = perf_counter()
    subprocess.run(
        [str(python), "-m", "pip", "install", "--no-index", "--no-deps", str(wheel)],
        cwd=work,
        check=True,
    )
    subprocess.run([str(cli), "--version"], cwd=work, check=True)
    demo = subprocess.run([str(cli), "demo"], cwd=work, check=True, capture_output=True, text=True)
    if demo.stdout.strip() not in (ROOT / "README.md").read_text(encoding="utf-8"):
        raise RuntimeError("installed demo output does not match the README")
    print(demo.stdout, end="")
    print(f"Offline wheel installation and demo: {perf_counter() - start:.2f} s")
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
        for artifact in (wheel, sdist):
            shutil.copy2(artifact, output / artifact.name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, help="copy the verified wheel and sdist here")
    args = parser.parse_args()
    with TemporaryDirectory(prefix="fitcheck-release-") as tmp:
        verify(Path(tmp), args.output_dir)
