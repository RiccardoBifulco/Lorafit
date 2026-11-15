# Contributing

Use Python 3.10 or later in a fresh environment:

```sh
python -m pip install ".[dev]"
python scripts/check.py
python scripts/verify_dist.py
```

The first script runs Ruff lint and format checks, mypy in strict mode, and pytest. The second
builds the source archive and wheel and runs the installed CLI in a fresh virtual environment.
Repeat the install after changing the package. Editable installs (`-e`) are also supported;
use `PYTHONUTF8=1` when an older Windows Python cannot decode the checkout path in its `.pth` file.
To format changes, run `python -m ruff format`. See [design](docs/design.md) for the scope and
[good first issues](docs/good-first-issues.md) for focused contributions.

For a bug report, include `lora-fitcheck --version`, your platform, the command, the exit code,
and a sanitized report. Prefer a tiny synthetic reproducer over attaching private weights or
sharing machine paths. Keep fixes focused and add a test that fails before a behavior fix.

Keep the core install dependency free and the checker offline. Confirm surprises with genuine
exports from the affected format/version when possible. Cite format specs and do not paste
third-party code unless the contribution includes license review and attribution.

Use descriptive commits such as `fix: report missing base shard`. By contributing you agree to
license the contribution under the project's [MIT license](LICENSE) and follow its
[code of conduct](CODE_OF_CONDUCT.md).
