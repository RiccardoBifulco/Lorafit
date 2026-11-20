# Contributing

New contributors are welcome. The maintainer is [@nazek52](https://github.com/nazek52).
Use the issue thread for questions and coordination so other contributors can follow along.

## Your first contribution

1. Fork the repository and follow the development setup below. Try `python -m lorafit demo`
   to see a matching adapter and a shape mismatch.
2. Choose a [starter task](docs/good-first-issues.md) or a task from the [roadmap](docs/roadmap.md).
   Check its comments and linked PRs, then leave a short comment about what you plan to work on.
   You can include the area where you would like guidance.
3. Create a branch for your change. The issue lists relevant files and
   completion criteria. Keep related questions in its thread.
4. Run the relevant example or tests and the checks below. Open a PR from your fork and link
   the task with `Fixes #NUMBER`. Include the commands you ran and their results.

Documentation, reproducible bug reports, and exporter examples are welcome alongside code.
The demo, core checks, and starter tasks run on CPU. The exporter example may use optional
PyTorch and SafeTensors packages in its development environment.

## Development setup

Clone your fork and create a branch. Replace `YOUR_GITHUB_USERNAME` with your GitHub username.

```sh
git clone https://github.com/YOUR_GITHUB_USERNAME/Lorafit.git
cd Lorafit
git switch -c your-change
```

Create a virtual environment with `python -m venv .venv`. Activate it with
`.venv\Scripts\activate` in PowerShell or `source .venv/bin/activate` in bash.

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

## Review and credit

The maintainer reviews changes against the issue's completion criteria, the supported layouts,
and the automated checks. Explain the behavior your change covers and include a runnable
example for new format support. If you use AI assistance, review the changes yourself and
describe how you verified them in the PR.

Keep your own Git author identity on your commits. Contributions are credited through the
commit history and linked PRs. The project follows the [code of conduct](CODE_OF_CONDUCT.md).

## Reporting bugs and proposing changes

For a bug report, include `lorafit --version`, your platform, the command, the exit code,
and a sanitized report. Prefer a tiny synthetic reproducer over attaching private weights or
sharing machine paths. Keep fixes focused and add a test that fails before a behavior fix.

Keep the core install dependency free and the checker offline. Confirm surprises with genuine
exports from the affected format/version when possible. Cite format specs and do not paste
third-party code unless the contribution includes license review and attribution.

Use descriptive commits such as `fix: report missing base shard`. By contributing you agree to
license the contribution under the project's [MIT license](LICENSE) and follow its
[code of conduct](CODE_OF_CONDUCT.md).
