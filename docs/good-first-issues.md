# Good first issues

These tasks are published on GitHub and welcome first-time contributors. Read the issue's
comments and linked PRs for its current availability. Comment on the issue to introduce
yourself and coordinate your approach with [@nazek52](https://github.com/nazek52).

| Task | Skills | Issue |
| --- | --- | --- |
| Official BF16 export example | Python and small tensor fixtures | [#1](https://github.com/nazek52/Lorafit/issues/1) |
| Filename suggestions for variants | Python and CLI diagnostics | [#2](https://github.com/nazek52/Lorafit/issues/2) |
| Missing and misindexed shard walkthrough | Markdown, JSON, and running CLI examples | [#3](https://github.com/nazek52/Lorafit/issues/3) |

All three tasks can be developed on CPU. Follow [CONTRIBUTING](../CONTRIBUTING.md) for setup
and PR checks. The [roadmap](roadmap.md) also includes a larger PEFT feature.

### Add a BF16 export fixture made by the official SafeTensors writer

[Read the task and completion criteria in issue #1](https://github.com/nazek52/Lorafit/issues/1).

The tests currently synthesize BF16 metadata. Add a small repeatable example using the official
writer in a separate development environment. Document the command and prove that LoRAFit
accepts mixed base/adapter storage dtypes within the supported float set. Keep the wheel free
of dependencies and avoid checking large weights into Git.

### Point people to the right file when a directory contains only a variant

[Read the task and completion criteria in issue #2](https://github.com/nazek52/Lorafit/issues/2).

`check` requires an explicit path for alternative filenames. On `base_not_found`, show up to
three suitable filenames in the directory, if any. Do not auto-select one. Add a test for a
directory that contains `model.fp16.safetensors` but no default export.

### Add a JSON report example for a missing or misindexed shard

[Read the task and completion criteria in issue #3](https://github.com/nazek52/Lorafit/issues/3).

Using `examples/make_demo.py`, document a reproducible edit to a temporary fixture and show how
a pipeline can distinguish exit `2` from a shape mismatch. Reference `docs/report.md` instead
of re-documenting the whole schema.
