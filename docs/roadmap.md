# Roadmap

LoRAFit checks a saved LoRA adapter against a local SafeTensors base before model loading.
Development focuses on format compatibility and useful diagnostics for that workflow.
The core runs offline with the Python standard library.

## Open contributions

| Work | User benefit | Task |
| --- | --- | --- |
| Official BF16 export example | Reproduce mixed-dtype structural checks with the official writer | [#1](https://github.com/nazek52/Lorafit/issues/1) |
| Explicit variant filename suggestions | Find the intended checkpoint after a missing-default diagnostic | [#2](https://github.com/nazek52/Lorafit/issues/2) |
| Incomplete JSON report walkthrough | Handle missing shards and bad index mappings in a pipeline | [#3](https://github.com/nazek52/Lorafit/issues/3) |
| PEFT per-module ranks | Check dense adapters that use different ranks for different modules | [#4](https://github.com/nazek52/Lorafit/issues/4) |

The issue threads track scope, current availability, linked PRs, and verification results.
Start with the [first-contribution guide](../CONTRIBUTING.md). The first three issues have the
`good first issue` label. Per-module rank support needs familiarity with PEFT's matching rules.

## Propose the next task

Bring a concrete input layout or workflow to an
[improvement request](https://github.com/nazek52/Lorafit/issues/new?template=enhancement.yml).
Include the command, expected result, current output, and a small reproducible example when
possible. Base revision verification is a candidate for a later proposal, with its own
evidence requirements.

Feature support is documented in the [README](../README.md#scope) as changes are released.
