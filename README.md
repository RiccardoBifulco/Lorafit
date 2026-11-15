# LoRA Fitcheck

![Version 0.1.0](https://img.shields.io/badge/version-0.1.0-blue) [![MIT license](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

**Find LoRA checkpoint mismatches before you load the model.**

```text
$ lora-fitcheck check base adapter
MATCH: 2/2 modules match; read 2/2 base headers

$ lora-fitcheck check wrong-base adapter
MISMATCH: 1/2 modules match; read 2/2 base headers
  shape_mismatch: 'model.layers.0.self_attn.v_proj': adapter B @ A gives (8, 16); base weight is (4, 16)
```

These are the two runs from the offline demo. The adapter is the same in both; the second
base has a narrower projection. An adapter and a base can each be valid files and still
fail together. Fitcheck points to the mismatched tensor using the files you already have,
without importing a model runtime, using a GPU, or reading tensor payload bytes.

## Quickstart

From the repository root, in a Python 3.10+ environment:

```sh
python -m pip install .
lora-fitcheck demo
```

The demo creates and checks two tiny synthetic examples in a temporary directory. No model
downloads or API keys are needed. This release is distributed from source and as a wheel;
it is not on PyPI. If your scripts directory is not on `PATH`, use `python -m lora_fitcheck`.

## Check your files

```sh
# A base exported by Transformers and a directory saved by PEFT:
lora-fitcheck check ./base-model ./adapter

# Select an explicit weight file (for example, when a directory has two variants):
lora-fitcheck check ./base-model/model.fp16.safetensors ./adapter

# Check a multi-shard base as a step in your pipeline:
lora-fitcheck check ./base-model/model.safetensors.index.json ./adapter --format json
```

The adapter directory needs `adapter_config.json` and `adapter_model.safetensors`.
For a base directory, Fitcheck looks for `model.safetensors` or
`model.safetensors.index.json`; pass the chosen file if both exist. The checker follows
the index only to the shards that should hold the adapter's targets. The header count
reports how many were opened out of the total referenced by the index.

| Exit | JSON status | Meaning |
| --- | --- | --- |
| `0` | `match` | All stored A/B pairs fit the base's exact tensor names, shapes and configured rank. |
| `1` | `mismatch` | The check found a mismatch such as an absent target, a missing A/B partner or a different shape. |
| `2` | `incomplete` | A required input was missing, unreadable or outside the supported layout. CLI syntax errors also return `2`. |

The text report limits its finding list to 12 by default; `--max-findings` changes that.
JSON reports every finding and includes a `schema_version` for consumers. See the
[scriptable example](examples/) and the [JSON schema guide](docs/report.md).

## Which tool should I use?

| Tool | When to use it |
| --- | --- |
| **LoRA Fitcheck** | Compare a saved dense PEFT LoRA to local SafeTensors files and gate a handoff on names, shapes and rank. |
| [PEFT](https://huggingface.co/docs/peft/main/developer_guides/checkpoint) | Load, run, or merge an adapter with the actual model and test its behavior. |
| [PEFT Doctor](https://pypi.org/project/peft-doctor/) | Validate a training recipe or its LoRA configuration. |
| [Weightlint](https://github.com/thisbejim/weightlint) | Check the integrity and packaging of a standalone model bundle. |

Version 0.1 supports plain PEFT LoRA on dense floating-point 2D weights, with a uniform rank
and the standard saved key names. It does not support convolutional, quantized or embedding
layouts, DoRA, fan-in/fan-out, or per-module rank/alpha patterns. A match does not establish
the base revision, check that every intended module was trained, validate the payload, or
guarantee that a given runtime will activate the adapter. Run an actual inference smoke test
when runtime behavior matters.

## Project links

- [Discovery and comparison](docs/discovery.md), [design](docs/design.md) and [references](docs/references.md)
- [Benchmark](benchmarks/), [changelog](CHANGELOG.md) and [release notes](docs/release-notes-v0.1.0.md)
- [Contributing](CONTRIBUTING.md), [good first issues](docs/good-first-issues.md) and [MIT license](LICENSE)

Roadmap: support common `rank_pattern` layouts; add more independently generated fixtures;
consider an explicit provenance check when exporters can supply a base revision or digest.

Inspired by the concrete format examples in the PEFT and SafeTensors documentation. To cite
the software, use [CITATION.cff](CITATION.cff) or this short reference:

```text
LoRA Fitcheck contributors. LoRA Fitcheck, version 0.1.0 (software, 2026).
```
