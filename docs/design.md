# Design: LoRA Fitcheck

LoRA Fitcheck helps language-model researchers check whether a saved PEFT LoRA adapter's
tensor names, shapes and rank fit a specific local base checkpoint before loading either model.

**Before:** an engineer transfers an adapter from a training node, chooses a base snapshot,
opens a model loader, and digs through a missing-key or shape error if the selection is wrong.
For handoffs, the person checking may have local files but no training environment.
**After:** the same two paths yield a short list of affected tensors and a machine-readable
verdict that can run before scheduling the next job.

## Version 0.1

- Input: a local Transformers base directory with `model.safetensors` or
  `model.safetensors.index.json`, an explicit `.safetensors` or `.safetensors.index.json` file,
  and a PEFT directory with `adapter_config.json` plus `adapter_model.safetensors`.
- Scope: saved vanilla LoRA `lora_A.weight`/`lora_B.weight` pairs for floating point, dense 2D
  weights with a uniform configured rank. PEFT's wrapper prefix is removed exactly once.
- Check pairing, positive dimensions, equal inner ranks, agreement with `r`, exact target names
  and base shape `(B.rows, A.columns)`. Do not try suffix matching or rename tensors.
- Read only the index and the headers of shards referenced by the adapter. Report how many
  shards were examined; distinguish missing index mappings from mappings to missing tensors.
- Unsupported layouts, unreadable inputs and bad checkpoint metadata never produce a pass.
  A shape mismatch and a check that could not be completed have different exit statuses.

Non-goals: verify provenance or base revision, predict PEFT/vLLM runtime behavior, validate all
of `LoraConfig`, enforce intended module coverage, inspect numerical values, validate all shards,
merge or repair weights, or download models. A shape match does not imply that an adapter was
trained on that base. Quantized weights, Conv/embedding LoRA, fan-in/fan-out, patterns and other
LoRA variants are outside the supported layout.

## Interface

```text
lora-fitcheck check BASE ADAPTER [--format text|json] [--max-findings N]
lora-fitcheck demo
lora-fitcheck --version
```

Return `0` when the supported layout matches, `1` for a completed check with mismatches, and
`2` for unsupported, incomplete or unusable inputs. A single invalid model directory is an input
error, never a fallback to a different checkpoint. `demo` builds both examples in a temporary
directory and returns zero if it gets the documented two verdicts. No configuration beyond the
original PEFT files. `python -m lora_fitcheck` is equivalent to the entry point.

## Architecture

Use `argparse`, `pathlib`, dataclasses and the JSON decoder in the standard library. A bounded
reader parses strictly typed metadata, never decodes a tensor payload, and checks header offsets
against file length. A separate matcher interprets only the supported PEFT representation and
creates a typed report. The CLI renders text or versioned JSON; expected errors become concise
messages and stable codes. Sort tensor names and shard paths for deterministic diagnostics.
Assume stable local files during a check. For sharded bases, the index may name only `.safetensors`
files in its own directory; standard local filesystem links are permitted.

A built-in demo provides the first useful result immediately after installation. Test on Windows,
Linux and macOS via the configured CI matrix. A synthetic benchmark measures index/header I/O
and wall time; interoperability tests should use actual output from PEFT and Transformers.
