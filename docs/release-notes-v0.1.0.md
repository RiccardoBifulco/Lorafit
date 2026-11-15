# LoRA Fitcheck v0.1.0

Use two local paths to verify that a saved PEFT LoRA's names, dimensions and configured
rank fit the intended SafeTensors checkpoint, before starting a model loader.

```sh
python -m pip install lora_fitcheck-0.1.0-py3-none-any.whl
lora-fitcheck demo
```

- Single-file and sharded `transformers` base exports are supported.
- Findings distinguish missing targets, broken shard references, missing partners, rank
  mismatches, shape mismatches, and unsupported input.
- JSON output and exit codes support CI; the wheel requires Python 3.10+ and no runtime packages.
- The initial scope is standard, dense 2D PEFT LoRA with one configured rank. This is a shape
  and naming preflight; use PEFT and an inference smoke test to check actual behavior.

Attach the verified wheel and source tarball from `dist/` to the GitHub release for tag `v0.1.0`.
Build and check them with `python scripts/verify_dist.py --output-dir dist`.
