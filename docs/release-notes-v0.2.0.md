# LoRAFit v0.2.0

LoRAFit is the new name for LoRA Fitcheck. The shorter `lorafit` command checks whether a saved PEFT LoRA's tensor names, shapes and configured rank match a local SafeTensors base.

```sh
python -m pip install --no-index lorafit-0.2.0-py3-none-any.whl
lorafit demo
```

Use Python 3.10 or newer. If you already installed `lora-fitcheck` v0.1.0 in this environment, uninstall that package before installing `lorafit` to replace its scripts cleanly. The new wheel includes the original `lora-fitcheck` and `python -m lora_fitcheck` entry points as aliases.

The README now leads with a match and mismatch from the bundled demo, followed by a two-command quickstart and CI examples. The check still opens only the required headers and reports structural fit for the supported PEFT layout.

Build the wheel and source archive with `python scripts/verify_dist.py --output-dir dist` from an environment with `.[dev]` installed. Attach both verified artifacts to the GitHub release for tag `v0.2.0`.
