# Examples

Create reusable synthetic checkpoint files and check the good and bad base:

```sh
python examples/make_demo.py demo-data
lorafit check demo-data/base demo-data/adapter --format json
lorafit check demo-data/wrong-base demo-data/adapter --format json
```

The second check exits `1` and names the mismatched `v_proj` weight. Use a new output
directory if these files already exist.

For your own artifacts, run this before kicking off a GPU job. This GitHub Actions step
expects the two artifacts to have already been staged in the checkout:

```yaml
- name: Check adapter against its base
  run: lorafit check checkpoints/base checkpoints/adapter
```

## Reproduce with real exporters

The following optional example instantiates a tiny randomly initialized Llama model on CPU,
saves it with Transformers, and saves a LoRA adapter with PEFT. It then checks the sharded
base and a different-width single-file base, expecting exits `0` and `1`. No pretrained
weights are downloaded or used. The example's dependencies are separate from the tool.

```sh
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install 'transformers==4.53.3' 'peft==0.17.1'
python examples/peft_roundtrip.py
```

Use Python 3.12 for the pinned example. On macOS, install the standard `torch==2.7.1` wheel
from PyPI instead of using the CPU wheel index.
