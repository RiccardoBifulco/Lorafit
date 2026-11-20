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

## Reproduce missing and misindexed shard reports

These walkthroughs show how an incomplete base (exit `2`) differs from a completed check that
found a shape mismatch (exit `1`). They use only the small generated files from
`examples/make_demo.py`, so they need no GPU, network access, or model weights. Run them from the
repository root after `python -m pip install .`. Each edit starts from fresh fixtures, so use
a new directory name every time. Paths in the reports below are relative to where you run the
command. Full field definitions are in [docs/report.md](../docs/report.md).

### Reference: the unedited `wrong-base`

```sh
python examples/make_demo.py demo-wrong
lorafit check demo-wrong/wrong-base demo-wrong/adapter --format json
```

The check finishes and exits `1`. Its report has `"status": "mismatch"` and one finding with
`"code": "shape_mismatch"` for `v_proj` (`expected_shape` `[4, 16]`, `actual_shape` `[8, 16]`).
Every shard was readable (`base_shards_read` is `2`), so the base is complete but the wrong width.

### Edit 1: remove a base shard (`file_missing`)

The index maps `q_proj` to the first shard, so deleting that file makes the base incomplete.

```sh
python examples/make_demo.py demo-missing
rm demo-missing/base/model-00001-of-00002.safetensors
lorafit check demo-missing/base demo-missing/adapter --format json
```

In PowerShell, use `Remove-Item demo-missing/base/model-00001-of-00002.safetensors` for the
second command. The command exits `2`, and the report is:

```json
{
  "schema_version": 1,
  "status": "incomplete",
  "summary": {
    "modules": 2,
    "matched": 1,
    "base_shards": 2,
    "base_shards_read": 1,
    "metadata_bytes": 857
  },
  "declared_base": null,
  "findings": [
    {
      "code": "file_missing",
      "detail": "file is missing",
      "module": null,
      "path": "demo-missing/base/model-00001-of-00002.safetensors",
      "expected_shape": null,
      "actual_shape": null
    }
  ]
}
```

Look at `findings[0].path` for the file to restore. `base_shards` is `2` but `base_shards_read` is
`1`, and `matched` covers only the `v_proj` pair that could be checked.

### Edit 2: point an index entry at the wrong shard (`indexed_tensor_missing`)

Here every file exists, but the index sends `q_proj` to the second shard, which only holds
`v_proj`. This Python snippet makes the edit and works the same in PowerShell and bash:

```sh
python examples/make_demo.py demo-misindexed
python - <<'PY'
import json
from pathlib import Path

index = Path("demo-misindexed/base/model.safetensors.index.json")
data = json.loads(index.read_text(encoding="utf-8"))
data["weight_map"]["model.layers.0.self_attn.q_proj.weight"] = "model-00002-of-00002.safetensors"
index.write_text(json.dumps(data), encoding="utf-8")
PY
lorafit check demo-misindexed/base demo-misindexed/adapter --format json
```

PowerShell has no here-document. Put the Python lines from `import json` through
`index.write_text(...)` in a file such as `edit_index.py`, and run `python edit_index.py` instead.
The check exits `2` with this report:

```json
{
  "schema_version": 1,
  "status": "incomplete",
  "summary": {
    "modules": 2,
    "matched": 1,
    "base_shards": 1,
    "base_shards_read": 1,
    "metadata_bytes": 857
  },
  "declared_base": null,
  "findings": [
    {
      "code": "indexed_tensor_missing",
      "detail": "'model.layers.0.self_attn.q_proj.weight' is absent from the indexed shard header",
      "module": "model.layers.0.self_attn.q_proj",
      "path": "demo-misindexed/base/model-00002-of-00002.safetensors",
      "expected_shape": null,
      "actual_shape": null
    }
  ]
}
```

`module` names the target and `path` names the shard the index points to. The shard exists and
parses, so the fix is to correct or regenerate the index (or the shard), not to restore a file.

### Capture the exit code

```sh
lorafit check demo-missing/base demo-missing/adapter --format json > report.json
echo "exit=$?"
```

```powershell
lorafit check demo-missing/base demo-missing/adapter --format json > report.json
"exit=$LASTEXITCODE"
```

Run the capture immediately after `lorafit`, because the next command overwrites the status.

### Branch on the exit code or JSON

Exit `0` means a match, `1` a completed check with a mismatch, and `2` an incomplete check. The
following Python works on any report from this page and prefers `status` and `code` over the
`detail` text. Save it as `triage.py` outside the fixture directories:

```python
import json
import subprocess
import sys

result = subprocess.run(
    ["lorafit", "check", sys.argv[1], sys.argv[2], "--format", "json"],
    capture_output=True,
    text=True,
)
report = json.loads(result.stdout)
codes = {finding["code"] for finding in report["findings"]}

if result.returncode == 0:
    print("match")
elif result.returncode == 1:
    print("mismatch: choose another base or re-export the adapter", sorted(codes))
elif codes & {"file_missing", "indexed_tensor_missing"}:
    print("incomplete base: fix the files or index", sorted(codes))
else:
    print("incomplete: see docs/report.md for", sorted(codes))
sys.exit(result.returncode)
```

```sh
python triage.py demo-missing/base demo-missing/adapter
python triage.py demo-wrong/wrong-base demo-wrong/adapter
```

These print `incomplete base: fix the files or index ['file_missing']` and
`mismatch: choose another base or re-export the adapter ['shape_mismatch']`.
