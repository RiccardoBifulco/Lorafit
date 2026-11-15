# Good first issues

These are drafts for the project owner to post when ready.

### Add a BF16 export fixture made by the official SafeTensors writer

The tests currently synthesize BF16 metadata. Add a small repeatable example using the official
writer in a separate development environment. Document the command and prove that Fitcheck
accepts mixed base/adapter storage dtypes within the supported float set. Keep the wheel free
of dependencies and avoid checking large weights into Git.

### Point people to the right file when a directory contains only a variant

`check` requires an explicit path for alternative filenames. On `base_not_found`, show up to
three suitable filenames in the directory, if any. Do not auto-select one. Add a test for a
directory that contains `model.fp16.safetensors` but no default export.

### Add a JSON report example for a missing or misindexed shard

Using `examples/make_demo.py`, document a reproducible edit to a temporary fixture and show how
a pipeline can distinguish exit `2` from a shape mismatch. Reference `docs/report.md` instead
of re-documenting the whole schema.
