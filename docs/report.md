# JSON report, schema version 1

Run `lorafit check BASE ADAPTER --format json`. Paths remain local; they are reported
when they help locate an error. File-based findings go to stdout in both modes. CLI argument
errors use stderr and do not emit a JSON report.

| Field | Type and meaning |
| --- | --- |
| `schema_version` | Integer `1`. |
| `status` | `match`, `mismatch`, or `incomplete`, corresponding to process exit codes `0`, `1`, `2`. Incomplete takes precedence if both kinds of finding exist. |
| `summary.modules` | Number of distinct recognized adapter module names, including those with a missing A/B partner. |
| `summary.matched` | Number of stored A/B pairs that passed rank and exact base shape/name checks. |
| `summary.base_shards` | Number of shard names in a successfully parsed index, or one for a selected single-file base. Zero if discovery did not reach that stage. |
| `summary.base_shards_read` | Number of base headers successfully parsed. A shard with no requested tensors need not be opened. |
| `summary.metadata_bytes` | Lengths of successfully parsed metadata (adapter config, adapter and base headers including prefix, and index if present). Partial reads from files that failed to parse are excluded. |
| `declared_base` | String from `adapter_config.json`, or `null`. It is informational and unverified. |
| `findings` | Array of objects with `code`, `detail`, optional `module` and `path` (nullable strings), and `expected_shape` / `actual_shape` (nullable integer arrays). For a shape mismatch, the expected shape comes from the base; the actual shape is the adapter's implied update. |

Branch on `status`, process status, or the `code` field rather than parsing `detail` text.
New finding codes may be added. Main codes:

| Code | Action |
| --- | --- |
| `target_missing` | Check that the adapter was exported for the model tree and base files you chose. |
| `shape_mismatch` | Switch to the intended base width/variant or re-export the adapter. |
| `rank_mismatch` / `missing_pair` | Verify the adapter config and that the adapter export is complete. |
| `indexed_tensor_missing` | Re-export or replace the shard/index; the named tensor is absent from the specified shard. |
| `file_missing`, `file_unreadable`, `not_a_file` | Confirm the reported path points to an accessible regular file. |
| `invalid_config`, `invalid_index`, `invalid_header`, `invalid_matrix` | Confirm complete files and regenerate malformed metadata. |
| `ambiguous_base`, `base_not_found`, `unsupported_base` | Pass an explicit existing `.safetensors` file or `.safetensors.index.json`. |
| `empty_adapter` | Confirm the adapter actually contains saved LoRA weights. |
| `unsupported_config`, `unsupported_tensor`, `unsupported_layout` | Use the actual PEFT/model runtime to assess layouts outside v0.1. |

For runnable `file_missing` and `indexed_tensor_missing` reproductions and a script that
branches on exit codes, see the
[missing and misindexed shard walkthrough](../examples/README.md#reproduce-missing-and-misindexed-shard-reports).
