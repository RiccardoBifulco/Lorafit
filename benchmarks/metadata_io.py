"""Measure a header-only check on a synthetic, physically written sharded checkpoint."""

import json
import platform
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory
from time import perf_counter

from lorafit._check import check
from lorafit._demo import write_weights


def main() -> None:
    with TemporaryDirectory(prefix="lorafit-bench-") as tmp:
        base, adapter = Path(tmp) / "base", Path(tmp) / "adapter"
        base.mkdir()
        adapter.mkdir()
        weight_map: dict[str, str] = {}
        for i in range(8):
            name = f"model.layers.{i}.self_attn.q_proj.weight"
            shard = f"model-{i + 1:05d}-of-00008.safetensors"
            weight_map[name] = shard
            write_weights(base / shard, {name: (1024, 2048)})
        (base / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": weight_map}), encoding="utf-8"
        )
        (adapter / "adapter_config.json").write_text(
            json.dumps({"peft_type": "LORA", "r": 2}), encoding="utf-8"
        )
        write_weights(
            adapter / "adapter_model.safetensors",
            {
                f"base_model.model.model.layers.{i}.self_attn.q_proj.lora_{side}.weight": shape
                for i in (0, 7)
                for side, shape in (("A", (2, 2048)), ("B", (1024, 2)))
            },
        )
        total = sum(file.stat().st_size for file in (*base.iterdir(), *adapter.iterdir()))
        first = check(base, adapter)
        assert (first.exit_code, first.matched, first.base_shards_read) == (0, 2, 2)
        times = []
        for _ in range(30):
            start = perf_counter()
            report = check(base, adapter)
            times.append((perf_counter() - start) * 1000)
            if (report.exit_code, report.metadata_bytes) != (0, first.metadata_bytes):
                raise RuntimeError("benchmark check changed between repetitions")
        print(f"{platform.system()} {platform.machine()} / Python {platform.python_version()}")
        print(f"Artifact files: {total:,} bytes")
        print(
            f"Metadata read per check: {first.metadata_bytes:,} bytes "
            f"({first.base_shards_read}/{first.base_shards} base headers)"
        )
        print(f"In-process warm median: {median(times):.3f} ms over {len(times)} checks")


if __name__ == "__main__":
    main()
