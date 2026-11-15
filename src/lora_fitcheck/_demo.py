"""Small synthetic fixtures for a completely offline demo."""

import json
from math import prod
from pathlib import Path

from lora_fitcheck._metadata import FLOAT_BYTES

Q = "model.layers.0.self_attn.q_proj"
V = "model.layers.0.self_attn.v_proj"


def write_weights(path: Path, shapes: dict[str, tuple[int, ...]], dtype: str = "F32") -> None:
    metadata: dict[str, object] = {"__metadata__": {"format": "pt"}}
    end = 0
    for name, shape in shapes.items():
        start, end = end, end + prod(shape) * FLOAT_BYTES[dtype]
        metadata[name] = {"dtype": dtype, "shape": shape, "data_offsets": [start, end]}
    header = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
    header += b" " * (-len(header) % 8)
    with path.open("xb") as handle:
        handle.write(len(header).to_bytes(8, "little"))
        handle.write(header)
        handle.write(bytes(end))


def build_demo(root: Path) -> tuple[Path, Path, Path]:
    base, wrong, adapter = (root / name for name in ("base", "wrong-base", "adapter"))
    for directory in (base, wrong, adapter):
        directory.mkdir(parents=True, exist_ok=False)
    for directory, v_width in ((base, 8), (wrong, 4)):
        shards = {
            Q + ".weight": "model-00001-of-00002.safetensors",
            V + ".weight": "model-00002-of-00002.safetensors",
        }
        (directory / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": shards}), encoding="utf-8"
        )
        for (key, filename), width in zip(shards.items(), (8, v_width), strict=True):
            write_weights(directory / filename, {key: (width, 16)})
    (adapter / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "r": 2}), encoding="utf-8"
    )
    write_weights(
        adapter / "adapter_model.safetensors",
        {
            f"base_model.model.{module}.lora_{side}.weight": shape
            for module in (Q, V)
            for side, shape in (("A", (2, 16)), ("B", (8, 2)))
        },
    )
    return base, wrong, adapter
