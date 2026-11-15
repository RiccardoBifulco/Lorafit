import json
from collections.abc import Iterator
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path

import pytest

from lora_fitcheck import _metadata
from lora_fitcheck._metadata import InputError, read_header, read_index


@pytest.mark.parametrize(
    "raw",
    [
        b"short",
        b"version https://git-lfs.github.com/spec/v1\n",
        b"\xff" * 8,
        b"\x02\x00\x00\x00\x00\x00\x00\x00{}X",
    ],
)
def test_malformed_headers(tmp_path: Path, raw: bytes) -> None:
    path = tmp_path / "broken.safetensors"
    path.write_bytes(raw)
    with pytest.raises(InputError) as exc:
        read_header(path)
    assert exc.value.code == "invalid_header"


@pytest.mark.parametrize(
    "raw",
    [
        b'{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,5]}}',
        b'{"w":{"dtype":"F32","shape":[true],"data_offsets":[0,4]}}',
        b'{"w":{"dtype":"F32","shape":[1],"data_offsets":[1,4]}}',
        b'{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"w":{}}',
        b'{"__metadata__":{"format":false}}',
    ],
)
def test_malformed_header_entries(tmp_path: Path, raw: bytes) -> None:
    path = tmp_path / "broken.safetensors"
    path.write_bytes(len(raw).to_bytes(8, "little") + raw + b"\0" * 4)
    with pytest.raises(InputError) as exc:
        read_header(path)
    assert exc.value.code == "invalid_header"


def test_reader_stops_at_payload_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    header = b'{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}'
    metadata = len(header).to_bytes(8, "little") + header

    class MetadataOnly(BytesIO):
        def read(self, size: int | None = -1) -> bytes:
            assert size is not None and 0 <= size <= len(metadata) - self.tell(), (
                "read would touch the payload"
            )
            return super().read(size)

    @contextmanager
    def guarded(_: Path) -> Iterator[tuple[BytesIO, int]]:
        yield MetadataOnly(metadata), len(metadata) + 4

    monkeypatch.setattr(_metadata, "_regular_file", guarded)
    result = read_header(Path("unused"))
    assert result.bytes_read == len(metadata)
    assert result.tensors["w"].shape == (1,)


@pytest.mark.parametrize(
    "shard", ["../weights.safetensors", "C:weights.safetensors", "CON.safetensors"]
)
def test_index_only_accepts_portable_basenames(tmp_path: Path, shard: str) -> None:
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"w": shard}}), encoding="utf-8")
    with pytest.raises(InputError) as exc:
        read_index(index)
    assert exc.value.code == "invalid_index"
