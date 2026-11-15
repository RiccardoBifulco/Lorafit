"""Read bounded metadata from local SafeTensors artifacts."""

from __future__ import annotations

import json
import os
import re
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, NoReturn, cast

MAX_HEADER = 100_000_000  # SafeTensors format limit, in bytes.
MAX_INDEX = 64 * 1024 * 1024
MAX_CONFIG = 1024 * 1024
MAX_DIMENSION = 2**64 - 1
FLOAT_BYTES = {"F16": 2, "BF16": 2, "F32": 4, "F64": 8}
SHARD_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\.safetensors\Z")
RESERVED = {"AUX", "CON", "NUL", "PRN"} | {
    f"{prefix}{number}" for prefix in ("COM", "LPT") for number in range(10)
}


class InputError(ValueError):
    def __init__(self, code: str, path: Path, detail: str) -> None:
        super().__init__(detail)
        self.code = code
        self.path = path


@dataclass(frozen=True)
class Tensor:
    dtype: str
    shape: tuple[int, ...]
    start: int
    end: int


@dataclass(frozen=True)
class Header:
    tensors: dict[str, Tensor]
    bytes_read: int


def _pairs(items: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for k, v in items:
        if k in result:
            raise ValueError(f"duplicate JSON key {k!r}")
        result[k] = v
    return result


def _reject_constant(value: str) -> NoReturn:
    raise ValueError(f"invalid JSON constant {value}")


def _object(raw: bytes, path: Path, code: str) -> dict[str, object]:
    try:
        value: object = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_pairs, parse_constant=_reject_constant
        )
    except (ValueError, RecursionError) as exc:
        raise InputError(code, path, f"invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise InputError(code, path, "expected a JSON object")
    return cast(dict[str, object], value)


@contextmanager
def _regular_file(path: Path) -> Iterator[tuple[BinaryIO, int]]:
    try:
        if not stat.S_ISREG(path.stat().st_mode):
            raise InputError("not_a_file", path, "expected a regular file")
        with path.open("rb", buffering=0) as handle:
            size = os.fstat(handle.fileno()).st_size
            yield handle, size
    except FileNotFoundError as exc:
        raise InputError("file_missing", path, "file is missing") from exc
    except (OSError, ValueError) as exc:
        if isinstance(exc, InputError):
            raise
        detail = exc.strerror if isinstance(exc, OSError) else str(exc)
        raise InputError("file_unreadable", path, detail or "could not read file") from exc


def read_json(path: Path, limit: int, code: str) -> tuple[dict[str, object], int]:
    with _regular_file(path) as (handle, size):
        if size > limit:
            raise InputError(code, path, f"file exceeds the {limit:,}-byte metadata limit")
        raw = handle.read(size)
        if len(raw) != size or handle.read(1):
            raise InputError(code, path, "file size changed during the check")
    return _object(raw, path, code), size


def _uint(value: object) -> bool:
    return type(value) is int and 0 <= value <= MAX_DIMENSION


def _tensor(name: str, raw: object, path: Path, payload_size: int) -> Tensor:
    if not name or not isinstance(raw, dict):
        raise InputError("invalid_header", path, f"invalid tensor entry {name!r}")
    dtype, shape, offsets = raw.get("dtype"), raw.get("shape"), raw.get("data_offsets")
    if not isinstance(dtype, str) or not dtype:
        raise InputError("invalid_header", path, f"tensor {name!r} has no valid dtype")
    if not isinstance(shape, list) or not all(_uint(x) for x in shape):
        raise InputError("invalid_header", path, f"tensor {name!r} has an invalid shape")
    if not isinstance(offsets, list) or len(offsets) != 2 or not all(_uint(x) for x in offsets):
        raise InputError("invalid_header", path, f"tensor {name!r} has invalid data offsets")
    start, end = offsets
    if not start <= end <= payload_size:
        raise InputError(
            "invalid_header", path, f"tensor {name!r} extends past the file or has reversed offsets"
        )
    return Tensor(dtype, tuple(shape), start, end)


def read_header(path: Path) -> Header:
    with _regular_file(path) as (handle, size):
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise InputError("invalid_header", path, "file is too short for a SafeTensors header")
        length = int.from_bytes(prefix, "little")
        if not 2 <= length <= MAX_HEADER or length > size - 8:
            raise InputError(
                "invalid_header",
                path,
                "invalid SafeTensors header length (possible truncated file or Git LFS pointer)",
            )
        raw = handle.read(length)
        if len(raw) != length or not raw.startswith(b"{"):
            raise InputError("invalid_header", path, "incomplete header or missing opening brace")
    value = _object(raw, path, "invalid_header")
    if "__metadata__" in value:
        info = value.pop("__metadata__")
        if not isinstance(info, dict) or not all(isinstance(v, str) for v in info.values()):
            raise InputError("invalid_header", path, "__metadata__ must map strings to strings")
    payload_size = size - length - 8
    tensors = {name: _tensor(name, item, path, payload_size) for name, item in value.items()}
    cursor = 0
    for name, tensor in sorted(tensors.items(), key=lambda item: (item[1].start, item[1].end)):
        if tensor.start != cursor:
            raise InputError("invalid_header", path, f"gap or overlap at tensor {name!r}")
        cursor = tensor.end
    if cursor != payload_size:
        raise InputError("invalid_header", path, "header offsets do not cover the file payload")
    return Header(tensors, length + 8)


def read_index(path: Path) -> tuple[dict[str, str], int]:
    value, size = read_json(path, MAX_INDEX, "invalid_index")
    weights = value.get("weight_map")
    if not isinstance(weights, dict) or not weights:
        raise InputError("invalid_index", path, "expected a nonempty weight_map object")
    result: dict[str, str] = {}
    for tensor, shard in weights.items():
        if not tensor or not isinstance(shard, str):
            raise InputError(
                "invalid_index", path, "weight_map must map tensor names to shard names"
            )
        if not SHARD_NAME.fullmatch(shard) or shard.split(".", 1)[0].upper() in RESERVED:
            raise InputError(
                "invalid_index", path, f"expected a local shard filename, got {shard!r}"
            )
        result[tensor] = shard
    return result, size
