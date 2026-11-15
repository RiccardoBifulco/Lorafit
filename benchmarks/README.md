# Metadata I/O benchmark

From the repository root, after installing the package:

```sh
python benchmarks/metadata_io.py
```

The script writes eight synthetic model shards (about 64 MiB total) and an adapter that touches
two of them. It checks the result, reports actual file lengths and metadata bytes returned by
the tool, then measures 30 warm in-process checks. Fixture generation, Python startup and
installation are outside the timing. Wall time depends on disk, OS, interpreter and caches;
the result is not a model-load or network benchmark.

One run on Windows x64, Python 3.12.16: 1,501 metadata bytes read out of 67,160,381 bytes
of local artifacts (2 of 8 base headers), with a warm median of 2.276 ms over 30 checks.

The [reader boundary test](../tests/test_metadata.py) independently fails if the parser
requests any payload bytes. The benchmark's byte counts are for application-level reads,
not physical I/O at the disk controller.
