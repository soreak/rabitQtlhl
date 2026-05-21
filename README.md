# RaBitQ-TLHL ann-benchmarks package

This is a clean package for testing the current RaBitQ-TLHL graph index in
ann-benchmarks. It intentionally excludes local build outputs, datasets, docs,
tests, and HDF5 sample programs.

Included:

- `include/`: required RaBitQ headers only.
- `src/rabitqtlhl_c_api.cpp`: C ABI wrapper around `TreeRaBitQGraphIndex`.
- `CMakeLists.txt`: builds the shared library used by Python.
- `ann_benchmarks/algorithms/rabitqtlhl/module.py`: ann-benchmarks wrapper.
- `ann_benchmarks/algorithms/rabitqtlhl/config.yml`: run groups and query args.
- `ann_benchmarks/algorithms/rabitqtlhl/Dockerfile`: optional Docker integration.

## Local build

From this package directory:

```bat
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

The shared library is copied next to `module.py`.

## Install into ann-benchmarks

Copy the algorithm folder into ann-benchmarks:

```bat
xcopy /E /I /Y ann_benchmarks\algorithms\rabitqtlhl C:\Users\caizehua\Desktop\code\ann-benchmarks\ann_benchmarks\algorithms\rabitqtlhl
```

Then copy or merge the `config.yml` into the same folder. The wrapper imports as:

```python
from ann_benchmarks.algorithms.rabitqtlhl import RabitQTLHL
```

## Query args

The ann-benchmarks `query_args` are:

```text
[efSearch, neighborCap, rerank]
```

Current best observed SIFT1M @10 candidate:

```text
[56, 24, 64]
```

## Notes

- The wrapper returns original row ids even though the index reorders points
  internally for cache locality.
- Batch query uses the C API loop so Python ThreadPool does not race on the
  index visit markers.
