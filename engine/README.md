# vdb_engine

A small, self-contained C++17 vector search engine: distance metrics,
brute-force top-K search, and an IVF (inverted file) approximate index. See
[include/vdb](include/vdb) for the public headers - each has a comment
explaining what it does and why.

## Building

```bash
cmake -S engine -B engine/build -DCMAKE_BUILD_TYPE=Release
cmake --build engine/build -j
```

This produces a static library (`vdb_engine`) and, by default, a test
binary. Run the tests with:

```bash
ctest --test-dir engine/build --output-on-failure
```

## Python bindings

The backend calls into this engine through a pybind11 module
(`vdb_native`), built separately since most environments don't need it -
`app.engine_bridge` falls back to a NumPy implementation of brute-force
search when the module isn't importable, and only approximate (IVF) search
strictly requires it.

To build it:

```bash
cmake -S engine -B engine/build -DCMAKE_BUILD_TYPE=Release -DVDB_BUILD_PYTHON_BINDINGS=ON
cmake --build engine/build -j
```

The resulting module lands at `engine/build/bindings/vdb_native*.so` (or
`.pyd` on Windows). Put its containing directory on `PYTHONPATH` before
starting the backend, e.g.:

```bash
export PYTHONPATH=$PWD/engine/build/bindings
```

The root `Dockerfile` does this automatically as part of the backend image
build.

## Layout

```
engine/
├── include/vdb/     public headers (distance, vector_store, search, ivf_index, topk_heap)
├── cpp/             implementation files
├── bindings/        pybind11 module
└── tests/           doctest-based unit tests
```
