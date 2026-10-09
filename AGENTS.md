# AGENTS.md

Working notes for the SuperKMeans repo.

## What it is

Fast k-means for high-dim vector embeddings, on `float32` **or quantized** vectors (`sq8`,
`lvq4`, `rabitq`) — you pass `float32`, it quantizes internally. Faster than FAISS /
scikit-learn at equal quality (~1M×1536 in seconds). Header-only **C++17** + **Python
bindings**; CPUs (ARM + x86) and GPUs. Core: `SuperKMeans::Train()` in
[`include/superkmeans/superkmeans.h`](include/superkmeans/superkmeans.h).
Paper: https://arxiv.org/pdf/2603.20009.

## Core idea: progressive pruning + PDX

Prunes centroids during the **assignment** step by interleaving a GEMM on the front `d'` dims
with **progressive**-pruning kernels every 64 dims in the trailing ones. Unlike Elkan's (a 1/0
decision), pruning is progressive — it keeps trying to prune while walking the trailing dims.
Vectors use a **hybrid PDX layout** (block-column-major, split every 64 dims) to make this
efficient. Papers: **ADSampling** https://dl.acm.org/doi/pdf/10.1145/3589282 (almost-lossless,
< 0.005 recall loss); **PDX** https://dl.acm.org/doi/pdf/10.1145/3725333.

## Where things live

| Path | What |
| --- | --- |
| `include/superkmeans/superkmeans.h` | Core — `Train()`, assign family, pruning loop |
| `include/superkmeans/hierarchical_superkmeans.h` | Hierarchical variant (use for n > 100K) |
| `include/superkmeans/common.h` | Shared `constexpr` constants + macros/pragmas, `Sgemm` (GEMM backend) |
| `include/superkmeans/executor.h` | `ParallelExecutor` + `ExecutorHolder`/`ParallelSection`/`ExecutorScope`; backends in `executors/` |
| `include/superkmeans/profiler.h` | `SKM_PROFILE_SCOPE` timing |
| `include/superkmeans/distance_computers/` | **All** SIMD distance kernels |
| `include/superkmeans/quantizers/` | `f32`/`sq8`/`lvq4`/`rabitq` + `quantizer.h`, `sq_common.h` |
| `include/superkmeans/pdx/` | PDX layout + `utils.h` |
| `benchmarks/` | Benchmarks; base example `ad_hoc_superkmeans.cpp` |
| `examples/` | `simple_clustering.{cpp,py}`, `hdf5_clustering.py` |
| `python/` | Bindings (`bindings/bindings.cpp`, `superkmeans/__init__.py`); `python/README.md` |
| `tests/` | GoogleTest C++; `python/tests/` for bindings |

Docs: `README.md`, `INSTALL.md`, `BENCHMARKING.md`, `CONTRIBUTING.md`, `python/README.md`.

## Verification gate (definition of done)

**A change isn't done until all pass. Run in the FOREGROUND — never background these.**

1. **Hierarchical parity (first, before building)** — if the change touched `superkmeans.h`,
   evaluate whether it must also be applied to `hierarchical_superkmeans.h`.
   `HierarchicalSuperKMeans` inherits `SuperKMeans` (ctor + shared members propagate via
   delegation) but **overrides `Train` and the clustering pipeline** — so edits to the
   training/pipeline path do NOT automatically carry over. Apply/verify there too.
2. **Format** — `./scripts/format.sh`, then `./scripts/format_check.sh` clean.
3. **Build** — `cmake . -DSKMEANS_COMPILE_TESTS=ON && make -j$(nproc) tests`, no errors.
4. **C++ tests** — `ctest --output-on-failure` all pass (a few parametrized cases skip by design).
   If the change touches `executor.h`/`executors/` or `Sgemm`, repeat 3–4 in separate build dirs
   with `-DSKMEANS_EXECUTOR=openmp` and with `-DSKMEANS_GEMM=blas`.
5. **Lint** — `./scripts/tidy_check.sh`: no `.clang-tidy` warnings from `include/superkmeans/`.
6. **Python** — `venv/bin/pip install .` (builds the bindings), then `venv/bin/pytest python/tests/`.
7. **Examples** — `make examples`, then run all of them to completion: the C++ ones
   (`./examples/{simple,hierarchical,quantized}_clustering.out <n> <d> <k>`, e.g. `100000 128 1000`)
   and the Python ones (`venv/bin/python3 examples/{simple,quantized}_clustering.py`;
   `hdf5_clustering.py` too if an `.hdf5` dataset is at hand — they live outside the repo).
8. **AGENTS.md accurate** — if the change invalidated anything here (paths, commands, contracts,
   thresholds, gotchas, SIMD org, style), update this file in the same change.

New feature ⇒ ship a unit test with it (C++ in `tests/`, Python in `python/tests/` if exposed).

## Build & run (beyond the gate)

Header-only; consumers link the `superkmeans` target.
```bash
cmake . && make examples          # examples on by default; ./examples/simple_clustering.out <n> <d> <k>
cmake . -DSKMEANS_COMPILE_BENCHMARKS=ON -DFAISS_OPT_LEVEL="avx512" && make benchmarks
./benchmarks/ad_hoc_superkmeans.out <dataset_id>   # base example + profiling logs
```
Knobs (see INSTALL.md):
- `-DSKMEANS_EXECUTOR` — default executor: `forkunion` (default; `serial` under Emscripten), `openmp`, `serial`.
- `-DSKMEANS_GEMM` — `auto` (Accelerate on Apple, Eigen elsewhere), `eigen`, `accelerate`, `blas`. With
  `blas`, workers call `sgemm_` concurrently, so the BLAS must be sequential and thread-safe (MKL
  `sequential`; OpenBLAS `USE_THREAD=0 USE_LOCKING=1`, or `OPENBLAS_NUM_THREADS=1`).
- `-DSKMEANS_MARCH` (default `native`), `-DSKMEANS_PORTABLE` (wheels: `-mavx2 -mfma` on x86_64, plain
  `-O3` elsewhere), `-DSKMEANS_SKIP_FFTW`. The default build needs no OpenMP and no BLAS.

## Code style

- **Naming**: `PascalCase` functions/classes/structs; `snake_case` variables/members;
  `UPPER_SNAKE_CASE` constants; namespaces lowercase.
- Follow `.clang-format` / `.clang-tidy`.
- **Memory**: RAII, no raw `new`/`delete`. Buffers that don't need zero-init → `new T[]` in a
  `unique_ptr`, **not** `std::vector`/`resize()`.
- Constants/magic numbers → `include/superkmeans/common.h` as `constexpr`.
- Keep comments simple. TODOs: `TODO(@<github_user>, <priority>): <summary>`.
- **Reuse before writing**: check `common.h`, `quantizers/sq_common.h`, `pdx/utils.h` first.

## Performance

Performance-critical — weigh every copy/allocation.
- **SIMD is centralized.** Distance kernels exist for **NEON / AVX2 / AVX512 / scalar**,
  dispatched at compile time and tagged by `Quantization` scheme (`f32`/`sq8`/`sq4`/`rabitq`).
  **All** SIMD lives in `distance_computers/` — don't scatter it elsewhere; keep all backends in
  sync when changing a kernel. `sq4` tags the legacy 4-bit nibble kernels + PDX machinery of a
  former quantizer — kept because they're non-trivial to reimplement and **lvq4 uses them**, but
  `SuperKMeans<Quantization::sq4>` itself `static_assert`s (not implemented).
- **Parallelism goes through the executor.** No OpenMP pragmas or `omp_*` in library code (only
  `executors/openmp.h` includes `<omp.h>`). A parallel loop is
  `GetExecutor().ParallelFor(n, [&](size_t begin, size_t end, size_t worker) { ... })`: static
  ranges, dense `worker < NumWorkers()`. Per-thread scratch is indexed by `worker`; reductions use
  per-worker partials combined after the loop; never call `ParallelFor` inside a `ParallelFor` body.
- **GEMMs are single-threaded.** f32 assignment is one `ParallelFor` per pass over
  `MINI_BATCH_SIZE`-row (256) blocks of X; each worker runs `Sgemm` per Y tile, then argmin/pruning
  on the spot. Scratch: `BatchComputer::ScratchSize(executor)`. `EIGEN_DONT_PARALLELIZE` is set on
  the target. The pruner's rotations (GEMM and DCT) run `ROTATION_BLOCK_SIZE`-row (2048) blocks;
  the in-place `Rotate<true>` copies `ROTATION_BLOCK_SIZE × NumWorkers()` rows at a time through
  scratch, so every worker gets a block.
- **`SKM_VECTORIZE_LOOP`** (`common.h`) forces loop autovectorization (esp. FP reductions).
  Other macros there: `SKM_RESTRICT`, `SKM_ALWAYS_INLINE`, `SKM_NO_INLINE`,
  `SKM_LIKELY`/`SKM_UNLIKELY`, `SKM_PREFETCH`.
- **Profiling**: `SKM_PROFILE_SCOPE("name")` (`profiler.h`); `ad_hoc_superkmeans.out` prints logs.

## Recipes

- **Add/change a distance kernel** — edit **all** backends in `distance_computers/`
  (`neon_computers.h`, `avx2_computers.h`, `avx512_computers.h`, `scalar_computers.h`) in
  lockstep. On ARM the x86 ones aren't compiled or linted, so eyeball them / rely on x86 CI.
  Cover it in `tests/test_distance_computers.cpp`.
- **Add a quantizer** — implement a class deriving `IQuantizer<q>`
  (`quantizers/quantizer.h`: `Fit`/`Encode`/`Decode`/`ComputeNorms`/`FindNearestNeighbor`/…); in
  `common.h` add a `Quantization` enum value, a `QuantizationName()` case, and a
  `QuantizerClass<q>` specialization pointing at the class (`CreateQuantizer()`/`GetQuantizer()`
  resolve through that trait — nothing to register in `superkmeans.h`); add a
  `using SuperKMeans<Name> = SuperKMeans<Quantization::…>` alias in `superkmeans.h` and its
  `HierarchicalSuperKMeans<Name>` twin in `hierarchical_superkmeans.h`; in the bindings
  instantiate `BindSuperKMeans<q>` + the hierarchical bind (`bindings.cpp`, plus a
  `QuantizerParams` overload if it exposes params) and add the string to `_QUANTIZER_MAP` in
  `__init__.py`; add tests (see Testing).
- **Add a benchmark** — register the target with `skmeans_add_benchmark(<name>.out <source>)` in
  `benchmarks/CMakeLists.txt` (`skmeans_add_faiss_benchmark` for FAISS-linked ones).

## Testing

- C++ in `tests/` (GoogleTest), Python in `python/tests/`. Compile with
  `-DSKMEANS_COMPILE_TESTS=ON`, run via `ctest`. Executor backends: `tests/test_executor.cpp`.
- CI (`.github/workflows/ci.yml`) also runs the tests with the `openmp`/`serial` executors and the
  `blas` GEMM, on Windows MSVC, on macOS (Accelerate) and as Wasm under Node.
- Quantized tests are **typed tests** (`TYPED_TEST_SUITE` + `TYPED_TEST`) over a
  `skmeans::QuantizationTag<...>` type list; the scheme is `TypeParam::value` and
  `SuperKMeans<TypeParam::value>` is constructed directly. Shared integration/pruning tests live
  **once** in `tests/test_quantized.cpp` (auto-covers sq8/lvq4/rabitq); quantizer-*specific* unit
  tests go in `test_quantized_{sq8,lvq4,rabitq}.cpp`. Hierarchical quantized tests are typed the
  same way in `test_hierarchical_superkmeans.cpp`. For runtime loops over schemes (benchmarks,
  non-fixture tests) use a generic lambda over `skmeans::QuantizationTag` values.
- **Unless the user says otherwise, a new test must cover all quantizers *and* hierarchical with
  all quantizers** — add it as a `TYPED_TEST` (flat) plus its hierarchical counterpart, not a
  single-quantizer case.
- Recall-checking tests need `#undef HAS_FFTW` at the top (see Gotchas).

## Gotchas

### Recall ground truth & `#undef HAS_FFTW`

SuperKMeans builds **IVF indexes for search**, so **`recall`** is the metric we defend (over
WCSS). Tests check `RECALL_GROUND_TRUTH` (`tests/recall_utils.h`) per pipeline (`f32`/`sq8`/
`lvq4`/`rabitq`, flat + hierarchical) via `EXPECT_NEAR(recall, expected, RECALL_TOL)`.
- **Generated by** `generate_recall_ground_truth.cpp` (`.out`): trains on the fixed
  `tests/test_data.bin` with the fixed config in `recall_utils.h`, prints values to paste back.
  `generate_wcss_ground_truth.cpp` is the WCSS twin.
- **`#undef HAS_FFTW`**: the ground truth used the **non-FFTW rotation path**. With FFTW
  installed CMake defines `HAS_FFTW` → FFT-based rotation → slightly different rotation → recall
  drifts past `RECALL_TOL`. So **every ground-truth-checking file starts with `#undef HAS_FFTW`**
  (`test_quantized*.cpp`, `test_hierarchical_superkmeans.cpp`, `test_wcss.cpp`, both generators)
  — keep that line.
- **Regenerating**: almost never — it can mask real regressions. **Only with explicit user
  confirmation**, when the ground truth is *meant* to change; run the generator, paste its block
  into `recall_utils.h`, note why.

### Training / assign paths (rotation, assign family, stale caches)

**Trained state is rotated.** `Train()` rotates all data up front (`SampleAndRotateVectors` →
ADSampling `pruner->Rotate`, seeded at construction), then `Fit()`/`Encode()` run on it — so the
quantizer params and trained state (`quantized_data`, `quantized_centroids`,
`horizontal_centroids`) all live in the rotated domain. Rotation is required by **(a)** RaBitQ,
and **(b) ADSampling pruning for every quantizer** (pruning bounds derive from the random
rotation → valid only on rotated data). sq8/lvq4 may skip rotation **only on a non-pruning
path**. Encoding in the wrong domain, or pruning on un-rotated data ⇒ **silently bad params /
invalid bounds** (no crash, just recall loss).

**`TrainInPlace(float* data, …)`** — memory-halving twin of `Train`, in both `SuperKMeans` and
`HierarchicalSuperKMeans`. It rotates the **caller's** buffer via `ConfigInPlaceTraining` (forces
`sampling_fraction = 1.0` with a warning, then `pruner->Rotate<true>`) and sets
`config.data_already_rotated = true`, so the shared `Train` body then skips rotation and uses `data`
directly as `data_to_cluster` — no `n × d` samples buffer is allocated at all (it is now allocated
only `if (n_samples < n || !data_already_rotated)`. It forces `unrotate_centroids = false`: the caller's buffer is now in the rotated domain, so the centroids must be too

**`SuperKMeansState` / `GetState()`** — how training was actually carried out (`trained`,
`trained_in_place`, `training_data_rotated`, `code_size`, `n_encoded`, `rotator`), recorded at train
time and exposed read-only, alongside `GetQuantizer()`, `GetQuantizedData()` and public
`sampled_indices`. The rotation is exposed as the **pruner, not a matrix**, because the DCT path has
no d × d matrix to hand out. With `data_already_rotated` the pruner is built without a rotation
(no d × d QR, the costly part at high d): its `Rotate`/`Unrotate` then warn and copy the vectors.

**Assign family (3 methods):**
- **`Assign`** — exact f32 brute force. Standalone, no trained state. Ground-truth reference.
- **`QuantizedAssign`** — standalone quantized analog: fits a *fresh* quantizer on the input, no
  `Train()` needed, never touches trained state, rotates **only for RaBitQ** (safe: non-pruning
  GEMM path, so sq8/lvq4 re-fit on un-rotated input). For arbitrary/new vectors.
- **`AssignTrainingPoints`** — reuses trained state; **`vectors` must equal the `Train()` data**
  (`n_vectors == n_train`). Three paths: (1) **pruning reuse** if `sampling_fraction==1.0 &&
  SupportsPruning && !use_blas_only && d ≥ 128 (DIMENSION_THRESHOLD_FOR_PRUNING) && n_clusters >
  256 (N_CLUSTERS_THRESHOLD_FOR_PRUNING) && iters > 1`; (2) **GEMM-only reuse** if `sampling==1.0`
  but the gate fails; (3) **fallback to `QuantizedAssign`** if `sampling < 1.0`. It and
  `QuantizedAssign` need **not** agree per-point (different domains for sq8/lvq4) — validate
  **recall**, not per-point agreement.

**Stale caches (classic footgun).** The quantized path caches centroid norms, data norms, and
**partial-norms** for pruning (keyed on `partial_d`). Reading a cache an earlier step didn't
refresh for the current `partial_d`/size ⇒ stale read: wrong distances, or **SIGSEGV** (cache
sized for a different `partial_d`). Defense = **self-heal**: FindNearestNeighbor rebuilds when
inconsistent (`if (cached_partial_d_ != partial_d || cache.size() != n_x) recompute(...)`, in
sq8/lvq4/rabitq). Keep that guard; use `InvalidateCaches()` / `Ensure…()` at entry points.

**Hierarchical + `iters_refinement == 0`.** Phases: mesoclustering → fineclustering →
refinement; **`iters_refinement` defaults to 0**. `partial_d` is shrunk to the refinement value
(~`vertical_d/3`) **unconditionally before** the loop, but `CacheDataPartialNorms` for it is
**inside** the loop — so at `iters_refinement == 0` the loop never runs and the trained state has
a small `partial_d` with a partial-norms cache keyed to the fineclustering `partial_d`. A later
`AssignTrainingPoints` (pruning reuse) reads that mismatch — the SIGSEGV above, survived only by
the self-heal. Preserve it if you touch these caches.

### Executor binding

`SuperKMeans` (protected), `IQuantizer` and `ADSamplingPruner` inherit `ExecutorHolder`: a
**borrowed** executor bound once with `SetExecutor`; unbound, `GetExecutor()` returns a static
`SerialExecutor`. The `SuperKMeans` constructor binds `config.executor` (or an owned
`MakeDefaultExecutor(config.n_threads)`) to itself, its pruner, `f32_quantizer` and every
`CreateQuantizer()` result.
- A helper object you add must be bound too: unbound it runs **serially** — right results,
  silently slow.
- Public entry points (`Train`, `Assign`, `QuantizedAssign`, `AssignTrainingPoints`, hierarchical
  `Train`, pruner `Rotate`/`Unrotate`) open a `ParallelSection`, which calls `ReleaseThreads()` on
  return: ForkUnion's idle workers busy-wait, so its pool stops between calls (respawned lazily).
  A new public entry point needs one too.
- Code without an object (static `BatchComputer`, `U8Gemm`/`U4Gemm`,
  `ComputeScalarQuantizationParams`) takes a `ParallelExecutor&`. Free utilities (`ComputeWCSS`,
  `MakeBlobs`, `ComputeNorms`, brute-force kNN) take a trailing `ParallelExecutor* = nullptr` and
  open an `ExecutorScope` (null: a default pool for that call).

### Portability (MSVC, Wasm)

- No GCC builtins (`SKM_POPCOUNT`/`SKM_POPCOUNT64` in `common.h`); POSIX-only code goes behind
  `#if !defined(_WIN32)`.
- Wasm builds have no threads (serial executor) and no ruy/cpuinfo: NumKong runs every 8-bit GEMM
  (`IS_WASM`).

## Python bindings

One `SuperKMeans` class hides flat/hierarchical and f32/quantized selection. Full API:
`python/README.md`; source: `python/bindings/bindings.cpp` + `python/superkmeans/__init__.py`.
```python
kmeans = SuperKMeans(n_clusters=k, dimensionality=d, quantizer="rabitq")  # or f32/sq8/lvq4
centroids   = kmeans.train(data)               # float32 centroids (k×d)
assignments = kmeans.assign(data, centroids)   # exact; quantized_assign(...) needs train() first
```
`train(data, overwrite_input=True)` is the `TrainInPlace` path (name follows SciPy's
`overwrite_x`/`overwrite_a`). It dispatches to a **separate** `train_in_place` binding declared
`py::array_t<float, py::array::c_style>` — **no `forcecast`** — so pybind raises instead of
converting; otherwise a float64/F-order/sliced input would be silently rotated as a temporary while
the caller's array stayed untouched. `writeable()` is checked explicitly (pybind does not), and
`validate_numpy_array(..., overwrite=True)` refuses rather than substituting an
`ascontiguousarray` copy. `unrotate_centroids` is exposed for the normal path, and forced to False
on the in-place one.
