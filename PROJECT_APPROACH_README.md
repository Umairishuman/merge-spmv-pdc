# Project README: Our Approach, Implementation & Slides
**CS-3006 · Parallel & Distributed Computing · Spring 2026**
**Group:** Muhammad Umair (23i-0662) · Awais Basheer (23i-0506)
**Base Paper:** Merge-Based Parallel SpMV — Merrill & Garland, SC'16
**Selected Gap:** Gap A — SIMD Vectorization of the Inner Accumulation Loop

---

## 1. Project Scope

The paper by Merrill & Garland achieves **thread-level load balance** through merge-path decomposition but **leaves each thread's inner loop entirely scalar**. Every thread computes:

```cpp
y[row] += values[nz] * x[col_indices[nz]];    // scalar — no SIMD
```

Modern CPUs supporting **AVX2** can execute 4 double-precision FMAs per cycle using 256-bit SIMD registers. This potential is completely unused in the paper's implementation.

**Our Question:** Can AVX2 gather-based vectorization of the inner accumulation loop provide a meaningful per-thread throughput improvement on top of the already-balanced merge-path decomposition?

---

## 2. What We Implemented

### Algorithm 1 — Baseline: Sequential CsrMV
Standard row-based sequential SpMV. Used to verify correctness and establish a single-thread baseline.

```cpp
for (int row = 0; row < A.num_rows; ++row) {
    double sum = 0.0;
    for (int nz = row_offsets[row]; nz < row_offsets[row+1]; ++nz)
        sum += values[nz] * x[col_indices[nz]];
    y[row] = sum;
}
```

### Algorithm 2 — Merge-Path + Scalar Inner Loop (paper-faithful)
OpenMP parallelization using merge-path decomposition with a scalar inner loop (replicates the paper's CPU implementation).

```cpp
// Per thread: binary search for diagonal, then:
for (; row < row_end; ++row) {
    double sum = 0.0;
    for (int nz = row_offsets[row]; nz < row_offsets[row+1]; ++nz)
        sum += values[nz] * x[col_indices[nz]];   // scalar
    y[row] = sum;
    sum = 0.0;
}
```

### Algorithm 3 — Merge-Path + AVX2 SIMD Inner Loop (our contribution)
Same merge-path decomposition and OpenMP structure, but the inner accumulation loop uses **AVX2 `_mm256_i32gather_pd`** to load 4 `x[]` values simultaneously based on 4 `col_indices[]` at once.

```cpp
#include <immintrin.h>

// Process 4 non-zeros at a time using AVX2 gather
__m256d acc = _mm256_setzero_pd();
for (; nz + 4 <= row_end; nz += 4) {
    __m128i cols = _mm_loadu_si128((__m128i*)&col_indices[nz]); // 4 int32
    __m256d xv   = _mm256_i32gather_pd(x, cols, 8);             // gather x[cols]
    __m256d vals = _mm256_loadu_pd(&values[nz]);                 // 4 doubles
    acc = _mm256_fmadd_pd(vals, xv, acc);                        // FMA
}
// Horizontal reduce acc + scalar tail
double sum = hsum256_pd(acc);
for (; nz < row_end; ++nz)
    sum += values[nz] * x[col_indices[nz]];                      // scalar tail
y[row] = sum;
```

---

## 3. Tools & Build Environment

| Tool | Version / Detail |
|---|---|
| Language | C++17 |
| Parallelism | OpenMP 4.5 |
| Compiler | GCC with `-O3 -march=native -fopenmp -mavx2 -mfma` |
| SIMD | AVX2 gather intrinsics (`_mm256_i32gather_pd`) |
| Matrix format | SuiteSparse `.mtx` (Matrix Market format) |
| Platform | Linux multi-core (Ubuntu) |
| Matrices | Selected from [SuiteSparse Matrix Collection](https://sparse.tamu.edu) |

### Build

```bash
# Scalar baseline + merge-path
g++ -O3 -march=native -fopenmp -std=c++17 -o spmv_scalar   spmv_scalar.cpp
g++ -O3 -march=native -fopenmp -std=c++17 -o spmv_merge    spmv_merge.cpp

# AVX2 merge-path
g++ -O3 -march=native -fopenmp -mavx2 -mfma -std=c++17 -o spmv_avx2 spmv_avx2.cpp
```

### Run

```bash
export OMP_NUM_THREADS=8

# Correctness check (compare outputs)
./spmv_scalar --mtx=matrix.mtx > out_scalar.txt
./spmv_merge  --mtx=matrix.mtx > out_merge.txt
./spmv_avx2   --mtx=matrix.mtx > out_avx2.txt
diff out_scalar.txt out_merge.txt   # should be empty
diff out_scalar.txt out_avx2.txt    # should be empty (within float tolerance)

# Timing / throughput
./spmv_merge --mtx=matrix.mtx --threads=8 --iters=100
./spmv_avx2  --mtx=matrix.mtx --threads=8 --iters=100
```

---

## 4. Matrices Used

Selected from SuiteSparse to cover a range of coefficient of variation (CV):

| Matrix | Domain | Rows | NNZ | Row CV | Type |
|---|---|---|---|---|---|
| thermomech_dK | FEM / thermal | ~204K | 2,846,228 | 0.10 | Regular |
| crankseg_2 | Structural | ~63K | 14,148,858 | ~0.3 | Semi-regular |
| cfd2 | CFD | ~124K | 3,085,406 | ~0.5 | Semi-regular |
| cnr-2000 | Web graph | ~326K | 3,216,152 | 2.1 | Irregular |
| ASIC_320k | Circuit sim | ~321K | 2,635,364 | 61.4 | Highly irregular |

*Final matrix selection may be adjusted based on availability.*

---

## 5. Measurements & Metrics

| Metric | Formula | What It Shows |
|---|---|---|
| **GFLOPs/s** | `2 × nnz / time_s / 1e9` | Compute throughput |
| **Speedup** | `T_baseline / T_impl` | Relative improvement |
| **Parallel efficiency** | `Speedup / num_threads` | Quality of parallelism |
| **Memory bandwidth** | `bytes_accessed / time_s` | Bandwidth utilization |
| **SIMD benefit** | `T_scalar / T_avx2` (same thread count) | Vectorization gain |
| **Correctness** | Max absolute difference vs. sequential | Numerical accuracy |

### Thread Scaling Study
Threads: 1, 2, 4, 8, 16, (max available)
For each: run merge-scalar and merge-AVX2, record GFLOPs/s and efficiency.

---

## 6. Hardware-Level Analysis (Milestone 3 Requirement)

### Why SpMV Is Memory-Bound

SpMV has very low arithmetic intensity (AI):
```
Flops per NZ = 2  (1 multiply + 1 add)
Bytes per NZ ≈ 8 (values) + 4 (col_idx) + ~8 (x[] gather) = ~20 bytes
AI = 2 / 20 = 0.1 FLOPs/byte
```

On a modern CPU with ~50 GB/s bandwidth and ~100 GFLOPs/s peak:
- Ridge point ≈ `peak_compute / peak_bandwidth` = 100 / 50 = **2 FLOPs/byte**
- SpMV AI = 0.1 << 2 → **deep memory-bound territory**

### What This Means for Our SIMD Approach

| Loop | What it touches | SIMD benefit? |
|---|---|---|
| `values[]` read | Sequential → very cache-friendly | ✓ AVX2 load |
| `row_offsets[]` read | Sequential, tiny → stays in L1 | ✓ Already fast |
| `col_indices[]` read | Sequential → cache-friendly | ✓ AVX2 load |
| `x[col_indices[nz]]` gather | **Random access** → frequent L3/RAM misses | ✗ Gather latency ≈ scalar |

SIMD can widen the FP pipeline but **cannot eliminate the gather penalty**. Our hypothesis is that SIMD helps on **regular matrices** (where `x[]` access has spatial locality) but provides diminishing returns on **highly irregular matrices** (where `x[]` gathers are pure random access).

### Roofline Analysis Plan

```
Step 1: Measure hardware limits
    - Peak BW: STREAM benchmark (./stream)
    - Peak compute: likwid or ERT

Step 2: Measure our implementation
    - Achieved BW = bytes_read / runtime
    - Achieved GFLOPs = 2*nnz / runtime

Step 3: Plot on roofline
    - X-axis: AI (FLOPs/byte)
    - Y-axis: GFLOPs/s
    - Show roof lines for BW and compute
    - Plot our (AI, GFLOPs) point per matrix

Step 4: Explain position
    - Below memory-BW roof → memory-bound (expected)
    - Gap between achieved BW and peak BW → cache miss overhead
```

### Hardware Profiling Commands

```bash
# Memory bandwidth (STREAM)
gcc -O3 -fopenmp stream.c -o stream && ./stream

# Cache miss analysis
perf stat -e cache-misses,cache-references,L1-dcache-load-misses \
    ./spmv_avx2 --mtx=ASIC_320k.mtx --threads=8

# FLOP counting (if using likwid)
likwid-perfctr -C 0-7 -g FLOPS_DP ./spmv_avx2 --mtx=matrix.mtx

# Bandwidth measurement
likwid-perfctr -C 0-7 -g MEM ./spmv_avx2 --mtx=matrix.mtx
```

---

## 7. Expected Results & Hypothesis

| Scenario | Expected Behavior |
|---|---|
| Regular matrix (CV ≈ 0.1) | AVX2 provides 1.5–2.5× speedup over scalar inner loop; x[] access has locality |
| Irregular matrix (CV > 10) | AVX2 provides minimal benefit (<1.2×); gather latency dominates |
| Thread scaling (1→max, regular) | Near-linear until memory bandwidth saturates |
| Thread scaling (1→max, irregular) | Saturates earlier; imbalance eliminated but gather misses increase with concurrency |
| Roofline position | Both scalar and AVX2 will be memory-bound; AVX2 edges closer to the BW roof |

**Expected insight:** Near-linear speedup with thread count on irregular matrices confirms merge-path's load-balance property. The bandwidth saturation point reveals the true scaling limit.

---

## 8. Slide Deck Summary (Milestone 2 — Already Presented)

| Slide | Title | Key Content |
|---|---|---|
| 1 | Title slide | Names, paper, CS-3006 |
| 2 | What is SpMV & Why Does It Matter? | y=Ax, CSR format, 4 application domains, GPU imbalance problem |
| 3 | The Problem: Load Imbalance in Row-Splitting | Thread workload diagram, real-world cuSPARSE 100× degradation table |
| 4 | Core Idea: Merge-Path Decomposition | 2D merge grid, 5-step algorithm, binary search |
| 5 | Parallelization Strategy & Cache Behavior | Decomposition/Assignment/Orchestration stages, cache-friendly vs. unfriendly arrays |
| 6 | Paper Results | 198× GPU speedup, 1.6× avg CPU, 4,201 matrices, 0ms preprocessing |
| 7 | Gap Analysis | Gap A: SIMD (selected), Gap B: GPU portability, Gap C: dynamic sparsity, Gap D: scaling study |
| 8–9 | Algorithm 2 (paper code) | Full OpenMP C++ implementation from the paper |
| 10 | Our Proposed Approach | Scope (AVX2 gather), metrics, timeline, tools |

---

## 9. Timeline

| Days | Activity |
|---|---|
| Days 1–2 | Baseline sequential CsrMV + CSR `.mtx` file loader + correctness verification |
| Days 3–4 | Merge-path OpenMP implementation (paper-faithful scalar inner loop) |
| Days 5–6 | AVX2 gather implementation + scaling experiments + data collection |
| Day 7 | Roofline analysis, charts, report writing, slide finalization |

---

## 10. Final Presentation Outline (Milestone 3)

1. **Brief paper recap** (2 min) — Problem, merge-path idea, paper results
2. **Project scope** (1 min) — Gap A chosen; what we added
3. **Implementation details** (3 min) — Scalar vs. AVX2 inner loop; build setup
4. **Baseline results** (2 min) — Sequential timings; correctness
5. **Parallel results** (4 min)
   - Speedup vs. thread count (regular + irregular matrices)
   - Efficiency curves
   - SIMD benefit (scalar vs. AVX2 per thread count)
6. **Hardware analysis** (3 min)
   - Roofline plot — where our implementation sits
   - Cache miss rates — regular vs. irregular
   - Bandwidth utilization — how close to peak
   - Answer: compute-bound / memory-bound / synchronization-limited?
7. **LLM Disclosure** (1 min) — Tools used, prompts, integrity statement
8. **Conclusions** (1 min)
9. **Q&A + Live Demo**

---

## 11. LLM Usage Disclosure (Template)

> This section must be included in the final report and presentation per Milestone 3 requirements.

- **Tools used:** [e.g., Claude, ChatGPT]
- **Where used:** Understanding paper notation, debugging AVX2 intrinsic syntax, reviewing report draft for grammar
- **Representative prompts:**
  - *"Explain what `_mm256_i32gather_pd` does and how to use it for SpMV"*
  - *"Review this OpenMP code for race conditions in the fix-up pass"*
- **Benefits:** Faster debugging, clearer understanding of intrinsic API
- **Limitations:** LLM suggested incorrect gather scale parameter; had to verify against Intel intrinsics guide
- **Correctness risks:** All generated code was independently tested against scalar output; no LLM output was submitted without manual verification
- **Academic integrity:** All analysis, experimental design, and conclusions are our own original work
