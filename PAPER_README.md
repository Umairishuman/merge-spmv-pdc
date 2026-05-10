# Paper README: Merge-Based Parallel Sparse Matrix-Vector Multiplication
**Merrill & Garland, NVIDIA Corporation — SC'16 (Supercomputing 2016)**
**Full title:** *Merge-based Parallel Sparse Matrix-Vector Multiplication*

---

## 1. What Problem Does the Paper Solve?

SpMV (Sparse Matrix-Vector Multiplication) computes **y = A·x**, where A is sparse and x, y are dense vectors. It is a performance-critical kernel appearing inside:
- Iterative linear solvers (Krylov subspace, eigenvalue systems)
- Graph algorithms (PageRank, BFS via GraphBLAS)
- Machine learning (GNN, recommender systems)
- Finite element / PDE simulations

The dominant in-memory format is **CSR (Compressed Sparse Row)** — three arrays:
- `values[]` — non-zero matrix entries
- `col_indices[]` — column index per non-zero
- `row_offsets[]` — index into the above two arrays for the start of each row

### The Core Problem: Load Imbalance

Traditional CsrMV parallelizations assign rows to threads (**row splitting**) or non-zeros to threads (**nonzero splitting**). Both fail under irregular sparsity:

| Strategy | Failure Mode |
|---|---|
| Row splitting | A thread assigned one huge row runs 100× longer than a thread with 100 tiny rows |
| Nonzero splitting | Processors may consume wildly different numbers of row offsets (especially for hypersparse / graph matrices) |

**Real-world evidence from the paper** (all ~3M non-zeros, same matrix size):

| Matrix | Row CV | Intel MKL CPU | NVIDIA cuSPARSE GPU |
|---|---|---|---|
| thermomech_dK (temperature) | 0.10 | 17.9 GFLOPs | 12.4 GFLOPs |
| cnr-2000 (web graph) | 2.1 | 13.4 GFLOPs | 5.9 GFLOPs |
| ASIC_320k (circuit sim) | 61.4 | 11.8 GFLOPs | **0.12 GFLOPs** |

cuSPARSE degrades **100×** on the highly irregular matrix. This is the exact problem the paper targets.

---

## 2. Key Contribution: Merge-Path Decomposition

### The Central Idea

Frame the parallel CsrMV decomposition as **logically merging two sorted lists**:
- **List A**: `row_offsets[]` (row end-offsets, length = `num_rows`)
- **List B**: `ℕ` — the natural numbers indexing non-zeros (length = `nnz`)

The merge decision path has length `num_rows + nnz`. Divide this path into **p equal diagonal slices** — each thread gets exactly `(num_rows + nnz) / p` merge steps, regardless of row length distribution.

### Why This Is Balanced

No thread can ever be assigned:
- (a) an arbitrarily long row (bounded by the diagonal length)
- (b) an arbitrarily large number of empty rows (bounded by the same)

### The Algorithm (CPU OpenMP — Algorithm 2 from paper)

```
total_merge_items = num_rows + nnz
items_per_thread  = ceil(total_merge_items / p)

for each thread tid in parallel:
    diagonal     = items_per_thread * tid
    diagonal_end = diagonal + items_per_thread

    # Binary search O(log n) — find (row_idx, nz_idx) at each diagonal
    (row_start, nz_start) = MergePathSearch(diagonal, row_offsets, ℕ, num_rows, nnz)
    (row_end,   nz_end  ) = MergePathSearch(diagonal_end, ...)

    # Sequential CsrMV on this slice
    running_total = 0.0
    for row in [row_start, row_end):
        for nz in [row_offsets[row], row_offsets[row+1]):
            running_total += values[nz] * x[col_indices[nz]]
        y[row] = running_total; running_total = 0.0

    # Handle partial last row (may be shared with next thread)
    for nz in [nz_start_of_last_row, nz_end):
        running_total += values[nz] * x[col_indices[nz]]
    save (row_end, running_total) as carry-out

# Sequential fix-up pass (rows that span multiple threads)
for tid in [0, p-1):
    if carry_out_row[tid] < num_rows:
        y[carry_out_row[tid]] += carry_out_value[tid]
```

### The Binary Search (MergePathSearch)

Along diagonal k, find the first coordinate (i, j) where `row_offsets[i] > j`, given `i + j = k`. This is a standard 1D binary search constrained to the diagonal — O(log n) per thread, O(N) total.

```cpp
CoordinateT MergePathSearch(int diagonal, ...) {
    int x_min = max(diagonal - b_len, 0);
    int x_max = min(diagonal, a_len);
    while (x_min < x_max) {
        int pivot = (x_min + x_max) >> 1;
        if (a[pivot] <= b[diagonal - pivot - 1])
            x_min = pivot + 1;
        else
            x_max = pivot;
    }
    return { min(x_min, a_len), diagonal - x_min };
}
```

---

## 3. Visualizing the Merge Grid

```
Row offsets:  [0, 2, 2, 4, 8]        (4 rows, last ends at 8)
NZ indices:   [0, 1, 2, 3, 4, 5, 6, 7]

       → Row offsets (List A)
       0    2    2    4    8
  ↓  ┌────┬────┬────┬────┐
  0  │ p0 │    │    │    │
  1  │    │ p0 │    │    │
  2  │    │p0/1│ p1 │    │
  3  │    │    │ p1 │    │
  4  │    │    │    │ p2 │
  5  │    │    │    │ p2 │
  6  │    │    │    │ p2 │
  7  │    │    │    │ p2 │
  ↓  └────┴────┴────┴────┘
  NZ indices (List B)

↓ = accumulate NZ · x[col]    → = flush y[row] and reset accumulator
```

---

## 4. Implementations

### CPU (OpenMP C++)
- Static scheduling: one diagonal slice per thread
- Thread affinity pinned to prevent migration across sockets
- **No** architecture-specific intrinsics (no AVX, no SIMD, no prefetch)
- Intentionally simple to demonstrate that balance alone drives performance

### CPU NUMA Optimization
- One-time inspector pass: identify which `values[]` / `col_indices[]` pages belong to each thread
- Migrate those pages to the NUMA node where that thread runs
- Algorithm unchanged; only OS page placement changes
- Reported as preprocessing overhead (~44× one SpMV time)

### GPU (CUDA C++)
- **Two-level** merge-path decomposition matching GPU's two-level hierarchy:
  1. Coarse level: divide merge path across thread blocks (fixed count = SM count)
  2. Fine level: each block processes the path in fixed-size **path-chunks** (e.g., 896 = 128 threads × 7 items)
- Strip-mined, coalesced loads of `row_offsets` and `values` into shared memory
- Single unified loop per thread (Algorithm 4 in paper) avoids doubly-nested branch divergence
- Two fix-up passes: block-wide (CUB) + device-wide (CUB)

---

## 5. Performance Results

### Experimental Setup

| Component | Spec |
|---|---|
| CPU | 2× Intel Xeon E5-2690v2 (10 cores, 2-way HT = 40 threads total) |
| CPU Cache | 25 MB L3 per socket (50 MB total) |
| CPU Bandwidth | 77.9 GB/s (Stream Triad) |
| GPU | NVIDIA Tesla K40 |
| GPU Parallelism | 15 SMs, 960 warps (30k threads) |
| GPU Cache | 2.3 MB aggregate (L2 + texture) |
| GPU Bandwidth | 249 GB/s (Stream Triad, ECC off) |
| Test corpus | 4,201 matrices — University of Florida Sparse Matrix Collection |

### CPU Speedup vs Intel MKL

| Configuration | Max Speedup | Min Speedup | Harmonic Mean (small) | Harmonic Mean (large) |
|---|---|---|---|---|
| Merge-based CsrMV | 15.8× | 0.51× | 1.22× | 1.06× |
| NUMA Merge-based CsrMV | 15.7× | 0.50× | 1.25× | **1.58×** |
| vs CSB SpMV | 445× | 0.65× | 9.21× | 1.09× |
| vs pOSKI SpMV | 24.4× | 0.59× | 11.0× | 1.10× |

### GPU Speedup vs NVIDIA cuSPARSE

| Configuration | Max Speedup | Min Speedup | Harmonic Mean (small) | Harmonic Mean (large) |
|---|---|---|---|---|
| vs cuSPARSE CsrMV | **198×** | 0.34× | 0.79× | 1.13× |
| vs cuSPARSE HybMV | 5.96× | 0.24× | 1.41× | 0.96× |
| vs yaSpMV BccooMV (fp32) | 2.43× | 0.39× | 0.78× | 0.75× |

### Key Observations

- **Performance predictability**: Merge-based CsrMV's runtime correlates with nnz at R=0.97 (CPU) / R=0.87 (GPU). cuSPARSE achieves only R=0.30.
- **Row-length imperviousness**: Correlation of GFLOPs to row-length variation is -0.01 (merge-based GPU) vs. -0.24 (cuSPARSE).
- **GPU on small matrices**: Up to 50% slower than cuSPARSE because merge-path binary search + second fix-up kernel overhead exceeds the SpMV work itself.
- **Remaining variation**: Differences in cache hit rate for random `x[]` accesses — not load imbalance.

---

## 6. Comparison with Specialized Formats

| Format | Preprocessing | Our Speedup | Notes |
|---|---|---|---|
| CSR (MKL) | None | 1.21× avg | Row-based, load imbalance |
| CSR (cuSPARSE) | None | 0.84× avg | Vectorized row-based |
| CSB | None | 6.58× avg | Nested COO-of-COO, Morton Z-order |
| HYB (ELL+COO) | 19× | 1.29× avg | ELL portion + COO fix-up |
| pOSKI | 484× | 7.86× avg | Per-matrix auto-tuned blocking |
| yaSpMV BCCOO (fp32) | 155,000× | 0.78× avg | Bit-compressed, 50% index savings |

The merge-based method matches or exceeds specialized formats **without any preprocessing**.

---

## 7. Data Format Reference (CSR)

```
Matrix A (4×5):
[ 1.0  --  1.0  --  -- ]
[ --   --  --   --  -- ]
[ --   --  3.0  3.0 -- ]
[ 4.0  4.0 4.0  1.0 -- ]

values[]      = [1.0, 1.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0]
col_indices[] = [0,   2,   2,   3,   0,   1,   2,   3  ]
row_offsets[] = [0,   2,   2,   4,   8]
                 ↑row0 ↑row1 ↑row2 ↑row3  ↑end
```

---

## 8. Limitations Identified by the Paper

1. **No SIMD vectorization** of the inner accumulation loop (kept scalar)
2. **GPU results are NVIDIA-only** (Tesla K40 — no AMD/integrated GPU testing)
3. **Static matrices** — no streaming or dynamic sparsity support
4. **Fixed hardware** — no strong/weak scaling curves (thread count vs. speedup)
5. **Small GPU matrices** — second kernel overhead hurts latency-sensitive workloads

---

## 9. Source Code

The artifact is publicly available:
```bash
git clone https://github.com/dumerrill/merge-spmv.git
cd merge-spmv
make cpu_spmv
make gpu_spmv sm=350
./cpu_spmv --mtx=path/to/matrix.mtx
./gpu_spmv --mtx=path/to/matrix.mtx
```

Test matrices: [SuiteSparse Matrix Collection](https://sparse.tamu.edu)

---

## 10. References (Key)

- Merrill & Garland, "Merge-based Parallel Sparse Matrix-Vector Multiplication," SC'16
- Bell & Garland, "Implementing SpMV on throughput-oriented processors," SC'09
- Greathouse & Daga, "Efficient SpMV on GPUs using CSR," SC'14
- Williams et al., "SpMV on Multicore and Accelerators," 2011
- University of Florida (SuiteSparse) Sparse Matrix Collection
