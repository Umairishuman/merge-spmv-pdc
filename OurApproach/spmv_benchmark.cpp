/*******************************************************************************
 * spmv_benchmark.cpp  —  FIXED VERSION v3
 *
 * Changes vs v2:
 *   0. APPROACH 0 ADDED — Author's original OmpMergeCsrmv from the paper
 *      (Merrill & Garland, SC'16) ported faithfully to double precision.
 *      All speedups are now reported relative to this author baseline.
 *      This is the "target to beat" as required by the assignment.
 *
 *   Other fixes carried over from v2:
 *   1. pf_dist is a CLI parameter (--pfdist=N, default 32).
 *   2. Effective memory bandwidth (GB/s) alongside GFLOPs.
 *   3. CSV header bug fixed.
 *   4. Thread affinity via setenv("OMP_PROC_BIND","close").
 *
 * CS-3006 · Parallel & Distributed Computing · Spring 2026
 * Muhammad Umair (23i-0662) · Awais Basheer (23i-0506)
 ******************************************************************************/

#include <omp.h>
#include <immintrin.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>

/* =========================================================================
 * Aligned allocation helpers
 * ========================================================================= */
static inline void* aligned_new(std::size_t bytes, std::size_t align = 64)
{
#if defined(_WIN32)
    return _aligned_malloc(bytes, align);
#else
    void* ptr = nullptr;
    if (posix_memalign(&ptr, align, bytes) != 0) ptr = nullptr;
    return ptr;
#endif
}
static inline void aligned_del(void* ptr)
{
#if defined(_WIN32)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

/* =========================================================================
 * CSR Matrix
 * ========================================================================= */
struct CsrMatrix {
    int     num_rows     = 0;
    int     num_cols     = 0;
    int     num_nonzeros = 0;
    int*    row_offsets  = nullptr;   // size num_rows + 1
    int*    col_indices  = nullptr;   // size num_nonzeros
    double* values       = nullptr;   // size num_nonzeros

    CsrMatrix() = default;
    ~CsrMatrix() {
        aligned_del(row_offsets);
        aligned_del(col_indices);
        aligned_del(values);
    }
    CsrMatrix(const CsrMatrix&) = delete;
    CsrMatrix& operator=(const CsrMatrix&) = delete;
};

/* =========================================================================
 * Matrix Market reader  (real / pattern / symmetric)
 * ========================================================================= */
bool LoadMatrixMarket(const std::string& path, CsrMatrix& csr)
{
    std::ifstream fin(path);
    if (!fin.is_open()) {
        fprintf(stderr, "ERROR: cannot open '%s'\n", path.c_str());
        return false;
    }
    std::string line;
    std::getline(fin, line);
    bool is_pattern   = (line.find("pattern")   != std::string::npos);
    bool is_symmetric = (line.find("symmetric") != std::string::npos ||
                         line.find("hermitian") != std::string::npos);
    while (fin.peek() == '%') std::getline(fin, line);

    int M, N, NNZ;
    fin >> M >> N >> NNZ;

    struct Triple { int r, c; double v; };
    std::vector<Triple> coo;
    coo.reserve(is_symmetric ? NNZ * 2 : NNZ);

    for (int k = 0; k < NNZ; ++k) {
        int r, c; double v = 1.0;
        fin >> r >> c;
        if (!is_pattern) fin >> v;
        --r; --c;
        coo.push_back({r, c, v});
        if (is_symmetric && r != c)
            coo.push_back({c, r, v});
    }
    fin.close();

    std::sort(coo.begin(), coo.end(), [](const Triple& a, const Triple& b){
        return a.r < b.r || (a.r == b.r && a.c < b.c);
    });

    int nnz = (int)coo.size();
    csr.num_rows     = M;
    csr.num_cols     = N;
    csr.num_nonzeros = nnz;

    csr.row_offsets = (int*)   aligned_new(sizeof(int)    * (M + 1));
    csr.col_indices = (int*)   aligned_new(sizeof(int)    * nnz);
    csr.values      = (double*)aligned_new(sizeof(double) * nnz);

    memset(csr.row_offsets, 0, sizeof(int) * (M + 1));
    for (auto& t : coo) csr.row_offsets[t.r + 1]++;
    for (int i = 0; i < M; ++i) csr.row_offsets[i+1] += csr.row_offsets[i];
    for (int k = 0; k < nnz; ++k) {
        csr.col_indices[k] = coo[k].c;
        csr.values[k]      = coo[k].v;
    }
    return true;
}

/* =========================================================================
 * High-resolution timer
 * ========================================================================= */
struct Timer {
    using Clock = std::chrono::high_resolution_clock;
    Clock::time_point t0;
    void start() { t0 = Clock::now(); }
    double elapsedMs() const {
        return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    }
};

/* =========================================================================
 * MergePath binary search (shared by all approaches)
 *   Finds (row_idx, nz_idx) where diagonal d intersects the merge path.
 * ========================================================================= */
static inline void MergePathSearch(
    int diagonal,
    const int* row_end_offsets,
    int num_rows, int num_nonzeros,
    int& row_idx, int& nz_idx)
{
    int x_min = std::max(diagonal - num_nonzeros, 0);
    int x_max = std::min(diagonal, num_rows);
    while (x_min < x_max) {
        int pivot = (x_min + x_max) >> 1;
        if (row_end_offsets[pivot] <= diagonal - pivot - 1)
            x_min = pivot + 1;
        else
            x_max = pivot;
    }
    row_idx = std::min(x_min, num_rows);
    nz_idx  = diagonal - x_min;
}

/* =========================================================================
 * AVX2 helper: gather 4 doubles from x[] using 4 int32 indices
 * ========================================================================= */
static inline __m256d gather4(const int* col_ptr, const double* x)
{
    __m128i vi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(col_ptr));
    return _mm256_i32gather_pd(x, vi, 8 /* scale = sizeof(double) */);
}

/* =========================================================================
 * Approach 0 – Author's Original OmpMergeCsrmv  (Paper: Merrill & Garland)
 *
 *  This is the BASELINE to beat.  Ported faithfully from the author's
 *  reference implementation (mergebased_spmv.cpp) to double precision.
 *  The only change is float → double; the algorithm, loop structure,
 *  and carry-out fix-up are identical to the paper's reference code.
 * ========================================================================= */
void SpMV_Author_Baseline(
    const CsrMatrix& A,
    const double* __restrict__ x,
    double* __restrict__ y,
    int num_threads,
    int* row_carry_out, double* val_carry_out)
{
    // row_end_offsets = A.row_offsets + 1  (author's pointer trick)
    const int* row_end_offsets = A.row_offsets + 1;
    const int  num_merge_items = A.num_rows + A.num_nonzeros;

    #pragma omp parallel for schedule(static) num_threads(num_threads)
    for (int tid = 0; tid < num_threads; tid++)
    {
        int items_per_thread = (num_merge_items + num_threads - 1) / num_threads;
        int start_diagonal   = std::min(items_per_thread * tid,        num_merge_items);
        int end_diagonal     = std::min(start_diagonal + items_per_thread, num_merge_items);

        int row_s, nz_s, row_e, nz_e;
        MergePathSearch(start_diagonal, row_end_offsets,
                        A.num_rows, A.num_nonzeros, row_s, nz_s);
        MergePathSearch(end_diagonal,   row_end_offsets,
                        A.num_rows, A.num_nonzeros, row_e, nz_e);

        // Consume whole rows
        for (; row_s < row_e; ++row_s) {
            double running_total = 0.0;
            for (; nz_s < row_end_offsets[row_s]; ++nz_s)
                running_total += A.values[nz_s] * x[A.col_indices[nz_s]];
            y[row_s] = running_total;
        }

        // Consume partial portion of thread's last row
        double running_total = 0.0;
        for (; nz_s < nz_e; ++nz_s)
            running_total += A.values[nz_s] * x[A.col_indices[nz_s]];

        // Save carry-outs
        row_carry_out[tid] = row_e;
        val_carry_out[tid] = running_total;
    }

    // Carry-out fix-up (rows spanning multiple threads)
    for (int tid = 0; tid < num_threads - 1; ++tid)
        if (row_carry_out[tid] < A.num_rows)
            y[row_carry_out[tid]] += val_carry_out[tid];
}

/* =========================================================================
 * Approach 1 – Sequential baseline  (Algorithm 1 in the paper)
 * ========================================================================= */
void SpMV_Sequential(const CsrMatrix& A,
                     const double* __restrict__ x,
                     double* __restrict__ y)
{
    for (int row = 0; row < A.num_rows; ++row) {
        double sum = 0.0;
        for (int k = A.row_offsets[row]; k < A.row_offsets[row+1]; ++k)
            sum += A.values[k] * x[A.col_indices[k]];
        y[row] = sum;
    }
}

/* =========================================================================
 * Approach 2 – OpenMP Merge-Path Scalar  (Algorithm 2 in the paper)
 * ========================================================================= */
void SpMV_OMP_MergePath_Scalar(
    const CsrMatrix& A,
    const double* __restrict__ x,
    double* __restrict__ y,
    int num_threads,
    int* row_carry_out, double* val_carry_out)
{
    const int* row_end  = A.row_offsets + 1;
    const int num_merge = A.num_rows + A.num_nonzeros;

    #pragma omp parallel for schedule(static) num_threads(num_threads)
    for (int tid = 0; tid < num_threads; ++tid)
    {
        int items_per_thread = (num_merge + num_threads - 1) / num_threads;
        int start_diag = std::min(items_per_thread * tid,        num_merge);
        int end_diag   = std::min(start_diag + items_per_thread, num_merge);

        int row_s, nz_s, row_e, nz_e;
        MergePathSearch(start_diag, row_end, A.num_rows, A.num_nonzeros, row_s, nz_s);
        MergePathSearch(end_diag,   row_end, A.num_rows, A.num_nonzeros, row_e, nz_e);

        // Whole rows
        for (; row_s < row_e; ++row_s) {
            double sum = 0.0;
            for (; nz_s < row_end[row_s]; ++nz_s)
                sum += A.values[nz_s] * x[A.col_indices[nz_s]];
            y[row_s] = sum;
        }
        // Partial last row
        double carry = 0.0;
        for (; nz_s < nz_e; ++nz_s)
            carry += A.values[nz_s] * x[A.col_indices[nz_s]];

        row_carry_out[tid] = row_e;
        val_carry_out[tid] = carry;
    }
    // Fix-up: rows that span multiple threads
    for (int tid = 0; tid < num_threads - 1; ++tid)
        if (row_carry_out[tid] < A.num_rows)
            y[row_carry_out[tid]] += val_carry_out[tid];
}

/* =========================================================================
 * Approach 3 – OpenMP Merge-Path AVX2  (vectorised inner loop, 4 doubles)
 * ========================================================================= */
void SpMV_OMP_MergePath_AVX2(
    const CsrMatrix& A,
    const double* __restrict__ x,
    double* __restrict__ y,
    int num_threads,
    int* row_carry_out, double* val_carry_out)
{
    const int* row_end  = A.row_offsets + 1;
    const int num_merge = A.num_rows + A.num_nonzeros;

    #pragma omp parallel for schedule(static) num_threads(num_threads)
    for (int tid = 0; tid < num_threads; ++tid)
    {
        int items_per_thread = (num_merge + num_threads - 1) / num_threads;
        int start_diag = std::min(items_per_thread * tid,        num_merge);
        int end_diag   = std::min(start_diag + items_per_thread, num_merge);

        int row_s, nz_s, row_e, nz_e;
        MergePathSearch(start_diag, row_end, A.num_rows, A.num_nonzeros, row_s, nz_s);
        MergePathSearch(end_diag,   row_end, A.num_rows, A.num_nonzeros, row_e, nz_e);

        for (; row_s < row_e; ++row_s) {
            int end_nz = row_end[row_s];
            __m256d vsum = _mm256_setzero_pd();
            int k = nz_s;
            for (; k + 3 < end_nz; k += 4) {
                __m256d vval = _mm256_loadu_pd(&A.values[k]);
                __m256d vx   = gather4(&A.col_indices[k], x);
                vsum = _mm256_fmadd_pd(vval, vx, vsum);
            }
            // Horizontal reduce
            __m128d lo  = _mm256_castpd256_pd128(vsum);
            __m128d hi  = _mm256_extractf128_pd(vsum, 1);
            __m128d s2  = _mm_add_pd(lo, hi);
            double  sum = _mm_cvtsd_f64(_mm_hadd_pd(s2, s2));
            for (; k < end_nz; ++k)
                sum += A.values[k] * x[A.col_indices[k]];
            y[row_s] = sum;
            nz_s = end_nz;
        }
        double carry = 0.0;
        for (; nz_s < nz_e; ++nz_s)
            carry += A.values[nz_s] * x[A.col_indices[nz_s]];
        row_carry_out[tid] = row_e;
        val_carry_out[tid] = carry;
    }
    for (int tid = 0; tid < num_threads - 1; ++tid)
        if (row_carry_out[tid] < A.num_rows)
            y[row_carry_out[tid]] += val_carry_out[tid];
}

/* =========================================================================
 * Approach 4 – OpenMP Merge-Path AVX2 ×2 Unrolled (8 doubles / iter)
 *   Two independent FMA accumulators break the 4-cycle FMA latency chain.
 * ========================================================================= */
void SpMV_OMP_MergePath_AVX2_Unroll2(
    const CsrMatrix& A,
    const double* __restrict__ x,
    double* __restrict__ y,
    int num_threads,
    int* row_carry_out, double* val_carry_out)
{
    const int* row_end  = A.row_offsets + 1;
    const int num_merge = A.num_rows + A.num_nonzeros;

    #pragma omp parallel for schedule(static) num_threads(num_threads)
    for (int tid = 0; tid < num_threads; ++tid)
    {
        int items_per_thread = (num_merge + num_threads - 1) / num_threads;
        int start_diag = std::min(items_per_thread * tid,        num_merge);
        int end_diag   = std::min(start_diag + items_per_thread, num_merge);

        int row_s, nz_s, row_e, nz_e;
        MergePathSearch(start_diag, row_end, A.num_rows, A.num_nonzeros, row_s, nz_s);
        MergePathSearch(end_diag,   row_end, A.num_rows, A.num_nonzeros, row_e, nz_e);

        for (; row_s < row_e; ++row_s) {
            int end_nz = row_end[row_s];
            __m256d vs0 = _mm256_setzero_pd();
            __m256d vs1 = _mm256_setzero_pd();
            int k = nz_s;
            for (; k + 7 < end_nz; k += 8) {
                vs0 = _mm256_fmadd_pd(_mm256_loadu_pd(&A.values[k]),
                                      gather4(&A.col_indices[k],   x), vs0);
                vs1 = _mm256_fmadd_pd(_mm256_loadu_pd(&A.values[k+4]),
                                      gather4(&A.col_indices[k+4], x), vs1);
            }
            __m256d vsT = _mm256_add_pd(vs0, vs1);
            for (; k + 3 < end_nz; k += 4)
                vsT = _mm256_fmadd_pd(_mm256_loadu_pd(&A.values[k]),
                                      gather4(&A.col_indices[k], x), vsT);
            __m128d lo  = _mm256_castpd256_pd128(vsT);
            __m128d hi  = _mm256_extractf128_pd(vsT, 1);
            __m128d s2  = _mm_add_pd(lo, hi);
            double  sum = _mm_cvtsd_f64(_mm_hadd_pd(s2, s2));
            for (; k < end_nz; ++k)
                sum += A.values[k] * x[A.col_indices[k]];
            y[row_s] = sum;
            nz_s = end_nz;
        }
        double carry = 0.0;
        for (; nz_s < nz_e; ++nz_s)
            carry += A.values[nz_s] * x[A.col_indices[nz_s]];
        row_carry_out[tid] = row_e;
        val_carry_out[tid] = carry;
    }
    for (int tid = 0; tid < num_threads - 1; ++tid)
        if (row_carry_out[tid] < A.num_rows)
            y[row_carry_out[tid]] += val_carry_out[tid];
}

/* =========================================================================
 * Approach 5 – AVX2 Unrolled×2 + tunable software prefetch
 *
 *  pf_dist: look-ahead in nonzeros.
 *    Too small → prefetch arrives too late (still a cache miss)
 *    Too large → cache pollution (evicts data we're about to touch)
 *    Typical sweet spot on modern x86:
 *      Regular matrices  (large rows, good locality): 32–64
 *      Irregular matrices (short rows, random x[]):   16–32
 *    Set --pfdist=0 to disable (identical to Approach 4).
 * ========================================================================= */
void SpMV_OMP_MergePath_AVX2_Prefetch(
    const CsrMatrix& A,
    const double* __restrict__ x,
    double* __restrict__ y,
    int num_threads,
    int* row_carry_out, double* val_carry_out,
    int pf_dist)
{
    const int* row_end  = A.row_offsets + 1;
    const int num_merge = A.num_rows + A.num_nonzeros;
    const int nnz       = A.num_nonzeros;

    #pragma omp parallel for schedule(static) num_threads(num_threads)
    for (int tid = 0; tid < num_threads; ++tid)
    {
        int items_per_thread = (num_merge + num_threads - 1) / num_threads;
        int start_diag = std::min(items_per_thread * tid,        num_merge);
        int end_diag   = std::min(start_diag + items_per_thread, num_merge);

        int row_s, nz_s, row_e, nz_e;
        MergePathSearch(start_diag, row_end, A.num_rows, A.num_nonzeros, row_s, nz_s);
        MergePathSearch(end_diag,   row_end, A.num_rows, A.num_nonzeros, row_e, nz_e);

        for (; row_s < row_e; ++row_s) {
            int end_nz = row_end[row_s];
            __m256d vs0 = _mm256_setzero_pd();
            __m256d vs1 = _mm256_setzero_pd();
            int k = nz_s;
            for (; k + 7 < end_nz; k += 8) {
                // Issue prefetches only when pf_dist > 0 and within bounds
                if (pf_dist > 0) {
                    int pf = k + pf_dist;
                    if (pf + 7 < nnz) {
                        __builtin_prefetch(&A.values[pf],        0, 1); // into L2
                        __builtin_prefetch(&A.col_indices[pf],   0, 1);
                        // Prefetch x[] at the gather addresses (best-effort;
                        // irregular access means these may still miss, but
                        // the hardware prefetcher can't handle them at all)
                        __builtin_prefetch(&x[A.col_indices[pf]],   0, 1);
                        __builtin_prefetch(&x[A.col_indices[pf+4]], 0, 1);
                    }
                }
                vs0 = _mm256_fmadd_pd(_mm256_loadu_pd(&A.values[k]),
                                      gather4(&A.col_indices[k],   x), vs0);
                vs1 = _mm256_fmadd_pd(_mm256_loadu_pd(&A.values[k+4]),
                                      gather4(&A.col_indices[k+4], x), vs1);
            }
            __m256d vsT = _mm256_add_pd(vs0, vs1);
            for (; k + 3 < end_nz; k += 4)
                vsT = _mm256_fmadd_pd(_mm256_loadu_pd(&A.values[k]),
                                      gather4(&A.col_indices[k], x), vsT);
            __m128d lo  = _mm256_castpd256_pd128(vsT);
            __m128d hi  = _mm256_extractf128_pd(vsT, 1);
            __m128d s2  = _mm_add_pd(lo, hi);
            double  sum = _mm_cvtsd_f64(_mm_hadd_pd(s2, s2));
            for (; k < end_nz; ++k)
                sum += A.values[k] * x[A.col_indices[k]];
            y[row_s] = sum;
            nz_s = end_nz;
        }
        double carry = 0.0;
        for (; nz_s < nz_e; ++nz_s)
            carry += A.values[nz_s] * x[A.col_indices[nz_s]];
        row_carry_out[tid] = row_e;
        val_carry_out[tid] = carry;
    }
    for (int tid = 0; tid < num_threads - 1; ++tid)
        if (row_carry_out[tid] < A.num_rows)
            y[row_carry_out[tid]] += val_carry_out[tid];
}

/* =========================================================================
 * Correctness check  (relative L-inf norm)
 * ========================================================================= */
bool Verify(const double* ref, const double* y, int n, double tol = 1e-6)
{
    for (int i = 0; i < n; ++i) {
        double diff = std::fabs(ref[i] - y[i]);
        double mag  = std::max(std::fabs(ref[i]), 1.0);
        if (diff / mag > tol) {
            fprintf(stderr, "  MISMATCH row %d: ref=%.15e got=%.15e\n", i, ref[i], y[i]);
            return false;
        }
    }
    return true;
}

/* =========================================================================
 * Benchmark runner: warmup + timed loop
 * ========================================================================= */
template <typename Fn>
double Benchmark(Fn fn, int warmup, int iters)
{
    for (int i = 0; i < warmup; ++i) fn();
    Timer t; t.start();
    for (int i = 0; i < iters; ++i) fn();
    return t.elapsedMs() / iters;
}

/* =========================================================================
 * Compute worst-case bytes transferred per SpMV
 * ========================================================================= */
static double BytesPerIter(const CsrMatrix& A)
{
    return (double)A.num_nonzeros * (8.0 + 4.0 + 8.0)   // values, col, x
         + (double)(A.num_rows + 1) * 4.0                 // row_offsets
         + (double)A.num_rows * 8.0;                      // y write
}

/* =========================================================================
 * Result row structure
 * ========================================================================= */
struct Row {
    std::string label;
    double time_ms;
    double gflops;
    double bw_gbs;
    double speedup;    // relative to Approach 0 (author baseline)
    bool   correct;
};

/* =========================================================================
 * Print ASCII table
 * ========================================================================= */
static void PrintTable(const std::vector<Row>& rows,
                       const std::string& mname, int threads)
{
    printf("\nMatrix: %s   (threads=%d)\n", mname.c_str(), threads);
    printf("  NOTE: Speedup is relative to Approach 0 (Author's Baseline)\n");
    printf("%-48s  %9s  %9s  %10s  %9s  %7s\n",
           "Approach", "Time(ms)", "GFLOPs/s", "BW(GB/s)", "Speedup", "Correct");
    printf("%s\n", std::string(104, '-').c_str());
    for (auto& r : rows)
        printf("%-48s  %9.4f  %9.4f  %10.3f  %9.3fx  %7s\n",
               r.label.c_str(), r.time_ms, r.gflops, r.bw_gbs,
               r.speedup, r.correct ? "PASS" : "FAIL");
    printf("\n");
}

/* =========================================================================
 * Write CSV  (always writes header; overwrites file each run)
 * ========================================================================= */
static void WriteCsv(const std::vector<Row>& rows,
                     const std::string& mname, int threads,
                     const std::string& path)
{
    FILE* f = fopen(path.c_str(), "w");
    if (!f) { fprintf(stderr, "Cannot write %s\n", path.c_str()); return; }
    fprintf(f, "matrix,threads,approach,time_ms,gflops,bw_gbs,speedup_vs_author,correct\n");
    for (auto& r : rows)
        fprintf(f, "%s,%d,%s,%.6f,%.6f,%.4f,%.4f,%s\n",
                mname.c_str(), threads, r.label.c_str(),
                r.time_ms, r.gflops, r.bw_gbs, r.speedup,
                r.correct ? "PASS" : "FAIL");
    fclose(f);
}

/* =========================================================================
 * Main
 * ========================================================================= */
int main(int argc, char** argv)
{
    std::string mtx_file;
    int  num_threads  = omp_get_max_threads();
    int  timing_iters = 100;
    int  warmup_iters = 5;
    int  pf_dist      = 32;
    std::string csv_out = "results.csv";

    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        auto eq = arg.find('=');
        if (eq == std::string::npos) continue;
        std::string key = arg.substr(2, eq - 2);
        std::string val = arg.substr(eq + 1);
        if (key == "mtx")     mtx_file     = val;
        if (key == "threads") num_threads  = std::stoi(val);
        if (key == "iters")   timing_iters = std::stoi(val);
        if (key == "warmup")  warmup_iters = std::stoi(val);
        if (key == "pfdist")  pf_dist      = std::stoi(val);
        if (key == "csv")     csv_out      = val;
    }

    if (mtx_file.empty()) {
        fprintf(stderr,
            "Usage: %s --mtx=<file.mtx> [--threads=N] [--iters=N]\n"
            "          [--warmup=N] [--pfdist=N (default 32, 0=off)] [--csv=out.csv]\n",
            argv[0]);
        return 1;
    }

    setenv("OMP_PROC_BIND", "close", 0);
    setenv("OMP_PLACES",    "cores", 0);

    printf("=================================================================\n");
    printf(" SpMV Benchmark  –  Merge-Path (Merrill & Garland SC'16)\n");
    printf("=================================================================\n");
    printf("  Matrix file  : %s\n", mtx_file.c_str());
    printf("  OMP threads  : %d\n", num_threads);
    printf("  Timing iters : %d  (warmup %d)\n", timing_iters, warmup_iters);
    printf("  Prefetch dist: %d nonzeros%s\n",
           pf_dist, pf_dist > 0 ? "" : "  (DISABLED)");

    CsrMatrix A;
    if (!LoadMatrixMarket(mtx_file, A)) return 1;

    long long nnz  = A.num_nonzeros;
    long long nrow = A.num_rows;
    double gflops_per_iter = 2.0 * nnz / 1e9;
    double bytes_per_iter  = BytesPerIter(A);
    double ai_flop_per_byte = (2.0 * nnz) / bytes_per_iter;

    printf("  Rows         : %lld\n", nrow);
    printf("  NNZ          : %lld\n", nnz);
    printf("  Avg nnz/row  : %.1f\n", (double)nnz / nrow);
    printf("  Data/iter    : %.1f MB (worst-case)\n", bytes_per_iter / 1e6);
    printf("  Arith. Intens: %.4f FLOP/byte (roofline)\n", ai_flop_per_byte);
    printf("=================================================================\n\n");

    double* x        = (double*)aligned_new(sizeof(double) * A.num_cols);
    double* yref     = (double*)aligned_new(sizeof(double) * A.num_rows);
    double* y        = (double*)aligned_new(sizeof(double) * A.num_rows);

    for (int i = 0; i < A.num_cols; ++i) x[i] = 1.0 / (i + 1.0);

    const int MAX_T = 1024;
    int*    row_carry = (int*)   aligned_new(sizeof(int)    * MAX_T);
    double* val_carry = (double*)aligned_new(sizeof(double) * MAX_T);

    // Reference answer from sequential (not the baseline, just for verification)
    SpMV_Sequential(A, x, yref);

    std::string mname = mtx_file;
    {
        auto s = mname.rfind('/');
        if (s != std::string::npos) mname = mname.substr(s + 1);
        auto d = mname.rfind('.');
        if (d != std::string::npos) mname = mname.substr(0, d);
    }

    std::vector<Row> results;
    double author_baseline_ms = 0.0;  // <-- all speedups relative to this

    auto make_row = [&](const std::string& lbl, double ms, bool ok) -> Row {
        double gf  = gflops_per_iter / (ms * 1e-3);
        double bw  = bytes_per_iter  / (ms * 1e-3) / 1e9;
        double spd = (author_baseline_ms > 0) ? author_baseline_ms / ms : 1.0;
        return Row{lbl, ms, gf, bw, spd, ok};
    };

    // ---- Approach 0: Author's Original OmpMergeCsrmv (THE BASELINE) ------
    {
        auto fn = [&]{
            SpMV_Author_Baseline(A, x, y, num_threads, row_carry, val_carry);
        };
        double ms = Benchmark(fn, warmup_iters, timing_iters);
        author_baseline_ms = ms;          // anchor for all speedup calculations
        bool ok = Verify(yref, y, A.num_rows);
        results.push_back(make_row("0. Author Baseline (OmpMergeCsrmv)", ms, ok));
    }

    // ---- Approach 1: Sequential ------------------------------------------
    {
        auto fn = [&]{ SpMV_Sequential(A, x, y); };
        double ms = Benchmark(fn, warmup_iters, timing_iters);
        bool ok = Verify(yref, y, A.num_rows);
        results.push_back(make_row("1. Sequential Baseline", ms, ok));
    }

    // ---- Approach 2: OMP Merge-Path Scalar (Paper Alg. 2) ---------------
    {
        auto fn = [&]{
            SpMV_OMP_MergePath_Scalar(A, x, y, num_threads, row_carry, val_carry);
        };
        double ms = Benchmark(fn, warmup_iters, timing_iters);
        bool ok = Verify(yref, y, A.num_rows);
        results.push_back(make_row("2. OMP Merge-Path Scalar", ms, ok));
    }

    // ---- Approach 3: OMP Merge-Path AVX2 (4-wide gather + FMA) ----------
    {
        auto fn = [&]{
            SpMV_OMP_MergePath_AVX2(A, x, y, num_threads, row_carry, val_carry);
        };
        double ms = Benchmark(fn, warmup_iters, timing_iters);
        bool ok = Verify(yref, y, A.num_rows);
        results.push_back(make_row("3. OMP Merge-Path AVX2", ms, ok));
    }

    // ---- Approach 4: OMP Merge-Path AVX2 ×2 Unrolled (8-wide) ----------
    {
        auto fn = [&]{
            SpMV_OMP_MergePath_AVX2_Unroll2(A, x, y, num_threads, row_carry, val_carry);
        };
        double ms = Benchmark(fn, warmup_iters, timing_iters);
        bool ok = Verify(yref, y, A.num_rows);
        results.push_back(make_row("4. OMP AVX2 Unrolled×2", ms, ok));
    }

    // ---- Approach 5: OMP Merge-Path AVX2 + Prefetch (tunable) ----------
    {
        auto fn = [&]{
            SpMV_OMP_MergePath_AVX2_Prefetch(A, x, y, num_threads,
                                              row_carry, val_carry, pf_dist);
        };
        double ms = Benchmark(fn, warmup_iters, timing_iters);
        bool ok = Verify(yref, y, A.num_rows);
        char lbl[64];
        snprintf(lbl, sizeof(lbl), "5. OMP AVX2+Prefetch(d=%d)", pf_dist);
        results.push_back(make_row(lbl, ms, ok));
    }

    PrintTable(results, mname, num_threads);
    WriteCsv(results, mname, num_threads, csv_out);
    printf("  Results appended to: %s\n\n", csv_out.c_str());

    aligned_del(x);
    aligned_del(yref);
    aligned_del(y);
    aligned_del(row_carry);
    aligned_del(val_carry);
    return 0;
}