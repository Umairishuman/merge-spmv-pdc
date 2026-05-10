/*******************************************************************************
 * spmv_benchmark.cpp
 *
 * Sparse Matrix-Vector Multiplication Benchmark
 * CS-3006 · Parallel & Distributed Computing · Spring 2026
 * Group: Muhammad Umair (23i-0662) · Awais Basheer (23i-0506)
 *
 * Based on: "Merge-based Parallel Sparse Matrix-Vector Multiplication"
 *           Merrill & Garland, SC'16
 *
 * Implements 5 SpMV approaches:
 *   1. Sequential Baseline        – standard CSR scalar
 *   2. OpenMP Merge-Path Scalar   – paper Algorithm 2, scalar inner loop
 *   3. OpenMP Merge-Path AVX2     – vectorised inner loop (256-bit FMA)
 *   4. OpenMP Merge-Path AVX2 ×2  – loop unrolled ×2 (8 doubles / iter)
 *   5. OpenMP Merge-Path AVX2+PF  – approach 4 + software prefetch
 *
 * Build:
 *   make            (see Makefile)
 *   ./spmv_benchmark --mtx=<file.mtx> [--threads=<N>] [--iters=<N>]
 *
 * No MKL dependency – uses only standard C++17 + OpenMP + AVX2/FMA.
 ******************************************************************************/

#include <omp.h>
#include <immintrin.h>   // AVX2 / FMA intrinsics

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>

/* =========================================================================
 * Platform-portable aligned allocation helpers
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
    int     num_rows    = 0;
    int     num_cols    = 0;
    int     num_nonzeros= 0;
    int*    row_offsets = nullptr;   // size num_rows+1
    int*    col_indices = nullptr;   // size num_nonzeros
    double* values      = nullptr;   // size num_nonzeros

    CsrMatrix() = default;
    ~CsrMatrix() {
        aligned_del(row_offsets);
        aligned_del(col_indices);
        aligned_del(values);
    }
    // non-copyable
    CsrMatrix(const CsrMatrix&) = delete;
    CsrMatrix& operator=(const CsrMatrix&) = delete;
};

/* =========================================================================
 * Matrix Market reader  (handles real / pattern / symmetric)
 * ========================================================================= */
bool LoadMatrixMarket(const std::string& path, CsrMatrix& csr)
{
    std::ifstream fin(path);
    if (!fin.is_open()) {
        fprintf(stderr, "ERROR: cannot open '%s'\n", path.c_str());
        return false;
    }

    // --- header ---
    std::string line;
    std::getline(fin, line);
    bool is_pattern   = (line.find("pattern")   != std::string::npos);
    bool is_symmetric = (line.find("symmetric") != std::string::npos ||
                         line.find("hermitian") != std::string::npos);

    // skip comments
    while (fin.peek() == '%') std::getline(fin, line);

    int M, N, NNZ;
    fin >> M >> N >> NNZ;

    // Read COO triples
    struct Triple { int r, c; double v; };
    std::vector<Triple> coo;
    coo.reserve(is_symmetric ? NNZ * 2 : NNZ);

    for (int k = 0; k < NNZ; ++k) {
        int r, c; double v = 1.0;
        fin >> r >> c;
        if (!is_pattern) fin >> v;
        --r; --c;  // 0-based
        coo.push_back({r, c, v});
        if (is_symmetric && r != c)
            coo.push_back({c, r, v});
    }
    fin.close();

    // Sort COO row-major
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
 * Timer
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
 * Counting iterator (for MergePath)
 * ========================================================================= */
struct CountIter {
    int val;
    explicit CountIter(int v) : val(v) {}
    int operator[](int i) const { return val + i; }
};

/* =========================================================================
 * MergePath binary search
 *   Finds the (row_idx, nz_idx) coordinate where diagonal d intersects
 *   the merge path of (row_end_offsets[], 0..nnz-1).
 * ========================================================================= */
static inline void MergePathSearch(
    int diagonal,
    const int* row_end_offsets,   // A  – length num_rows
    int        num_rows,
    int        num_nonzeros,
    int&       row_idx,
    int&       nz_idx)
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
 * Approach 1 – Sequential baseline
 * ========================================================================= */
void SpMV_Sequential(const CsrMatrix& A, const double* __restrict__ x,
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
 * Approach 2 – OpenMP Merge-Path Scalar  (Paper Algorithm 2)
 * ========================================================================= */
void SpMV_OMP_MergePath_Scalar(
    const CsrMatrix& A,
    const double* __restrict__ x,
    double* __restrict__ y,
    int num_threads,
    int* row_carry_out,
    double* val_carry_out)
{
    const int* row_end = A.row_offsets + 1;   // A-list: row end offsets
    const int num_merge = A.num_rows + A.num_nonzeros;

    #pragma omp parallel for schedule(static) num_threads(num_threads)
    for (int tid = 0; tid < num_threads; ++tid)
    {
        int items_per_thread = (num_merge + num_threads - 1) / num_threads;
        int start_diag = std::min(items_per_thread * tid,       num_merge);
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

    // Fix-up
    for (int tid = 0; tid < num_threads - 1; ++tid)
        if (row_carry_out[tid] < A.num_rows)
            y[row_carry_out[tid]] += val_carry_out[tid];
}

/* =========================================================================
 * AVX2 helper: gather 4 doubles from x[] using 4 int32 indices
 *   col_ptr  – pointer to 4 consecutive int32 column indices
 *   x        – base pointer of the dense vector
 * Returns __m256d with x[col[0..3]]
 * ========================================================================= */
static inline __m256d gather4(const int* col_ptr, const double* x)
{
    // Load 4 int32 indices
    __m128i vindex = _mm_loadu_si128(reinterpret_cast<const __m128i*>(col_ptr));
    // Gather 4 double-precision values
    return _mm256_i32gather_pd(x, vindex, 8 /*scale = sizeof(double)*/);
}

/* =========================================================================
 * Approach 3 – OpenMP Merge-Path AVX2 (vectorised inner loop, 4 doubles)
 * ========================================================================= */
void SpMV_OMP_MergePath_AVX2(
    const CsrMatrix& A,
    const double* __restrict__ x,
    double* __restrict__ y,
    int num_threads,
    int* row_carry_out,
    double* val_carry_out)
{
    const int* row_end   = A.row_offsets + 1;
    const int num_merge  = A.num_rows + A.num_nonzeros;

    #pragma omp parallel for schedule(static) num_threads(num_threads)
    for (int tid = 0; tid < num_threads; ++tid)
    {
        int items_per_thread = (num_merge + num_threads - 1) / num_threads;
        int start_diag = std::min(items_per_thread * tid,        num_merge);
        int end_diag   = std::min(start_diag + items_per_thread, num_merge);

        int row_s, nz_s, row_e, nz_e;
        MergePathSearch(start_diag, row_end, A.num_rows, A.num_nonzeros, row_s, nz_s);
        MergePathSearch(end_diag,   row_end, A.num_rows, A.num_nonzeros, row_e, nz_e);

        for (; row_s < row_e; ++row_s)
        {
            int row_end_nz = row_end[row_s];
            __m256d vsum = _mm256_setzero_pd();

            // Vectorised strip (4 doubles per iteration)
            int k = nz_s;
            for (; k + 3 < row_end_nz; k += 4) {
                __m256d vval = _mm256_loadu_pd(&A.values[k]);
                __m256d vx   = gather4(&A.col_indices[k], x);
                vsum = _mm256_fmadd_pd(vval, vx, vsum);
            }
            // Horizontal reduce
            __m128d lo  = _mm256_castpd256_pd128(vsum);
            __m128d hi  = _mm256_extractf128_pd(vsum, 1);
            __m128d sum2 = _mm_add_pd(lo, hi);
            double sum  = _mm_cvtsd_f64(_mm_hadd_pd(sum2, sum2));

            // Scalar tail
            for (; k < row_end_nz; ++k)
                sum += A.values[k] * x[A.col_indices[k]];

            y[row_s] = sum;
            nz_s = row_end_nz;
        }

        // Partial last row
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
 * ========================================================================= */
void SpMV_OMP_MergePath_AVX2_Unroll2(
    const CsrMatrix& A,
    const double* __restrict__ x,
    double* __restrict__ y,
    int num_threads,
    int* row_carry_out,
    double* val_carry_out)
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

        for (; row_s < row_e; ++row_s)
        {
            int row_end_nz = row_end[row_s];
            __m256d vsum0 = _mm256_setzero_pd();
            __m256d vsum1 = _mm256_setzero_pd();

            int k = nz_s;
            // 8 doubles per iteration (2 × 4-wide vectors)
            for (; k + 7 < row_end_nz; k += 8) {
                __m256d vval0 = _mm256_loadu_pd(&A.values[k]);
                __m256d vx0   = gather4(&A.col_indices[k],   x);
                vsum0 = _mm256_fmadd_pd(vval0, vx0, vsum0);

                __m256d vval1 = _mm256_loadu_pd(&A.values[k+4]);
                __m256d vx1   = gather4(&A.col_indices[k+4], x);
                vsum1 = _mm256_fmadd_pd(vval1, vx1, vsum1);
            }
            // Fold the two accumulators
            __m256d vsumT = _mm256_add_pd(vsum0, vsum1);

            // Scalar tail for remainder (< 8 left), try vectorised 4 first
            for (; k + 3 < row_end_nz; k += 4) {
                __m256d vval = _mm256_loadu_pd(&A.values[k]);
                __m256d vx   = gather4(&A.col_indices[k], x);
                vsumT = _mm256_fmadd_pd(vval, vx, vsumT);
            }

            // Horizontal reduce
            __m128d lo   = _mm256_castpd256_pd128(vsumT);
            __m128d hi   = _mm256_extractf128_pd(vsumT, 1);
            __m128d sum2 = _mm_add_pd(lo, hi);
            double  sum  = _mm_cvtsd_f64(_mm_hadd_pd(sum2, sum2));

            for (; k < row_end_nz; ++k)
                sum += A.values[k] * x[A.col_indices[k]];

            y[row_s] = sum;
            nz_s = row_end_nz;
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
 * Approach 5 – OpenMP Merge-Path AVX2 Unrolled + Software Prefetch
 *   Same as Approach 4, plus __builtin_prefetch on values[] and x[] ahead.
 *   Prefetch distance PF_DIST chosen experimentally; 32 cache lines ≈ 256B.
 * ========================================================================= */
static constexpr int PF_DIST = 16;   // nonzero look-ahead in elements

void SpMV_OMP_MergePath_AVX2_Prefetch(
    const CsrMatrix& A,
    const double* __restrict__ x,
    double* __restrict__ y,
    int num_threads,
    int* row_carry_out,
    double* val_carry_out)
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

        for (; row_s < row_e; ++row_s)
        {
            int row_end_nz = row_end[row_s];
            __m256d vsum0 = _mm256_setzero_pd();
            __m256d vsum1 = _mm256_setzero_pd();

            int k = nz_s;
            for (; k + 7 < row_end_nz; k += 8) {
                // Prefetch future values[] and col_indices[] into L1
                __builtin_prefetch(&A.values[k + PF_DIST],      0, 1);
                __builtin_prefetch(&A.col_indices[k + PF_DIST], 0, 1);
                // Prefetch future x[] – best-effort (irregular access pattern)
                if (k + PF_DIST < A.num_nonzeros) {
                    __builtin_prefetch(&x[A.col_indices[k + PF_DIST]],   0, 1);
                    __builtin_prefetch(&x[A.col_indices[k + PF_DIST + 4]], 0, 1);
                }

                __m256d vval0 = _mm256_loadu_pd(&A.values[k]);
                __m256d vx0   = gather4(&A.col_indices[k],   x);
                vsum0 = _mm256_fmadd_pd(vval0, vx0, vsum0);

                __m256d vval1 = _mm256_loadu_pd(&A.values[k+4]);
                __m256d vx1   = gather4(&A.col_indices[k+4], x);
                vsum1 = _mm256_fmadd_pd(vval1, vx1, vsum1);
            }
            __m256d vsumT = _mm256_add_pd(vsum0, vsum1);

            for (; k + 3 < row_end_nz; k += 4) {
                __m256d vval = _mm256_loadu_pd(&A.values[k]);
                __m256d vx   = gather4(&A.col_indices[k], x);
                vsumT = _mm256_fmadd_pd(vval, vx, vsumT);
            }

            __m128d lo   = _mm256_castpd256_pd128(vsumT);
            __m128d hi   = _mm256_extractf128_pd(vsumT, 1);
            __m128d sum2 = _mm_add_pd(lo, hi);
            double  sum  = _mm_cvtsd_f64(_mm_hadd_pd(sum2, sum2));

            for (; k < row_end_nz; ++k)
                sum += A.values[k] * x[A.col_indices[k]];

            y[row_s] = sum;
            nz_s = row_end_nz;
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
 * Correctness check: compare y against reference (L-inf norm)
 * ========================================================================= */
bool Verify(const double* ref, const double* y, int n, double tol = 1e-6)
{
    for (int i = 0; i < n; ++i) {
        double diff = std::fabs(ref[i] - y[i]);
        double mag  = std::max(std::fabs(ref[i]), 1.0);
        if (diff / mag > tol) {
            fprintf(stderr, "  MISMATCH at row %d: ref=%.15e got=%.15e\n",
                    i, ref[i], y[i]);
            return false;
        }
    }
    return true;
}

/* =========================================================================
 * Benchmark runner: warmup + timed iterations → returns avg ms
 * ========================================================================= */
template <typename Fn>
double Benchmark(Fn fn, int warmup, int iters)
{
    for (int i = 0; i < warmup; ++i) fn();
    Timer t;
    t.start();
    for (int i = 0; i < iters; ++i) fn();
    return t.elapsedMs() / iters;
}

/* =========================================================================
 * CSV logger
 * ========================================================================= */
struct CsvRow {
    std::string approach;
    double time_ms;
    double gflops;
    double speedup;
    bool   correct;
};

static void PrintTable(const std::vector<CsvRow>& rows, const std::string& matrix_name)
{
    printf("\n");
    printf("Matrix: %s\n", matrix_name.c_str());
    printf("%-42s  %10s  %10s  %10s  %8s\n",
           "Approach", "Time(ms)", "GFLOPs/s", "Speedup", "Correct");
    printf("%s\n", std::string(88, '-').c_str());
    for (auto& r : rows) {
        printf("%-42s  %10.4f  %10.4f  %10.3fx  %8s\n",
               r.approach.c_str(), r.time_ms, r.gflops, r.speedup,
               r.correct ? "PASS" : "FAIL");
    }
    printf("\n");
}

static void WriteCsv(const std::vector<CsvRow>& rows,
                     const std::string& matrix_name,
                     const std::string& csv_path)
{
    // append mode – one file collects all matrices
    FILE* f = fopen(csv_path.c_str(), "a");
    if (!f) { fprintf(stderr, "Cannot write %s\n", csv_path.c_str()); return; }
    // header on first write
    static bool header_written = false;
    if (!header_written) {
        fprintf(f, "matrix,approach,time_ms,gflops,speedup,correct\n");
        header_written = true;
    }
    for (auto& r : rows)
        fprintf(f, "%s,%s,%.6f,%.6f,%.4f,%s\n",
                matrix_name.c_str(), r.approach.c_str(),
                r.time_ms, r.gflops, r.speedup,
                r.correct ? "PASS" : "FAIL");
    fclose(f);
}

/* =========================================================================
 * Main
 * ========================================================================= */
int main(int argc, char** argv)
{
    // --- parse arguments ----
    std::string mtx_file;
    int  num_threads    = omp_get_max_threads();
    int  timing_iters   = 100;
    int  warmup_iters   = 5;
    std::string csv_out = "results.csv";

    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        auto eq = arg.find('=');
        if (eq == std::string::npos) continue;
        std::string key = arg.substr(2, eq - 2);   // strip leading "--"
        std::string val = arg.substr(eq + 1);
        if (key == "mtx")     mtx_file     = val;
        if (key == "threads") num_threads  = std::stoi(val);
        if (key == "iters")   timing_iters = std::stoi(val);
        if (key == "warmup")  warmup_iters = std::stoi(val);
        if (key == "csv")     csv_out      = val;
    }

    if (mtx_file.empty()) {
        fprintf(stderr,
            "Usage: %s --mtx=<file.mtx> [--threads=N] [--iters=N] "
            "[--warmup=N] [--csv=out.csv]\n", argv[0]);
        return 1;
    }

    printf("=================================================================\n");
    printf(" SpMV Benchmark  –  Merge-Path (Merrill & Garland SC'16)\n");
    printf("=================================================================\n");
    printf("  Matrix file : %s\n", mtx_file.c_str());
    printf("  OMP threads : %d\n", num_threads);
    printf("  Timing iters: %d  (warmup %d)\n", timing_iters, warmup_iters);

    // --- load matrix ---
    CsrMatrix A;
    if (!LoadMatrixMarket(mtx_file, A)) return 1;

    long long nnz  = A.num_nonzeros;
    long long rows = A.num_rows;
    double gflops_per_iter = 2.0 * nnz / 1e9;   // 2 FLOPs per nonzero (mul+add)

    printf("  Rows        : %lld\n", rows);
    printf("  NNZ         : %lld\n", nnz);
    printf("  Avg nnz/row : %.1f\n", (double)nnz / rows);
    printf("=================================================================\n\n");

    // --- allocate vectors ---
    double* x   = (double*)aligned_new(sizeof(double) * A.num_cols);
    double* yref= (double*)aligned_new(sizeof(double) * A.num_rows);
    double* y   = (double*)aligned_new(sizeof(double) * A.num_rows);

    for (int i = 0; i < A.num_cols; ++i) x[i] = 1.0 / (i + 1.0);

    // carry-out scratch (max 1024 threads)
    const int MAX_T = 1024;
    int*    row_carry = (int*)   aligned_new(sizeof(int)    * MAX_T);
    double* val_carry = (double*)aligned_new(sizeof(double) * MAX_T);

    // --- reference ---
    SpMV_Sequential(A, x, yref);

    // ---- matrix name (strip path and extension) ----
    std::string mname = mtx_file;
    auto slash = mname.rfind('/');
    if (slash != std::string::npos) mname = mname.substr(slash+1);
    auto dot = mname.rfind('.');
    if (dot != std::string::npos) mname = mname.substr(0, dot);

    std::vector<CsvRow> rows_out;
    double base_ms = 0.0;

    // ---- Approach 1: Sequential -------------------------------------------
    {
        auto fn = [&]{ SpMV_Sequential(A, x, y); };
        double ms = Benchmark(fn, warmup_iters, timing_iters);
        bool ok = Verify(yref, y, A.num_rows);
        base_ms = ms;
        rows_out.push_back({"1. Sequential Baseline", ms,
            gflops_per_iter / (ms * 1e-3), 1.0, ok});
    }

    // ---- Approach 2: OMP Merge-Path Scalar --------------------------------
    {
        auto fn = [&]{
            SpMV_OMP_MergePath_Scalar(A, x, y, num_threads, row_carry, val_carry);
        };
        double ms = Benchmark(fn, warmup_iters, timing_iters);
        bool ok = Verify(yref, y, A.num_rows);
        rows_out.push_back({"2. OMP Merge-Path Scalar", ms,
            gflops_per_iter / (ms * 1e-3), base_ms / ms, ok});
    }

    // ---- Approach 3: OMP Merge-Path AVX2 ----------------------------------
    {
        auto fn = [&]{
            SpMV_OMP_MergePath_AVX2(A, x, y, num_threads, row_carry, val_carry);
        };
        double ms = Benchmark(fn, warmup_iters, timing_iters);
        bool ok = Verify(yref, y, A.num_rows);
        rows_out.push_back({"3. OMP Merge-Path AVX2", ms,
            gflops_per_iter / (ms * 1e-3), base_ms / ms, ok});
    }

    // ---- Approach 4: OMP Merge-Path AVX2 ×2 Unrolled ----------------------
    {
        auto fn = [&]{
            SpMV_OMP_MergePath_AVX2_Unroll2(A, x, y, num_threads, row_carry, val_carry);
        };
        double ms = Benchmark(fn, warmup_iters, timing_iters);
        bool ok = Verify(yref, y, A.num_rows);
        rows_out.push_back({"4. OMP Merge-Path AVX2 Unrolled×2", ms,
            gflops_per_iter / (ms * 1e-3), base_ms / ms, ok});
    }

    // ---- Approach 5: OMP Merge-Path AVX2 + Prefetch -----------------------
    {
        auto fn = [&]{
            SpMV_OMP_MergePath_AVX2_Prefetch(A, x, y, num_threads, row_carry, val_carry);
        };
        double ms = Benchmark(fn, warmup_iters, timing_iters);
        bool ok = Verify(yref, y, A.num_rows);
        rows_out.push_back({"5. OMP Merge-Path AVX2 Unrolled+Prefetch", ms,
            gflops_per_iter / (ms * 1e-3), base_ms / ms, ok});
    }

    // ---- Print ASCII table + write CSV ------------------------------------
    PrintTable(rows_out, mname);
    WriteCsv(rows_out, mname, csv_out);
    printf("  Results appended to: %s\n\n", csv_out.c_str());

    // --- cleanup ---
    aligned_del(x);
    aligned_del(yref);
    aligned_del(y);
    aligned_del(row_carry);
    aligned_del(val_carry);

    return 0;
}
