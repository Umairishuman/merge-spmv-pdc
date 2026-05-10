/******************************************************************************
 * Copyright (c) 2011-2016, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

//---------------------------------------------------------------------
// SpMV comparison tool — fixed for CUDA 11/12
// Changes from original:
//   1. Removed #include <cub/iterator/tex_ref_input_iterator.cuh>
//      (texture references removed in CUDA 12)
//   2. Replaced cusparseScsrmv / cusparseDcsrmv with cusparseSpMV
//      generic API (old routines removed in CUDA 11)
//   3. Replaced cusparseHYB functions (HybMV) with no-op stubs
//      (HYB format entirely removed in CUDA 11)
//   4. Added const_cast on cudaMemcpy source pointers
//      (CUDA 12 tightened const-correctness rules)
//   5. Added const_cast on DeviceFree pointers (same reason)
//---------------------------------------------------------------------

#include <stdio.h>
#include <map>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <fstream>

#include <cusparse.h>

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <cub/device/device_spmv.cuh>
#include <cub/util_allocator.cuh>
// REMOVED: #include <cub/iterator/tex_ref_input_iterator.cuh>
// tex_ref_input_iterator uses texture<T> and cudaBindTexture,
// both removed in CUDA 12. Not needed for merge-based SpMV.

#include "sparse_matrix.h"
#include <utils.h>

using namespace cub;


//---------------------------------------------------------------------
// Globals, constants, and type declarations
//---------------------------------------------------------------------

bool                    g_quiet     = false;
bool                    g_verbose   = false;
bool                    g_verbose2  = false;
CachingDeviceAllocator  g_allocator(true);


//---------------------------------------------------------------------
// SpMV verification
//---------------------------------------------------------------------

template <typename ValueT, typename OffsetT>
void SpmvGold(
    CsrMatrix<ValueT, OffsetT>&     a,
    ValueT*                         vector_x,
    ValueT*                         vector_y_in,
    ValueT*                         vector_y_out,
    ValueT                          alpha,
    ValueT                          beta)
{
    for (OffsetT row = 0; row < a.num_rows; ++row)
    {
        ValueT partial = beta * vector_y_in[row];
        for (OffsetT offset = a.row_offsets[row];
             offset < a.row_offsets[row + 1];
             ++offset)
        {
            partial += alpha * a.values[offset] * vector_x[a.column_indices[offset]];
        }
        vector_y_out[row] = partial;
    }
}


//---------------------------------------------------------------------
// cuSPARSE HybMV — STUBBED OUT
// cusparseHybMat_t, cusparseCreateHybMat, cusparseDcsr2hyb,
// cusparseShybmv, cusparseDhybmv, cusparseDestroyHybMat were all
// removed in CUDA 11. HybMV results are replaced with zeros so the
// rest of the benchmark still runs and reports merge-based results.
//---------------------------------------------------------------------

template <typename OffsetT>
float TestCusparseHybmv(
    float*                          vector_y_in,
    float*                          reference_vector_y_out,
    SpmvParams<float, OffsetT>&     params,
    int                             timing_iterations,
    float&                          setup_ms,
    cusparseHandle_t                cusparse)
{
    setup_ms = 0.0;
    if (!g_quiet)
        printf("\t[SKIPPED — HybMV removed in CUDA 11+]\n");
    fflush(stdout);
    return 0.0f;
}

template <typename OffsetT>
float TestCusparseHybmv(
    double*                         vector_y_in,
    double*                         reference_vector_y_out,
    SpmvParams<double, OffsetT>&    params,
    int                             timing_iterations,
    float&                          setup_ms,
    cusparseHandle_t                cusparse)
{
    setup_ms = 0.0;
    if (!g_quiet)
        printf("\t[SKIPPED — HybMV removed in CUDA 11+]\n");
    fflush(stdout);
    return 0.0f;
}


//---------------------------------------------------------------------
// cuSPARSE CsrMV — updated to generic cusparseSpMV API
// cusparseScsrmv and cusparseDcsrmv removed in CUDA 11.
// Replaced with cusparseSpMV (generic API, CUDA 10.2+).
//---------------------------------------------------------------------

// Helper: run cusparseSpMV for float
template <typename OffsetT>
cusparseStatus_t RunCusparseSpMV_fp32(
    cusparseHandle_t                cusparse,
    SpmvParams<float, OffsetT>&     params,
    float*                          alpha_ptr,
    float*                          beta_ptr)
{
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;

    cusparseCreateCsr(
        &matA,
        params.num_rows, params.num_cols, params.num_nonzeros,
        (void*)params.d_row_end_offsets,
        (void*)params.d_column_indices,
        (void*)params.d_values,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

    cusparseCreateDnVec(&vecX, params.num_cols, (void*)params.d_vector_x, CUDA_R_32F);
    cusparseCreateDnVec(&vecY, params.num_rows, (void*)params.d_vector_y, CUDA_R_32F);

    size_t bufSize = 0;
    cusparseSpMV_bufferSize(
        cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
        alpha_ptr, matA, vecX, beta_ptr, vecY,
        CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bufSize);

    void* buf = nullptr;
    if (bufSize > 0) cudaMalloc(&buf, bufSize);

    cusparseStatus_t st = cusparseSpMV(
        cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
        alpha_ptr, matA, vecX, beta_ptr, vecY,
        CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, buf);

    if (buf) cudaFree(buf);
    cusparseDestroySpMat(matA);
    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecY);
    return st;
}

// Helper: run cusparseSpMV for double
template <typename OffsetT>
cusparseStatus_t RunCusparseSpMV_fp64(
    cusparseHandle_t                cusparse,
    SpmvParams<double, OffsetT>&    params,
    double*                         alpha_ptr,
    double*                         beta_ptr)
{
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;

    cusparseCreateCsr(
        &matA,
        params.num_rows, params.num_cols, params.num_nonzeros,
        (void*)params.d_row_end_offsets,
        (void*)params.d_column_indices,
        (void*)params.d_values,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

    cusparseCreateDnVec(&vecX, params.num_cols, (void*)params.d_vector_x, CUDA_R_64F);
    cusparseCreateDnVec(&vecY, params.num_rows, (void*)params.d_vector_y, CUDA_R_64F);

    size_t bufSize = 0;
    cusparseSpMV_bufferSize(
        cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
        alpha_ptr, matA, vecX, beta_ptr, vecY,
        CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufSize);

    void* buf = nullptr;
    if (bufSize > 0) cudaMalloc(&buf, bufSize);

    cusparseStatus_t st = cusparseSpMV(
        cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
        alpha_ptr, matA, vecX, beta_ptr, vecY,
        CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, buf);

    if (buf) cudaFree(buf);
    cusparseDestroySpMat(matA);
    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecY);
    return st;
}


/**
 * Run cuSparse CsrMV (fp32)
 */
template <typename OffsetT>
float TestCusparseCsrmv(
    float*                          vector_y_in,
    float*                          reference_vector_y_out,
    SpmvParams<float, OffsetT>&     params,
    int                             timing_iterations,
    float&                          setup_ms,
    cusparseHandle_t                cusparse)
{
    setup_ms = 0.0;

    // Reset output vector
    CubDebugExit(cudaMemcpy(
        params.d_vector_y, vector_y_in,
        sizeof(float) * params.num_rows, cudaMemcpyHostToDevice));

    // Warmup
    AssertEquals(CUSPARSE_STATUS_SUCCESS,
        RunCusparseSpMV_fp32(cusparse, params, &params.alpha, &params.beta));

    if (!g_quiet)
    {
        int compare = CompareDeviceResults(
            reference_vector_y_out, params.d_vector_y,
            params.num_rows, true, g_verbose);
        printf("\t%s\n", compare ? "FAIL" : "PASS");
        fflush(stdout);
    }

    // Timing
    float elapsed_ms = 0.0;
    GpuTimer timer;
    timer.Start();
    for (int it = 0; it < timing_iterations; ++it)
    {
        AssertEquals(CUSPARSE_STATUS_SUCCESS,
            RunCusparseSpMV_fp32(cusparse, params, &params.alpha, &params.beta));
    }
    timer.Stop();
    elapsed_ms += timer.ElapsedMillis();

    return elapsed_ms / timing_iterations;
}


/**
 * Run cuSparse CsrMV (fp64)
 */
template <typename OffsetT>
float TestCusparseCsrmv(
    double*                         vector_y_in,
    double*                         reference_vector_y_out,
    SpmvParams<double, OffsetT>&    params,
    int                             timing_iterations,
    float&                          setup_ms,
    cusparseHandle_t                cusparse)
{
    setup_ms = 0.0;

    // Reset output vector
    CubDebugExit(cudaMemcpy(
        params.d_vector_y, vector_y_in,
        sizeof(double) * params.num_rows, cudaMemcpyHostToDevice));

    // Warmup
    AssertEquals(CUSPARSE_STATUS_SUCCESS,
        RunCusparseSpMV_fp64(cusparse, params, &params.alpha, &params.beta));

    if (!g_quiet)
    {
        int compare = CompareDeviceResults(
            reference_vector_y_out, params.d_vector_y,
            params.num_rows, true, g_verbose);
        printf("\t%s\n", compare ? "FAIL" : "PASS");
        fflush(stdout);
    }

    // Timing
    float elapsed_ms = 0.0;
    GpuTimer timer;
    timer.Start();
    for (int it = 0; it < timing_iterations; ++it)
    {
        AssertEquals(CUSPARSE_STATUS_SUCCESS,
            RunCusparseSpMV_fp64(cusparse, params, &params.alpha, &params.beta));
    }
    timer.Stop();
    elapsed_ms += timer.ElapsedMillis();

    return elapsed_ms / timing_iterations;
}


//---------------------------------------------------------------------
// GPU Merge-based SpMV — unchanged from original
//---------------------------------------------------------------------

template <typename ValueT, typename OffsetT>
float TestGpuMergeCsrmv(
    ValueT*                         vector_y_in,
    ValueT*                         reference_vector_y_out,
    SpmvParams<ValueT, OffsetT>&    params,
    int                             timing_iterations,
    float&                          setup_ms)
{
    setup_ms = 0.0;

    size_t temp_storage_bytes = 0;
    void*  d_temp_storage     = NULL;

    // Get temp storage size
    CubDebugExit(DeviceSpmv::CsrMV(
        d_temp_storage, temp_storage_bytes,
        params.d_values, params.d_row_end_offsets, params.d_column_indices,
        params.d_vector_x, params.d_vector_y,
        params.num_rows, params.num_cols, params.num_nonzeros,
        (cudaStream_t) 0, false));

    // Allocate temp storage
    CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

    // Reset output vector
    CubDebugExit(cudaMemcpy(
        params.d_vector_y, vector_y_in,
        sizeof(ValueT) * params.num_rows, cudaMemcpyHostToDevice));

    // Warmup
    CubDebugExit(DeviceSpmv::CsrMV(
        d_temp_storage, temp_storage_bytes,
        params.d_values, params.d_row_end_offsets, params.d_column_indices,
        params.d_vector_x, params.d_vector_y,
        params.num_rows, params.num_cols, params.num_nonzeros,
        (cudaStream_t) 0, !g_quiet));

    if (!g_quiet)
    {
        int compare = CompareDeviceResults(
            reference_vector_y_out, params.d_vector_y,
            params.num_rows, true, g_verbose);
        printf("\t%s\n", compare ? "FAIL" : "PASS");
        fflush(stdout);
    }

    // Timing
    GpuTimer timer;
    float elapsed_ms = 0.0;
    timer.Start();
    for (int it = 0; it < timing_iterations; ++it)
    {
        CubDebugExit(DeviceSpmv::CsrMV(
            d_temp_storage, temp_storage_bytes,
            params.d_values, params.d_row_end_offsets, params.d_column_indices,
            params.d_vector_x, params.d_vector_y,
            params.num_rows, params.num_cols, params.num_nonzeros,
            (cudaStream_t) 0, false));
    }
    timer.Stop();
    elapsed_ms += timer.ElapsedMillis();

    CubDebugExit(g_allocator.DeviceFree(d_temp_storage));
    return elapsed_ms / timing_iterations;
}


//---------------------------------------------------------------------
// Display perf
//---------------------------------------------------------------------

template <typename ValueT, typename OffsetT>
void DisplayPerf(
    float                           device_giga_bandwidth,
    double                          setup_ms,
    double                          avg_ms,
    CsrMatrix<ValueT, OffsetT>&     csr_matrix)
{
    double nz_throughput, effective_bandwidth;
    size_t total_bytes =
        (csr_matrix.num_nonzeros * (sizeof(ValueT) * 2 + sizeof(OffsetT))) +
        (csr_matrix.num_rows) * (sizeof(OffsetT) + sizeof(ValueT));

    nz_throughput       = double(csr_matrix.num_nonzeros) / avg_ms / 1.0e6;
    effective_bandwidth = double(total_bytes) / avg_ms / 1.0e6;

    if (!g_quiet)
        printf("fp%d: %.4f setup ms, %.4f avg ms, %.5f gflops, %.3lf effective GB/s (%.2f%% peak)\n",
            (int)(sizeof(ValueT) * 8),
            setup_ms, avg_ms,
            2 * nz_throughput,
            effective_bandwidth,
            effective_bandwidth / device_giga_bandwidth * 100);
    else
        printf("%.5f, %.5f, %.6f, %.3lf, ",
            setup_ms, avg_ms,
            2 * nz_throughput,
            effective_bandwidth);

    fflush(stdout);
}


//---------------------------------------------------------------------
// RunTest
//---------------------------------------------------------------------

template <typename ValueT, typename OffsetT>
void RunTest(
    ValueT                      alpha,
    ValueT                      beta,
    CooMatrix<ValueT, OffsetT>& coo_matrix,
    int                         timing_iterations,
    CommandLineArgs&            args)
{
    if (timing_iterations == -1)
        timing_iterations = (int)std::min(
            50000ull,
            std::max(100ull, ((16ull << 30) / coo_matrix.num_nonzeros)));

    if (!g_quiet)
        printf("\t%d timing iterations\n", timing_iterations);

    // Convert to CSR
    CsrMatrix<ValueT, OffsetT> csr_matrix(coo_matrix);
    if (!args.CheckCmdLineFlag("csrmv"))
        coo_matrix.Clear();

    csr_matrix.Stats().Display(!g_quiet);
    if (!g_quiet)
    {
        printf("\n");
        csr_matrix.DisplayHistogram();
        printf("\n");
        if (g_verbose2) csr_matrix.Display();
        printf("\n");
    }
    fflush(stdout);

    // Host vectors
    ValueT* vector_x     = new ValueT[csr_matrix.num_cols];
    ValueT* vector_y_in  = new ValueT[csr_matrix.num_rows];
    ValueT* vector_y_out = new ValueT[csr_matrix.num_rows];

    for (int col = 0; col < csr_matrix.num_cols; ++col)  vector_x[col]    = 1.0;
    for (int row = 0; row < csr_matrix.num_rows; ++row)  vector_y_in[row] = 1.0;

    SpmvGold(csr_matrix, vector_x, vector_y_in, vector_y_out, alpha, beta);

    float avg_ms, setup_ms;
    float device_giga_bandwidth = args.device_giga_bandwidth;

    if (g_quiet)
    {
        printf("%s, %s, ",
            args.deviceProp.name,
            (sizeof(ValueT) > 4) ? "fp64" : "fp32");
        fflush(stdout);
    }

    // Allocate GPU buffers
    SpmvParams<ValueT, OffsetT> params;

    CubDebugExit(g_allocator.DeviceAllocate((void**)&params.d_values,          sizeof(ValueT) * csr_matrix.num_nonzeros));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&params.d_row_end_offsets, sizeof(OffsetT) * (csr_matrix.num_rows + 1)));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&params.d_column_indices,  sizeof(OffsetT) * csr_matrix.num_nonzeros));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&params.d_vector_x,        sizeof(ValueT) * csr_matrix.num_cols));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&params.d_vector_y,        sizeof(ValueT) * csr_matrix.num_rows));

    params.num_rows     = csr_matrix.num_rows;
    params.num_cols     = csr_matrix.num_cols;
    params.num_nonzeros = csr_matrix.num_nonzeros;
    params.alpha        = alpha;
    params.beta         = beta;

    // FIX: const_cast required — CUDA 12 tightened const rules on cudaMemcpy src
    CubDebugExit(cudaMemcpy(params.d_values,          const_cast<ValueT*>(csr_matrix.values),         sizeof(ValueT)  * csr_matrix.num_nonzeros,      cudaMemcpyHostToDevice));
    CubDebugExit(cudaMemcpy(params.d_row_end_offsets, const_cast<OffsetT*>(csr_matrix.row_offsets),   sizeof(OffsetT) * (csr_matrix.num_rows + 1),    cudaMemcpyHostToDevice));
    CubDebugExit(cudaMemcpy(params.d_column_indices,  const_cast<OffsetT*>(csr_matrix.column_indices),sizeof(OffsetT) * csr_matrix.num_nonzeros,      cudaMemcpyHostToDevice));
    CubDebugExit(cudaMemcpy(params.d_vector_x,        vector_x,                                        sizeof(ValueT)  * csr_matrix.num_cols,          cudaMemcpyHostToDevice));

    // ── Merge-based CsrMV ────────────────────────────────────────────
    if (!g_quiet) printf("\n\n");
    printf("Merge-based CsrMV, "); fflush(stdout);
    avg_ms = TestGpuMergeCsrmv(vector_y_in, vector_y_out, params, timing_iterations, setup_ms);
    DisplayPerf(device_giga_bandwidth, setup_ms, avg_ms, csr_matrix);

    // Initialize cuSPARSE
    cusparseHandle_t cusparse;
    AssertEquals(CUSPARSE_STATUS_SUCCESS, cusparseCreate(&cusparse));

    // ── cuSPARSE CsrMV ───────────────────────────────────────────────
    if (!g_quiet) printf("\n\n");
    printf("cuSPARSE CsrMV, "); fflush(stdout);
    avg_ms = TestCusparseCsrmv(vector_y_in, vector_y_out, params, timing_iterations, setup_ms, cusparse);
    DisplayPerf(device_giga_bandwidth, setup_ms, avg_ms, csr_matrix);

    // ── cuSPARSE HybMV (stubbed — removed in CUDA 11) ────────────────
    if (!g_quiet) printf("\n\n");
    printf("cuSPARSE HybMV, "); fflush(stdout);
    avg_ms = TestCusparseHybmv(vector_y_in, vector_y_out, params, timing_iterations, setup_ms, cusparse);
    DisplayPerf(device_giga_bandwidth, setup_ms, avg_ms, csr_matrix);

    // Cleanup
    // FIX: const_cast required — DeviceFree expects void*, not const T*
    if (params.d_values)          CubDebugExit(g_allocator.DeviceFree((void*)params.d_values));
    if (params.d_row_end_offsets) CubDebugExit(g_allocator.DeviceFree((void*)params.d_row_end_offsets));
    if (params.d_column_indices)  CubDebugExit(g_allocator.DeviceFree((void*)params.d_column_indices));
    if (params.d_vector_x)        CubDebugExit(g_allocator.DeviceFree((void*)params.d_vector_x));
    if (params.d_vector_y)        CubDebugExit(g_allocator.DeviceFree((void*)params.d_vector_y));

    delete[] vector_x;
    delete[] vector_y_in;
    delete[] vector_y_out;

    cusparseDestroy(cusparse);
}


//---------------------------------------------------------------------
// RunTests
//---------------------------------------------------------------------

template <typename ValueT, typename OffsetT>
void RunTests(
    ValueT              alpha,
    ValueT              beta,
    const std::string&  mtx_filename,
    int                 grid2d,
    int                 grid3d,
    int                 wheel,
    int                 dense,
    int                 timing_iterations,
    CommandLineArgs&    args)
{
    CooMatrix<ValueT, OffsetT> coo_matrix;

    if (!mtx_filename.empty())
    {
        coo_matrix.InitMarket(mtx_filename, 1.0, !g_quiet);
        if ((coo_matrix.num_rows == 1) ||
            (coo_matrix.num_cols == 1) ||
            (coo_matrix.num_nonzeros == 1))
        {
            if (!g_quiet) printf("Trivial dataset\n");
            exit(0);
        }
        printf("%s, ", mtx_filename.c_str()); fflush(stdout);
    }
    else if (grid2d > 0)
    {
        printf("grid2d_%d, ", grid2d); fflush(stdout);
        coo_matrix.InitGrid2d(grid2d, false);
    }
    else if (grid3d > 0)
    {
        printf("grid3d_%d, ", grid3d); fflush(stdout);
        coo_matrix.InitGrid3d(grid3d, false);
    }
    else if (wheel > 0)
    {
        printf("wheel_%d, ", wheel); fflush(stdout);
        coo_matrix.InitWheel(wheel);
    }
    else if (dense > 0)
    {
        OffsetT size = 1 << 24;
        args.GetCmdLineArgument("size", size);
        OffsetT rows = size / dense;
        printf("dense_%d_x_%d, ", rows, dense); fflush(stdout);
        coo_matrix.InitDense(rows, dense);
    }
    else
    {
        fprintf(stderr, "No graph type specified.\n");
        exit(1);
    }

    RunTest(alpha, beta, coo_matrix, timing_iterations, args);
}


//---------------------------------------------------------------------
// Main
//---------------------------------------------------------------------

int main(int argc, char** argv)
{
    CommandLineArgs args(argc, argv);
    if (args.CheckCmdLineFlag("help"))
    {
        printf(
            "%s "
            "[--device=<device-id>] "
            "[--quiet] "
            "[--v] "
            "[--i=<timing iterations>] "
            "[--fp32] "
            "[--alpha=<alpha (default: 1.0)>] "
            "[--beta=<beta (default: 0.0)>] "
            "\n\t--mtx=<matrix market file>"
            "\n\t--dense=<cols>"
            "\n\t--grid2d=<width>"
            "\n\t--grid3d=<width>"
            "\n\t--wheel=<spokes>"
            "\n", argv[0]);
        exit(0);
    }

    bool        fp32               = false;
    std::string mtx_filename;
    int         grid2d             = -1;
    int         grid3d             = -1;
    int         wheel              = -1;
    int         dense              = -1;
    int         timing_iterations  = -1;
    float       alpha              = 1.0f;
    float       beta               = 0.0f;

    g_verbose  = args.CheckCmdLineFlag("v");
    g_verbose2 = args.CheckCmdLineFlag("v2");
    g_quiet    = args.CheckCmdLineFlag("quiet");
    fp32       = args.CheckCmdLineFlag("fp32");
    args.GetCmdLineArgument("i",      timing_iterations);
    args.GetCmdLineArgument("mtx",    mtx_filename);
    args.GetCmdLineArgument("grid2d", grid2d);
    args.GetCmdLineArgument("grid3d", grid3d);
    args.GetCmdLineArgument("wheel",  wheel);
    args.GetCmdLineArgument("dense",  dense);
    args.GetCmdLineArgument("alpha",  alpha);
    args.GetCmdLineArgument("beta",   beta);

    CubDebugExit(args.DeviceInit());

    if (fp32)
        RunTests<float,  int>(alpha, beta, mtx_filename, grid2d, grid3d, wheel, dense, timing_iterations, args);
    else
        RunTests<double, int>(alpha, beta, mtx_filename, grid2d, grid3d, wheel, dense, timing_iterations, args);

    CubDebugExit(cudaDeviceSynchronize());
    printf("\n");
    return 0;
}