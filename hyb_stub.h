#pragma once
#include <cusparse.h>
#include <cuda_runtime.h>
#if CUDART_VERSION >= 11000
typedef void* cusparseHybMat_t;
#define cusparseCreateHybMat(h)      CUSPARSE_STATUS_SUCCESS
#define cusparseDestroyHybMat(h)     CUSPARSE_STATUS_SUCCESS
#define CUSPARSE_HYB_PARTITION_AUTO  0
#define cusparseDcsr2hyb(...)        CUSPARSE_STATUS_SUCCESS
#define cusparseDhybmv(...)          CUSPARSE_STATUS_SUCCESS
#define cusparseScsr2hyb(...)        CUSPARSE_STATUS_SUCCESS
#define cusparseShybmv(...)          CUSPARSE_STATUS_SUCCESS
#endif
