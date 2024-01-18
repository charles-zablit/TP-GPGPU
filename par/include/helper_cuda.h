#pragma once
#include <stdio.h>
#include <stdlib.h>
#include "log.h"

#define gpuErrchk(ans)                        \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

#ifdef DEBUG
#define kernelRetchk                    \
    gpuErrchk(cudaDeviceSynchronize()); \
    gpuErrchk(cudaPeekAtLastError());

// cublas API error checking
#define CUBLAS_CHECK(err)                                                        \
    do                                                                           \
    {                                                                            \
        cublasStatus_t err_ = (err);                                             \
        if (err_ != CUBLAS_STATUS_SUCCESS)                                       \
        {                                                                        \
            std::printf("cublas error %d at %s:%d\n", err_, __FILE__, __LINE__); \
            throw std::runtime_error("cublas error");                            \
        }                                                                        \
    } while (0)
#else
#define CUBLAS_CHECK
#define kernelRetchk
#endif