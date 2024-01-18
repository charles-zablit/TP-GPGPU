#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <cublas_v2.h>
#define __CUDA_NO_HALF_CONVERSIONS__
#include "cuda_fp16.h"

#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))
#define OFFSET(row, col, ld) ((row) * (ld) + (col))

const int THREADS_PER_BLOCK = 32;
const int BLOCK_SIZE_M = 24;
const int BLOCK_SIZE_K = 16;
const int BLOCK_SIZE_N = 16;
const int THREAD_SIZE_Y = 6;
const int THREAD_SIZE_X = 4;

typedef struct
{
    __half *m;
    uint16_t columns;
    uint16_t rows;
} matrix_t;

matrix_t *cuda_alloc_matrix(uint16_t rows, uint16_t columns);

matrix_t *alloc_matrix(uint16_t rows, uint16_t columns);

void free_matrix(matrix_t *m);

void cuda_free_matrix(matrix_t *m);

void print_matrix(matrix_t *m, bool is_short);

void cuda_print_matrix(matrix_t *d_m, bool is_short);

__global__ void hadamard_product_kernel(__half *A, __half *B, __half *C, uint16_t numRows, uint16_t numColumns);

void hadamard_product(matrix_t *m1, matrix_t *m2, matrix_t *res);

__global__ void matrix_sum_kernel(__half *A, __half *B, __half *C, uint16_t numRows, uint16_t numColumns);

void matrix_sum(matrix_t *m1, matrix_t *m2, matrix_t *res);

__global__ void matrix_minus_kernel(__half *A, __half *B, __half *C, uint16_t numRows, uint16_t numColumns);

void matrix_minus(matrix_t *m1, matrix_t *m2, matrix_t *res);

__global__ void matrix_gemm_kernel(__half *A, __half *B, __half *C, __half alpha, __half beta, uint16_t numRowsA, uint16_t numColumnsA, uint16_t numColumnsB);

void matrix_gemm(cublasHandle_t *handle, matrix_t *d_m1, matrix_t *d_m2, matrix_t *d_res, cublasOperation_t t_m1 = CUBLAS_OP_N, cublasOperation_t t_m2 = CUBLAS_OP_N, __half alpha = __float2half(1.0f), __half beta = __float2half(0.0f));

__global__ void matrix_function_kernel(__half *A, __half *B, bool prime, __half numRows, __half numColumns);

void matrix_function(matrix_t *m1, bool prime, matrix_t *res);

__global__ void matrix_transpose_kernel(__half *A, __half *B, uint16_t numRows, uint16_t numColumns);

void matrix_transpose(matrix_t *m1, matrix_t *res);

void matrix_memcpy(matrix_t *dest, const matrix_t *src);

void matrix_cudaMemcpy(matrix_t *dest, const matrix_t *src, cudaMemcpyKind kind);

void init_ones(matrix_t *d_m);
