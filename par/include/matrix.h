#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <stdbool.h>

#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))
#define OFFSET(row, col, ld) ((row) * (ld) + (col))

const int THREADS_PER_BLOCK = 8;
const int BLOCK_SIZE_M = 24;
const int BLOCK_SIZE_K = 16;
const int BLOCK_SIZE_N = 16;
const int THREAD_SIZE_Y = 6;
const int THREAD_SIZE_X = 4;

typedef struct
{
    double *m;
    unsigned columns;
    unsigned rows;
} matrix_t;

matrix_t *cuda_alloc_matrix(unsigned rows, unsigned columns);

matrix_t *alloc_matrix(unsigned rows, unsigned columns);

void free_matrix(matrix_t *m);

void cuda_free_matrix(matrix_t *m);

void print_matrix(matrix_t *m, bool is_short);

void cuda_print_matrix(matrix_t *d_m, bool is_short);

__global__ void hadamard_product_kernel(double *A, double *B, double *C, int numRows, int numColumns);

void hadamard_product(matrix_t *m1, matrix_t *m2, matrix_t *res);

__global__ void matrix_sum_kernel(double *A, double *B, double *C, int numRows, int numColumns);

void matrix_sum(matrix_t *m1, matrix_t *m2, matrix_t *res);

__global__ void matrix_minus_kernel(double *A, double *B, double *C, int numRows, int numColumns);

void matrix_minus(matrix_t *m1, matrix_t *m2, matrix_t *res);

__global__ void matrix_minus_inplace(double *A, double *B, int nb_rows, int nb_cols);

void matrix_minus_inplace(matrix_t *d_m1, matrix_t *d_m2);

__global__ void matrix_gemm_kernel(double *A, double *B, double *C, double alpha, double beta, int nb_rows_A, int nb_cols_A, int nb_cols_B);

void matrix_gemm(matrix_t *d_m1, matrix_t *d_m2, matrix_t *d_res, double alpha = 1.0, double beta = 0.0);

__global__ void matrix_function_kernel(double *A, double *B, bool prime, int numRows, int numColumns);

void matrix_function(matrix_t *m1, bool prime, matrix_t *res);

__global__ void matrix_transpose_kernel(double *A, double *B, int numRows, int numColumns);

void matrix_transpose(matrix_t *m1, matrix_t *res);

__global__ void matrix_scalar_kernel(double *A, double s, int numRows, int numColumns);

void matrix_scalar(matrix_t *d_m, double s);

void matrix_memcpy(matrix_t *dest, const matrix_t *src);

void matrix_cudaMemcpy(matrix_t *dest, const matrix_t *src, cudaMemcpyKind kind);

void init_ones(matrix_t *d_m);
