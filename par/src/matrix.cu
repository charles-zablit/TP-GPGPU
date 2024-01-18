#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "helper_cuda.h"
#include "matrix.h"

matrix_t *cuda_alloc_matrix(uint16_t rows, uint16_t columns)
{
    matrix_t *g_res = (matrix_t *)malloc(sizeof(matrix_t));
    __half *m;
    cudaMalloc((__half **)&m, columns * rows * sizeof(__half));
    kernelRetchk;
    g_res->m = m;
    g_res->columns = columns;
    g_res->rows = rows;
    return g_res;
}

matrix_t *alloc_matrix(uint16_t rows, uint16_t columns)
{
    matrix_t *res = (matrix_t *)malloc(sizeof(matrix_t));
    res->m = (__half *)calloc(columns * rows, sizeof(__half));
    res->columns = columns;
    res->rows = rows;
    return res;
}

void free_matrix(matrix_t *m)
{
    free(m->m);
    free(m);
}

void cuda_free_matrix(matrix_t *m)
{
    cudaFree(m->m);
    kernelRetchk;
    free(m);
}

void print_matrix(matrix_t *m, bool is_short)
{
    uint16_t lim_rows = 0;
    uint16_t lim_col = 0;

    if (is_short)
    {
        lim_rows = MIN(m->rows, 4);
        lim_col = MIN(m->columns, 10);
    }
    else
    {
        lim_rows = m->rows;
        lim_col = m->columns;
    }

    for (int row = 0; row < lim_rows; row++)
    {
        printf("|");
        for (int col = 0; col < lim_col; col++)
        {
            printf("%.2lf ", __half2float(m->m[col + row * m->columns]));
        }
        if (is_short && lim_col != m->columns)
            printf("...");
        printf("|\n");
    }
    if (is_short && lim_rows != m->rows)
    {
        printf("...\n");
    }
}

void cuda_print_matrix(matrix_t *d_m, bool is_short)
{
    matrix_t *m = alloc_matrix(d_m->rows, d_m->columns);
    matrix_cudaMemcpy(m, d_m, cudaMemcpyDeviceToHost);
    print_matrix(m, is_short);
    free_matrix(m);
}

__global__ void hadamard_product_kernel(__half *A, __half *B, __half *C, uint16_t numRows, uint16_t numColumns)
{
    uint16_t row = blockIdx.y * blockDim.y + threadIdx.y;
    uint16_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < numRows && col < numColumns)
    {
        C[row * numColumns + col] = A[row * numColumns + col] * B[row * numColumns + col];
    }
}

void hadamard_product(matrix_t *d_m1, matrix_t *d_m2, matrix_t *d_res)
{
    assert((d_m1->columns == d_m2->columns) &&
           (d_m1->columns == d_res->columns) &&
           (d_m1->rows == d_m2->rows) &&
           (d_m1->rows == d_res->rows));

    dim3 threadsPerBlock(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
    dim3 blocksPerGrid((d_m1->columns + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (d_m1->rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    hadamard_product_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_m1->m, d_m2->m, d_res->m, d_res->rows, d_res->columns);
    kernelRetchk;
}

__global__ void matrix_minus_kernel(__half *A, __half *B, __half *C, uint16_t numRows, uint16_t numColumns)
{
    uint16_t row = blockIdx.y * blockDim.y + threadIdx.y;
    uint16_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < numRows && col < numColumns)
    {
        C[row * numColumns + col] = A[row * numColumns + col] - B[row * numColumns + col];
    }
}

void matrix_minus(matrix_t *d_m1, matrix_t *d_m2, matrix_t *d_res)
{
    assert((d_m1->columns == d_m2->columns) &&
           (d_m1->columns == d_res->columns) &&
           (d_m1->rows == d_m2->rows) &&
           (d_m1->rows == d_res->rows));

    dim3 threadsPerBlock(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
    dim3 blocksPerGrid((d_m1->columns + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (d_m1->rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_minus_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_m1->m, d_m2->m, d_res->m, d_res->rows, d_res->columns);
    kernelRetchk;
}

void matrix_gemm(cublasHandle_t *handle, matrix_t *d_m1, matrix_t *d_m2, matrix_t *d_res, cublasOperation_t t_m1, cublasOperation_t t_m2, __half alpha, __half beta)
{
    const int m = t_m1 == CUBLAS_OP_T ? d_m1->columns : d_m1->rows;
    const int n = t_m2 == CUBLAS_OP_T ? d_m2->rows : d_m2->columns;
    const int k = t_m1 == CUBLAS_OP_T ? d_m1->rows : d_m1->columns;

    assert((m == d_res->rows) &&
           (n == d_res->columns));

    CUBLAS_CHECK(cublasHgemm(*handle,
                             t_m2,
                             t_m1,
                             n, m, k,
                             &alpha,
                             d_m2->m, d_res->columns,
                             d_m1->m, d_m1->columns,
                             &beta,
                             d_res->m, d_res->columns));
}

__global__ void matrix_function_kernel(__half *A, __half *B, bool prime, uint16_t numRows, uint16_t numColumns)
{
    const uint16_t row = blockIdx.y * blockDim.y + threadIdx.y;
    const uint16_t col = blockIdx.x * blockDim.x + threadIdx.x;
    const __half one = __float2half(1.0f);
    if (row < numRows && col < numColumns)
    {
        __half x = A[row * numColumns + col];
        __half sig = one / (one + hexp(-x));
        if (prime)
        {
            sig = sig * (one - sig);
        }
        B[row * numColumns + col] = sig;
    }
}

void matrix_function(matrix_t *d_m, bool prime, matrix_t *d_res)
{
    assert((d_m->columns == d_res->columns) &&
           (d_m->rows == d_res->rows));

    dim3 threadsPerBlock(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
    dim3 blocksPerGrid((d_m->columns + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (d_m->rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_function_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_m->m, d_res->m, prime, d_res->rows, d_res->columns);
    kernelRetchk;
}

__global__ void matrix_transpose_kernel(__half *A, __half *B, uint16_t nb_rows, uint16_t nb_cols)
{
    __shared__ __half s[THREADS_PER_BLOCK][THREADS_PER_BLOCK + 1];

    uint16_t row = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
    uint16_t col = blockIdx.y * THREADS_PER_BLOCK + threadIdx.y;
    if ((row < nb_rows) && (col < nb_cols))
    {
        s[threadIdx.y][threadIdx.x] = A[col * nb_rows + row];
    }

    __syncthreads();

    row = blockIdx.y * THREADS_PER_BLOCK + threadIdx.x;
    col = blockIdx.x * THREADS_PER_BLOCK + threadIdx.y;
    if ((row < nb_cols) && (col < nb_rows))
    {
        B[col * nb_cols + row] = s[threadIdx.x][threadIdx.y];
    }
}

void matrix_transpose(matrix_t *d_m, matrix_t *d_res)
{
    assert((d_m->columns == d_res->rows) &&
           (d_m->rows == d_res->columns));

    dim3 threadsPerBlock(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
    dim3 blocksPerGrid((d_m->columns + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (d_m->rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_m->m, d_res->m, d_res->rows, d_res->columns);
    kernelRetchk;
}

void matrix_memcpy(matrix_t *dest, const matrix_t *src)
{
    assert((dest->rows == src->rows) &&
           (dest->columns == src->columns));

    memcpy(dest->m, src->m, src->columns * src->rows * sizeof(__half));
}

void matrix_cudaMemcpy(matrix_t *dest, const matrix_t *src, cudaMemcpyKind kind)
{
    assert((dest->rows == src->rows) &&
           (dest->columns == src->columns));

    cudaMemcpy(dest->m, src->m, src->columns * src->rows * sizeof(__half), kind);
    kernelRetchk;
}

void init_ones(matrix_t *d_m)
{
    matrix_t *h_m = alloc_matrix(d_m->rows, d_m->columns);
    for (int idx = 0; idx < h_m->columns * h_m->rows; idx++)
    {
        h_m->m[idx] = __float2half(1.0f);
    }
    matrix_cudaMemcpy(d_m, h_m, cudaMemcpyHostToDevice);
}