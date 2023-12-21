#include "helper_cuda.h"
#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"
#include <string.h>

#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))

matrix_t *cuda_alloc_matrix(unsigned rows, unsigned columns)
{
    matrix_t *g_res = (matrix_t *)malloc(sizeof(matrix_t));
    double *m;
    cudaMalloc((double **)&m, columns * rows * sizeof(double));
    kernelRetchk;
    g_res->m = m;
    g_res->columns = columns;
    g_res->rows = rows;
    return g_res;
}

matrix_t *alloc_matrix(unsigned rows, unsigned columns)
{
    matrix_t *res = (matrix_t *)malloc(sizeof(matrix_t));
    res->m = (double *)calloc(columns * rows, sizeof(double));
    res->columns = columns;
    res->rows = rows;
    return res;
}

void free_matrix(matrix_t *m)
{
    // printf("free %p %p\n", m, m->m);
    free(m->m);
    free(m);
}

void cuda_free_matrix(matrix_t *m)
{
    // printf("free %p %p\n", m, m->m);
    cudaFree(m->m);
    kernelRetchk;
    free(m);
}

void print_matrix(matrix_t *m, bool is_short)
{
    unsigned lim_rows = 0;
    unsigned lim_col = 0;

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
            printf("%.2lf ", m->m[col + row * m->columns]);
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

__global__ void hadamard_product_kernel(double *A, double *B, double *C, int numRows, int numColumns)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
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

__global__ void matrix_sum_kernel(double *A, double *B, double *C, int numRows, int numColumns)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < numRows && col < numColumns)
    {
        C[row * numColumns + col] = A[row * numColumns + col] + B[row * numColumns + col];
    }
}

void matrix_sum(matrix_t *d_m1, matrix_t *d_m2, matrix_t *d_res)
{
    assert((d_m1->columns == d_m2->columns) &&
           (d_m1->columns == d_res->columns) &&
           (d_m1->rows == d_m2->rows) &&
           (d_m1->rows == d_res->rows));

    dim3 threadsPerBlock(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
    dim3 blocksPerGrid((d_m1->columns + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (d_m1->rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_sum_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_m1->m, d_m2->m, d_res->m, d_res->rows, d_res->columns);
    kernelRetchk;
}

__global__ void matrix_minus_kernel(double *A, double *B, double *C, int nb_rows, int nb_cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < nb_rows && col < nb_cols)
    {
        C[row * nb_cols + col] = A[row * nb_cols + col] - B[row * nb_cols + col];
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

__global__ void matrix_minus_inplace(double *A, double *B, int nb_rows, int nb_cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < nb_rows && col < nb_cols)
    {
        A[row * nb_cols + col] -= B[row * nb_cols + col];
    }
}

void matrix_minus_inplace(matrix_t *d_m1, matrix_t *d_m2)
{
    assert((d_m1->columns == d_m2->columns) &&
           (d_m1->rows == d_m2->rows));

    dim3 threadsPerBlock(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
    dim3 blocksPerGrid((d_m1->columns + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (d_m1->rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_minus_inplace<<<blocksPerGrid, threadsPerBlock>>>(d_m1->m, d_m2->m, d_m1->rows, d_m1->columns);
    kernelRetchk;
}

__global__ void matrix_dot_kernel(double *A, double *B, double *C, int nb_rows_A, int nb_cols_A, int nb_cols_B)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < nb_rows_A && col < nb_cols_B)
    {
        double sum = 0;
        for (int i = 0; i < nb_cols_A; i++)
        {
            sum += A[row * nb_cols_A + i] * B[i * nb_cols_B + col];
        }
        C[row * nb_cols_B + col] = sum;
    }
}

void matrix_dot(matrix_t *d_m1, matrix_t *d_m2, matrix_t *d_res)
{
    assert((d_m1->columns == d_m2->rows) &&
           (d_m1->rows == d_res->rows) &&
           (d_m2->columns == d_res->columns));

    dim3 threadsPerBlock(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
    dim3 blocksPerGrid((d_res->columns + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (d_res->rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_dot_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_m1->m, d_m2->m, d_res->m, d_m1->rows, d_m1->columns, d_m2->columns);
    kernelRetchk;
}

__global__ void matrix_function_kernel(double *A, double *B, bool prime, int numRows, int numColumns)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < numRows && col < numColumns)
    {
        double x = A[row * numColumns + col];
        double sig = 1 / (1 + exp(-x));
        if (prime)
        {
            sig = sig * (1 - sig);
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

__global__ void matrix_transpose_kernel(double *A, double *B, int nb_rows, int nb_cols)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < nb_rows && col < nb_cols)
    {
        B[row * nb_cols + col] = A[col * nb_rows + row];
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

__global__ void matrix_scalar_kernel(double *A, double s, int numRows, int numColumns)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < numRows && col < numColumns)
    {
        A[row * numColumns + col] *= s;
    }
}

void matrix_scalar(matrix_t *d_m, double s)
{
    dim3 threadsPerBlock(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
    dim3 blocksPerGrid((d_m->columns + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (d_m->rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_scalar_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_m->m, s, d_m->rows, d_m->columns);
    kernelRetchk;
}

void matrix_memcpy(matrix_t *dest, const matrix_t *src)
{
    assert((dest->rows == src->rows) &&
           (dest->columns == src->columns));

    memcpy(dest->m, src->m, src->columns * src->rows * sizeof(double));
}

void matrix_cudaMemcpy(matrix_t *dest, const matrix_t *src, cudaMemcpyKind kind)
{
    assert((dest->rows == src->rows) &&
           (dest->columns == src->columns));

    cudaMemcpy(dest->m, src->m, src->columns * src->rows * sizeof(double), kind);
    kernelRetchk;
}

void init_ones(matrix_t *d_m)
{
    matrix_t *h_m = alloc_matrix(d_m->rows, d_m->columns);
    for (int idx = 0; idx < h_m->columns * h_m->rows; idx++)
    {
        h_m->m[idx] = 1.0f;
    }
    matrix_cudaMemcpy(d_m, h_m, cudaMemcpyHostToDevice);
}