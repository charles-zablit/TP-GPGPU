#include "helper_cuda.h"
#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"
#include <string.h>

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

__global__ void matrix_gemm_kernel(double *A, double *B, double *C, double alpha, double beta, int M, int K, int N)
{

    // size of thread block
    const int bszx = BLOCK_SIZE_N / THREAD_SIZE_X;
    const int bszy = BLOCK_SIZE_M / THREAD_SIZE_Y;
    const int THREAD_NUM_PER_BLOCK = bszy * bszx;

    // thread id
    const int tid = threadIdx.y * bszx + threadIdx.x;

    // shared memory
    __shared__ double As[BLOCK_SIZE_M][BLOCK_SIZE_K]; // avoid bank conflict
    __shared__ double Bs[BLOCK_SIZE_K][BLOCK_SIZE_N];

    double accum[THREAD_SIZE_Y][THREAD_SIZE_X] = {0};

    const int A_TILE_ROW = tid / BLOCK_SIZE_K;
    const int B_TILE_ROW = tid / BLOCK_SIZE_N;

    const int A_TILE_COL = tid % BLOCK_SIZE_K;
    const int B_TILE_COL = tid % BLOCK_SIZE_N;

    const int A_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / BLOCK_SIZE_K;
    const int B_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / BLOCK_SIZE_N;

    const int A_S = BLOCK_SIZE_M / THREAD_SIZE_Y;
    const int B_S = BLOCK_SIZE_N / THREAD_SIZE_X;

    for (int tile_idx = 0; tile_idx < K; tile_idx += BLOCK_SIZE_K)
    {
        // load A from global memory to shared memory
#pragma unroll
        for (int i = 0; i < BLOCK_SIZE_M; i += A_TILE_ROW_STRIDE)
        {
            const int row = BLOCK_SIZE_M * blockIdx.y + i + A_TILE_ROW;
            const int col = A_TILE_COL + tile_idx;
            if (blockIdx.x == gridDim.x - 1 || blockIdx.y == gridDim.y - 1)
            {
                As[i + A_TILE_ROW][A_TILE_COL] = row < M && col < K ? A[OFFSET(
                                                                          row, // row
                                                                          col, // col
                                                                          K)]
                                                                    : 0;
            }
            else
            {
                As[i + A_TILE_ROW][A_TILE_COL] = A[OFFSET(
                    row, // row
                    col, // col
                    K)];
            }
        }

        // load B from global memory to shared memory
#pragma unroll
        for (int i = 0; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE)
        {
            const int row = tile_idx + i + B_TILE_ROW;
            const int col = B_TILE_COL + BLOCK_SIZE_N * blockIdx.x;
            if (blockIdx.x == gridDim.x - 1 || blockIdx.y == gridDim.y - 1)
            {
                Bs[i + B_TILE_ROW][B_TILE_COL] = row < K && col < N ? B[OFFSET(
                                                                          row, // row
                                                                          col, // col
                                                                          N)]
                                                                    : 0;
            }
            else
            {
                Bs[i + B_TILE_ROW][B_TILE_COL] = B[OFFSET(
                    row, // row
                    col, // col
                    N)];
            }
        }

        __syncthreads();

        // compute c
#pragma unroll
        for (int k = 0; k < BLOCK_SIZE_K; ++k)
        {
#pragma unroll
            for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y)
            {
#pragma unroll
                for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x)
                {
                    // accum[thread_y][thread_x] += frag_a[thread_y] * frag_b[thread_x];
                    accum[thread_y][thread_x] += As[thread_y * A_S + threadIdx.y][k] * Bs[k][thread_x * B_S + threadIdx.x];
                }
            }
        }
        __syncthreads();
    }

    // store back to C
#pragma unroll
    for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y)
    {
#pragma unroll
        for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x)
        {
            const int row = BLOCK_SIZE_M * blockIdx.y + thread_y * A_S + threadIdx.y;
            const int col = BLOCK_SIZE_N * blockIdx.x + thread_x * B_S + threadIdx.x;
            if (blockIdx.x == gridDim.x - 1 || blockIdx.y == gridDim.y - 1)
            {
                if (row < M && col < N)
                {
                    C[OFFSET(row, col, N)] = C[OFFSET(row, col, N)] * beta + accum[thread_y][thread_x] * alpha;
                }
            }
            else
            {
                C[OFFSET(row, col, N)] = C[OFFSET(row, col, N)] * beta + accum[thread_y][thread_x] * alpha;
            }
        }
    }
}

void matrix_gemm(matrix_t *d_m1, matrix_t *d_m2, matrix_t *d_res, double alpha, double beta)
{
    assert((d_m1->columns == d_m2->rows) &&
           (d_m1->rows == d_res->rows) &&
           (d_m2->columns == d_res->columns));

    dim3 threadsPerBlock(BLOCK_SIZE_N / THREAD_SIZE_X, BLOCK_SIZE_M / THREAD_SIZE_Y);
    dim3 blocksPerGrid(CEIL_DIV(d_res->columns, BLOCK_SIZE_N), CEIL_DIV(d_res->rows, BLOCK_SIZE_M));

    matrix_gemm_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_m1->m, d_m2->m, d_res->m, alpha, beta, d_res->rows, d_m1->columns, d_res->columns);
    kernelRetchk;
}

__global__ void matrix_function_kernel(double *A, double *B, bool prime, int numRows, int numColumns)
{
    const unsigned row = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned col = blockIdx.x * blockDim.x + threadIdx.x;

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
    __shared__ float s[THREADS_PER_BLOCK][THREADS_PER_BLOCK + 1];

    int row = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
    int col = blockIdx.y * THREADS_PER_BLOCK + threadIdx.y;
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

__global__ void matrix_scalar_kernel(double *A, double s, int numRows, int numColumns)
{
    const unsigned row = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned col = blockIdx.x * blockDim.x + threadIdx.x;

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