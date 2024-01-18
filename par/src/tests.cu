#include "tests.h"

#include <stdio.h>
#include <stdlib.h>

#include "mnist.h"
#include "matrix.h"
#include "ann.h"
#include <math.h>
#include <string.h>
#include <time.h>

void matrix_dot_ref(matrix_t *m1, matrix_t *m2, matrix_t *res)
{
    assert((m1->columns == m2->rows) &&
           (m1->rows == res->rows) &&
           (m2->columns == res->columns));

    for (int row = 0; row < m1->rows; row++)
    {
        for (int col = 0; col < m2->columns; col++)
        {
            int idx = col + row * m2->columns;
            __half var = 0.0;

            for (int ii = 0; ii < m1->columns; ii++)
            {
                var += m1->m[ii + row * m1->columns] * m2->m[col + ii * m2->columns];
            }

            res->m[idx] = var;
        }
    }
}

void test_matrix_gemm()
{
    printf("----------------\n");
    printf("Dot product test\n");
    printf("----------------\n");
    unsigned n = 784, m = 30, p = 10;
    matrix_t *h_m1 = alloc_matrix(n, m);
    matrix_t *d_m1 = cuda_alloc_matrix(n, m);
    for (int i = 0; i < n * m; i++)
    {
        h_m1->m[i] = i + 1;
    }
    matrix_cudaMemcpy(d_m1, h_m1, cudaMemcpyHostToDevice);
    cuda_print_matrix(d_m1, false);
    printf("\n");

    matrix_t *h_m2 = alloc_matrix(m, p);
    matrix_t *d_m2 = cuda_alloc_matrix(m, p);
    for (int i = 0; i < m * p; i++)
    {
        h_m2->m[i] = i + 1;
    }
    matrix_cudaMemcpy(d_m2, h_m2, cudaMemcpyHostToDevice);
    cuda_print_matrix(d_m2, false);
    printf("\n");

    matrix_t *h_m3 = alloc_matrix(n, p);
    matrix_t *h_m3ref = alloc_matrix(n, p);
    matrix_t *d_m3 = cuda_alloc_matrix(n, p);
    for (int i = 0; i < n * p; i++)
    {
        h_m3->m[i] = 0.0;
    }
    matrix_cudaMemcpy(d_m3, h_m3, cudaMemcpyHostToDevice);
    matrix_gemm(d_m1, d_m2, d_m3);
    matrix_dot_ref(h_m1, h_m2, h_m3ref);
    print_matrix(h_m3ref, false);
    printf("\n");
    cuda_print_matrix(d_m3, false);
    printf("\n");
    matrix_cudaMemcpy(h_m3, d_m3, cudaMemcpyDeviceToHost);

    for (int i = 0; i < n * p; i++)
    {
        if (h_m3->m[i] != h_m3ref->m[i])
        {
            int row = (int)i / p;
            int col = i % p;
            printf("(%d  %d) got: %f, expected: %f\n", row, col, h_m3->m[i], h_m3ref->m[i]);
        }
    }

    free_matrix(h_m1);
    free_matrix(h_m2);
    free_matrix(h_m3);
    cuda_free_matrix(d_m1);
    cuda_free_matrix(d_m2);
    cuda_free_matrix(d_m3);
    printf("---\n");
    printf("OK\n");
    printf("---\n");
}

void test_hadamard_product()
{
    printf("---------------------\n");
    printf("Hadamard product test\n");
    printf("---------------------\n");
    unsigned n = 50, m = 30;
    matrix_t *h_m1 = alloc_matrix(n, m);
    matrix_t *d_m1 = cuda_alloc_matrix(n, m);
    for (int i = 0; i < n * m; i++)
    {
        h_m1->m[i] = i + 1;
    }
    matrix_cudaMemcpy(d_m1, h_m1, cudaMemcpyHostToDevice);
    cuda_print_matrix(d_m1, false);

    matrix_t *h_m2 = alloc_matrix(n, m);
    matrix_t *d_m2 = cuda_alloc_matrix(n, m);
    for (int i = 0; i < n * m; i++)
    {
        h_m2->m[i] = i + 1;
    }
    matrix_cudaMemcpy(d_m2, h_m2, cudaMemcpyHostToDevice);
    cuda_print_matrix(d_m2, false);

    matrix_t *h_m3 = alloc_matrix(n, m);
    matrix_t *d_m3 = cuda_alloc_matrix(n, m);
    hadamard_product(d_m1, d_m2, d_m3);
    cuda_print_matrix(d_m3, false);
    matrix_cudaMemcpy(h_m3, d_m3, cudaMemcpyDeviceToHost);

    for (int i = 0; i < n * m; i++)
    {
        assert(h_m3->m[i] == (__half)((i + 1) * (i + 1)));
    }

    free_matrix(h_m1);
    free_matrix(h_m2);
    free_matrix(h_m3);
    cuda_free_matrix(d_m1);
    cuda_free_matrix(d_m2);
    cuda_free_matrix(d_m3);
    printf("---\n");
    printf("OK\n");
    printf("---\n");
}

void test_matrix_transpose()
{
    printf("--------------\n");
    printf("Transpose test\n");
    printf("--------------\n");
    unsigned n = 2;
    matrix_t *h_m1 = alloc_matrix(n, n);
    matrix_t *d_m1 = cuda_alloc_matrix(n, n);
    for (int i = 0; i < n * n; i++)
    {
        h_m1->m[i] = i + 1;
    }
    matrix_cudaMemcpy(d_m1, h_m1, cudaMemcpyHostToDevice);
    cuda_print_matrix(d_m1, false);

    matrix_t *h_m2 = alloc_matrix(n, n);
    matrix_t *d_m2 = cuda_alloc_matrix(n, n);

    matrix_transpose(d_m1, d_m2);
    cuda_print_matrix(d_m2, false);
    matrix_cudaMemcpy(h_m2, d_m2, cudaMemcpyDeviceToHost);

    assert(h_m2->m[0] == (__half)1.0);
    assert(h_m2->m[1] == (__half)3.0);
    assert(h_m2->m[2] == (__half)2.0);
    assert(h_m2->m[3] == (__half)4.0);

    free_matrix(h_m1);
    free_matrix(h_m2);
    cuda_free_matrix(d_m1);
    cuda_free_matrix(d_m2);
    printf("---\n");
    printf("OK\n");
    printf("---\n");
}

void test_matrix_sum()
{
    printf("--------\n");
    printf("Sum test\n");
    printf("--------\n");
    unsigned n = 50, m = 30;
    matrix_t *h_m1 = alloc_matrix(n, m);
    matrix_t *d_m1 = cuda_alloc_matrix(n, m);
    for (int i = 0; i < n * m; i++)
    {
        h_m1->m[i] = i + 1;
    }
    matrix_cudaMemcpy(d_m1, h_m1, cudaMemcpyHostToDevice);
    cuda_print_matrix(d_m1, false);

    matrix_t *h_m2 = alloc_matrix(n, m);
    matrix_t *d_m2 = cuda_alloc_matrix(n, m);
    matrix_sum(d_m1, d_m1, d_m2);
    cuda_print_matrix(d_m2, false);
    matrix_cudaMemcpy(h_m2, d_m2, cudaMemcpyDeviceToHost);

    for (int i = 0; i < n * m; i++)
    {
        assert(h_m2->m[i] == (__half)(2.0 * (i + 1)));
    }

    free_matrix(h_m1);
    free_matrix(h_m2);
    cuda_free_matrix(d_m1);
    cuda_free_matrix(d_m2);
    printf("---\n");
    printf("OK\n");
    printf("---\n");
}

void test_matrix_minus()
{
    printf("----------\n");
    printf("Minus test\n");
    printf("----------\n");
    unsigned n = 50, m = 30;
    matrix_t *h_m1 = alloc_matrix(n, m);
    matrix_t *d_m1 = cuda_alloc_matrix(n, m);
    for (int i = 0; i < n * m; i++)
    {
        h_m1->m[i] = i + 1;
    }
    matrix_cudaMemcpy(d_m1, h_m1, cudaMemcpyHostToDevice);
    cuda_print_matrix(d_m1, false);

    matrix_t *h_m2 = alloc_matrix(n, m);
    matrix_t *d_m2 = cuda_alloc_matrix(n, m);
    matrix_minus(d_m1, d_m1, d_m2);
    cuda_print_matrix(d_m2, false);
    matrix_cudaMemcpy(h_m2, d_m2, cudaMemcpyDeviceToHost);

    for (int i = 0; i < n * m; i++)
    {
        assert(h_m2->m[i] == (__half)0.0);
    }

    free_matrix(h_m1);
    free_matrix(h_m2);
    cuda_free_matrix(d_m1);
    cuda_free_matrix(d_m2);
    printf("---\n");
    printf("OK\n");
    printf("---\n");
}

int run_tests()
{
    test_matrix_gemm();
    test_hadamard_product();
    test_matrix_transpose();
    test_matrix_sum();
    test_matrix_minus();
    return 0;
}