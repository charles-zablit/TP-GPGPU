#include "tests.h"

#include <stdio.h>
#include <stdlib.h>

#include "mnist.h"
#include "matrix.h"
#include "ann.h"
#include <math.h>
#include <string.h>
#include <time.h>

void test_matrix_dot()
{
    printf("----------------\n");
    printf("Dot product test\n");
    printf("----------------\n");
    unsigned n = 30, m = 16;
    matrix_t *h_m1 = alloc_matrix(n, m);
    matrix_t *d_m1 = cuda_alloc_matrix(n, m);
    for (int i = 0; i < n * m; i++)
    {
        h_m1->m[i] = i + 1;
    }
    matrix_cudaMemcpy(d_m1, h_m1, cudaMemcpyHostToDevice);
    cuda_print_matrix(d_m1, false);

    matrix_t *h_m2 = alloc_matrix(m, m);
    matrix_t *d_m2 = cuda_alloc_matrix(m, m);
    for (int i = 0; i < m * m; i++)
    {
        if (i % (m + 1) == 0)
        {
            h_m2->m[i] = 1;
        }
    }
    matrix_cudaMemcpy(d_m2, h_m2, cudaMemcpyHostToDevice);
    cuda_print_matrix(d_m2, false);

    matrix_t *h_m3 = alloc_matrix(n, m);
    matrix_t *d_m3 = cuda_alloc_matrix(n, m);
    matrix_dot(d_m1, d_m2, d_m3);
    cuda_print_matrix(d_m3, false);
    matrix_cudaMemcpy(h_m3, d_m3, cudaMemcpyDeviceToHost);

    for (int i = 0; i < n * m; i++)
    {
        assert(h_m3->m[i] == h_m1->m[i]);
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
        assert(h_m3->m[i] == (i + 1) * (i + 1));
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

void test_matrix_scalar()
{
    printf("-------------------\n");
    printf("Scalar product test\n");
    printf("-------------------\n");
    unsigned n = 50;
    matrix_t *h_m1 = alloc_matrix(n, n);
    matrix_t *d_m1 = cuda_alloc_matrix(n, n);
    for (int i = 0; i < n * n; i++)
    {
        h_m1->m[i] = i + 1;
    }
    matrix_cudaMemcpy(d_m1, h_m1, cudaMemcpyHostToDevice);
    cuda_print_matrix(d_m1, false);

    matrix_scalar(d_m1, 10.0);
    cuda_print_matrix(d_m1, false);
    matrix_cudaMemcpy(h_m1, d_m1, cudaMemcpyDeviceToHost);

    for (int i = 0; i < n * n; i++)
    {
        assert(h_m1->m[i] == (i + 1) * 10.0);
    }

    free_matrix(h_m1);
    cuda_free_matrix(d_m1);
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

    assert(h_m2->m[0] == 1.0);
    assert(h_m2->m[1] == 3.0);
    assert(h_m2->m[2] == 2.0);
    assert(h_m2->m[3] == 4.0);

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
        assert(h_m2->m[i] == 2.0 * (i + 1));
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
        assert(h_m2->m[i] == 0.0);
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
    test_matrix_dot();
    test_hadamard_product();
    test_matrix_scalar();
    test_matrix_transpose();
    test_matrix_sum();
    test_matrix_minus();
    return 0;
}