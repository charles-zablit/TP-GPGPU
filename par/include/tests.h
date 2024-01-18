#pragma once

#undef __CUDA_NO_HALF_CONVERSIONS__
#include "cuda_fp16.h"

void test_matrix_gemm();

void test_hadamard_product();

void test_matrix_transpose();

void test_matrix_minus();

int run_tests();
