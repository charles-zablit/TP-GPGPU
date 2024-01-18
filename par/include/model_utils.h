#pragma once

#include <stdint.h>

#include "mnist.h"
#include "ann.h"
#include "helper_cuda.h"
#include "model_utils.h"
#include "cuda_fp16.h"

void zero_to_n(uint16_t n, uint16_t *t);

void shuffle(uint16_t *t, const uint16_t size, const uint16_t number_of_switch);

double accuracy(image *d_test_img, byte *d_test_label, byte *test_label, uint16_t datasize, uint16_t minibatch_size, ann_t *nn);

__global__ void populateX(__half *x, image *img, uint16_t *minibatch_idx, uint16_t minibatch_size, uint16_t img_size);

__global__ void populateY(__half *y, byte *label, uint16_t *minibatch_idx, uint16_t minibatch_size, uint16_t label_size);

void gpuPopulateMinibatch(__half *d_x, __half *d_y, uint16_t *minibatch_idx, uint16_t minibatch_size, image *d_img, uint16_t img_size, byte *d_label, uint16_t datasize);