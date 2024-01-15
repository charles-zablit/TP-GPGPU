#pragma once

#include "mnist.h"
#include "ann.h"
#include "helper_cuda.h"

#include "model_utils.h"

void zero_to_n(unsigned n, unsigned *t);

void shuffle(unsigned *t, const unsigned size, const unsigned number_of_switch);

double accuracy(image *d_test_img, byte *d_test_label, byte *test_label, unsigned datasize, unsigned minibatch_size, ann_t *nn);

__global__ void populateX(double *x, image *img, unsigned *minibatch_idx, unsigned minibatch_size, unsigned img_size);

__global__ void populateY(double *y, byte *label, unsigned *minibatch_idx, unsigned minibatch_size, unsigned label_size);

void gpuPopulateMinibatch(double *d_x, double *d_y, unsigned *minibatch_idx, unsigned minibatch_size, image *d_img, unsigned img_size, byte *d_label, unsigned datasize);