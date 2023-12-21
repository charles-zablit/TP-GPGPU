#pragma once

#include "mnist.h"
#include "ann.h"
#include "helper_cuda.h"

void populate_minibatch(double *x, double *y, unsigned *minibatch_idx, unsigned minibatch_size, image *img, unsigned img_size, byte *label, unsigned label_size);

void zero_to_n(unsigned n, unsigned *t);

void shuffle(unsigned *t, const unsigned size, const unsigned number_of_switch);

double accuracy(image *test_img, byte *test_label, unsigned datasize, unsigned minibatch_size, ann_t *nn);