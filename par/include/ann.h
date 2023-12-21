#pragma once
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "matrix.h"

typedef struct
{
    unsigned minibatch_size;
    unsigned number_of_neurons;

    matrix_t *d_weights;
    matrix_t *d_biases;

    matrix_t *d_z;
    matrix_t *d_activations;

    matrix_t *d_delta;
} layer_t;

typedef struct
{
    void (*f)(double *, double *, unsigned, unsigned);
    void (*fd)(double *, double *, unsigned, unsigned);
    double alpha;
    unsigned minibatch_size;
    unsigned input_size;
    unsigned number_of_layers;
    layer_t **layers;
    matrix_t *d_one;
    matrix_t *d_oneT;
} ann_t;

double normalRand(double mu, double sigma);

ann_t *create_ann(double alpha, unsigned minibatch_size, unsigned number_of_layers, unsigned *nneurons_per_layer);

layer_t *create_layer(unsigned l, unsigned number_of_neurons, unsigned nneurons_previous_layer, unsigned minibatch_size);

void set_input(ann_t *nn, matrix_t *input);

void print_nn(ann_t *nn);

void forward(ann_t *nn);

void backward(ann_t *nn, matrix_t *y);
