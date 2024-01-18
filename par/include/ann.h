#pragma once
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include "matrix.h"
#include "cuda_fp16.h"

typedef struct
{
    uint16_t minibatch_size;
    uint16_t number_of_neurons;

    matrix_t *d_weights;
    matrix_t *d_biases;

    matrix_t *d_z;
    matrix_t *d_activations;

    matrix_t *d_delta;
} layer_t;

typedef struct
{
    __half alpha;
    uint16_t minibatch_size;
    uint16_t input_size;
    uint16_t number_of_layers;
    layer_t **layers;
    matrix_t *d_one;
    matrix_t *d_oneT;
} ann_t;

void init_weight(matrix_t *w, uint16_t nneurones_prev);

void print_layer(layer_t *layer);

ann_t *create_ann(__half alpha, uint16_t minibatch_size, uint16_t number_of_layers, uint16_t *nneurons_per_layer);

layer_t *create_layer(uint16_t l, uint16_t number_of_neurons, uint16_t nneurons_previous_layer, uint16_t minibatch_size);

void set_input(ann_t *nn, matrix_t *input);

void print_nn(ann_t *nn);

void forward(ann_t *nn);

void backward(ann_t *nn, matrix_t *y);
