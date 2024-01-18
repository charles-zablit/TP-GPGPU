#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>
#include <stdint.h>
#include <random>

#include "ann.h"
#include "matrix.h"
#include "helper_cuda.h"
#include "log.h"

void init_weight(matrix_t *d_w, uint16_t nneurones_prev)
{
    log_debug("Init weights");

    std::mt19937 gen(0);
    std::normal_distribution<float> d(0, 1 / sqrt(nneurones_prev));

    matrix_t *h_w = alloc_matrix(d_w->rows, d_w->columns);
    for (int idx = 0; idx < h_w->columns * h_w->rows; idx++)
    {
        h_w->m[idx] = __float2half(d(gen));
    }

    matrix_cudaMemcpy(d_w, h_w, cudaMemcpyHostToDevice);
    free_matrix(h_w);
    log_debug("Done init weights");
}

ann_t *create_ann(__half alpha, uint16_t minibatch_size, uint16_t number_of_layers, uint16_t *nneurons_per_layer)
{
    ann_t *nn = (ann_t *)malloc(sizeof(ann_t));

    nn->layers = (layer_t **)malloc(number_of_layers * sizeof(layer_t *));
    nn->number_of_layers = number_of_layers;
    nn->alpha = alpha;
    nn->minibatch_size = minibatch_size;
    nn->d_one = cuda_alloc_matrix(1, nn->minibatch_size);
    init_ones(nn->d_one);
    nn->d_oneT = cuda_alloc_matrix(nn->minibatch_size, 1);
    init_ones(nn->d_oneT);

    log_debug("Creating layer [%d/%d]", 0, number_of_layers - 1);
    nn->layers[0] = create_layer(0, nneurons_per_layer[0], minibatch_size, minibatch_size);
    for (int l = 1; l < number_of_layers; l++)
    {
        log_debug("Creating layer [%d/%d]", l, number_of_layers - 1);
        nn->layers[l] = create_layer(l, nneurons_per_layer[l], nneurons_per_layer[l - 1], minibatch_size);
    }
    log_debug("Done creating layers");

    return nn;
}

layer_t *create_layer(uint16_t layer_number, uint16_t number_of_neurons, uint16_t nneurons_previous_layer, uint16_t minibatch_size)
{
    layer_t *layer = (layer_t *)malloc(sizeof(layer_t));

    layer->number_of_neurons = number_of_neurons;
    layer->minibatch_size = minibatch_size;
    layer->d_activations = cuda_alloc_matrix(number_of_neurons, minibatch_size);
    layer->d_z = cuda_alloc_matrix(number_of_neurons, minibatch_size);
    layer->d_delta = cuda_alloc_matrix(number_of_neurons, minibatch_size);
    layer->d_weights = cuda_alloc_matrix(number_of_neurons, nneurons_previous_layer);
    layer->d_biases = cuda_alloc_matrix(number_of_neurons, 1);

    if (layer_number > 0)
    {
        init_weight(layer->d_weights, nneurons_previous_layer);
    }

    return layer;
}

void set_input(ann_t *nn, matrix_t *input)
{
    matrix_memcpy(nn->layers[0]->d_activations, input);
}

void print_layer(layer_t *layer)
{
    printf("-- neurons:%d, minibatch size:%d\n", layer->number_of_neurons, layer->minibatch_size);

    printf(">> Weighted inputs --\n");
    cuda_print_matrix(layer->d_z, true);
    printf(">> Activations --\n");
    cuda_print_matrix(layer->d_activations, true);

    printf(">> Weights --\n");
    cuda_print_matrix(layer->d_weights, true);
    printf(">> Biases --\n");
    cuda_print_matrix(layer->d_biases, true);

    printf(">> Delta --\n");
    cuda_print_matrix(layer->d_delta, true);
}

void print_nn(ann_t *nn)
{
    printf("ANN -- nlayers:%d, alpha:%lf, minibatch size: %d\n", nn->number_of_layers, __half2float(nn->alpha), nn->minibatch_size);
    for (uint16_t l = 0; l < nn->number_of_layers; l++)
    {
        printf("Layer %d ", l);
        print_layer(nn->layers[l]);
    }
}

void forward(ann_t *nn)
{
    const __half one = __float2half(1.0f);
    for (uint16_t l = 1; l < nn->number_of_layers; l++)
    {
        matrix_gemm(nn->layers[l]->d_biases, nn->d_one, nn->layers[l]->d_z);                                   // d_z^l <- b^l x 1
        matrix_gemm(nn->layers[l]->d_weights, nn->layers[l - 1]->d_activations, nn->layers[l]->d_z, one, one); // d_z^l <- w^l x a^(l-1) + d_z^l <=> d_z^l <- w^l x a^(l-1) + b^l x 1
        matrix_function(nn->layers[l]->d_z, false, nn->layers[l]->d_activations);                              // a^l = f(d_z^l)
    }
}

void backward(ann_t *nn, matrix_t *y)
{
    const uint16_t L = nn->number_of_layers - 1;

    matrix_t *d_dfzL = cuda_alloc_matrix(nn->layers[L]->number_of_neurons, nn->minibatch_size);

    matrix_minus(nn->layers[L]->d_activations, y, nn->layers[L]->d_delta);    // delta^(L) = (a^L - y)
    matrix_function(nn->layers[L]->d_z, true, d_dfzL);                        // f'(d_z^(L))
    hadamard_product(nn->layers[L]->d_delta, d_dfzL, nn->layers[L]->d_delta); // delta^(L) = (a^L - y) o f'(d_z^(L))

    cuda_free_matrix(d_dfzL);

    for (uint16_t l = L; l > 1; l--)
    {
        matrix_t *d_tw, *d_delta_tmp, *d_dfz;
        d_tw = cuda_alloc_matrix(nn->layers[l - 1]->number_of_neurons, nn->layers[l]->number_of_neurons);
        d_delta_tmp = cuda_alloc_matrix(nn->layers[l - 1]->number_of_neurons, nn->minibatch_size);
        d_dfz = cuda_alloc_matrix(nn->layers[l - 1]->number_of_neurons, nn->minibatch_size);

        matrix_transpose(nn->layers[l]->d_weights, d_tw);                 // (w^l)T
        matrix_gemm(d_tw, nn->layers[l]->d_delta, d_delta_tmp);           // (w^l)T x delta^l
        matrix_function(nn->layers[l - 1]->d_z, true, d_dfz);             // f'(d_z^(l-1))
        hadamard_product(d_delta_tmp, d_dfz, nn->layers[l - 1]->d_delta); // delta^(l-1) = (w^l)T x delta^l o f'(d_z^(l-1))

        cuda_free_matrix(d_tw);
        cuda_free_matrix(d_delta_tmp);
        cuda_free_matrix(d_dfz);
    }

    for (uint16_t l = 1; l < nn->number_of_layers; l++)
    {
        __half one = __float2half(1.0f);
        matrix_t *d_ta;
        d_ta = cuda_alloc_matrix(nn->minibatch_size, nn->layers[l - 1]->number_of_neurons);

        matrix_transpose(nn->layers[l - 1]->d_activations, d_ta);                                                                 // ta <- (a^(l-1))^T
        matrix_gemm(nn->layers[l]->d_delta, d_ta, nn->layers[l]->d_weights, -nn->alpha / __double2half(nn->minibatch_size), one); // w^l <- w^l - alpha /m . delta^l x (a^(l-1))^T

        cuda_free_matrix(d_ta);

        matrix_gemm(nn->layers[l]->d_delta, nn->d_oneT, nn->layers[l]->d_biases, -nn->alpha / __double2half(nn->minibatch_size), one);
    }
}