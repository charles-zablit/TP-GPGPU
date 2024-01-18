#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <time.h>

#include "mnist.h"
#include "matrix.h"
#include "ann.h"
#include "log.h"
#include "tests.h"
#include "helper_cuda.h"
#include "model_utils.h"

int main(int argc, char *argv[])
{
#ifdef DEBUG
    setvbuf(stdout, NULL, _IOLBF, 0);
    run_tests();
    return;
#endif
    srand(0);
    log_debug("Starting program");
    uint16_t datasize, ntest;
    log_debug("Reading files");
    image *train_img = read_images("/home/charles/Developer/TP-GPGPU/train-images-idx3-ubyte", &datasize);
    byte *train_label = read_labels("/home/charles/Developer/TP-GPGPU/train-labels-idx1-ubyte", &datasize);
    image *test_img = read_images("/home/charles/Developer/TP-GPGPU/t10k-images-idx3-ubyte", &ntest);
    byte *test_label = read_labels("/home/charles/Developer/TP-GPGPU/t10k-labels-idx1-ubyte", &ntest);
    log_debug("Done reading files");

    image *d_train_img;
    byte *d_train_label;
    image *d_test_img;
    byte *d_test_label;
    cudaMalloc((void **)&d_train_img, sizeof(image) * datasize);
    cudaMalloc((void **)&d_train_label, sizeof(byte) * datasize);
    cudaMemcpy(d_train_img, train_img, sizeof(image) * datasize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_train_label, train_label, sizeof(byte) * datasize, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&d_test_img, sizeof(image) * ntest);
    cudaMalloc((void **)&d_test_label, sizeof(byte) * ntest);
    cudaMemcpy(d_test_img, test_img, sizeof(image) * ntest, cudaMemcpyHostToDevice);
    cudaMemcpy(d_test_label, test_label, sizeof(byte) * ntest, cudaMemcpyHostToDevice);

    log_debug("Creating neural network");
    ann_t *nn;
    __half alpha = __float2half(0.05);
    uint16_t minibatch_size = 16;
    uint16_t number_of_layers = 3;
    uint16_t nneurons_per_layer[3] = {28 * 28, 30, 10};
    nn = create_ann(alpha, minibatch_size, number_of_layers, nneurons_per_layer);
#ifdef DEBUG
    print_nn(nn);
#endif

    log_info("starting accuracy %lf", accuracy(d_test_img, d_test_label, test_label, ntest, minibatch_size, nn));

    uint16_t *shuffled_idx = (uint16_t *)malloc(datasize * sizeof(uint16_t));
    __half *x = (__half *)malloc(28 * 28 * minibatch_size * sizeof(__half));
    __half *y = (__half *)malloc(10 * minibatch_size * sizeof(__half));
    matrix_t *out = cuda_alloc_matrix(10, minibatch_size);

    for (int epoch = 0; epoch < 40; epoch++)
    {
        log_info("start learning epoch %d", epoch);

        shuffle(shuffled_idx, datasize, datasize);

        for (int i = 0; i < datasize - minibatch_size; i += minibatch_size)
        {
            gpuPopulateMinibatch(nn->layers[0]->d_activations->m, out->m, shuffled_idx + i, minibatch_size, d_train_img, 28 * 28, d_train_label, datasize);
            forward(nn);
            backward(nn, out);
        }
        log_info("epoch %d accuracy %lf", epoch, accuracy(d_test_img, d_test_label, test_label, ntest, minibatch_size, nn));
    }
    log_info("ending accuracy %lf", accuracy(d_test_img, d_test_label, test_label, ntest, minibatch_size, nn));

    free(x);
    free(y);
    free(shuffled_idx);
    cuda_free_matrix(out);

    return 0;
}
