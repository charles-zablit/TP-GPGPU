#include "model_utils.h"

void zero_to_n(uint16_t n, uint16_t *t)
{
    for (uint16_t i = 0; i < n; i++)
    {
        t[i] = i;
    }
}

void shuffle(uint16_t *t, const uint16_t size, const uint16_t number_of_switch)
{
    zero_to_n(size, t);
    for (uint16_t i = 0; i < number_of_switch; i++)
    {
        uint16_t x = rand() % size;
        uint16_t y = rand() % size;
        uint16_t tmp = t[x];
        t[x] = t[y];
        t[y] = tmp;
    }
}

double accuracy(image *d_test_img, byte *d_test_label, byte *test_label, uint16_t datasize, uint16_t minibatch_size, ann_t *nn)
{
    uint16_t good = 0;
    uint16_t idx[datasize];
    __half *d_y;
    __half *pred = (__half *)malloc(10 * minibatch_size * sizeof(__half));

    cudaMalloc((void **)&d_y, 10 * minibatch_size * sizeof(__half));

    zero_to_n(datasize, idx);

    for (uint16_t i = 0; i < datasize - minibatch_size; i += minibatch_size)
    {
        gpuPopulateMinibatch(nn->layers[0]->d_activations->m, d_y, &idx[i], minibatch_size, d_test_img, 28 * 28, d_test_label, datasize);
        kernelRetchk;
        forward(nn);
        cudaMemcpy(pred, nn->layers[nn->number_of_layers - 1]->d_activations->m, 10 * minibatch_size * sizeof(__half), cudaMemcpyDeviceToHost);
        kernelRetchk;

        for (uint16_t col = 0; col < minibatch_size; col++)
        {
            uint16_t idxTrainingData = col + i;
            __half max = __float2half(0.0f);
            uint16_t idx_max = 0;
            for (uint16_t row = 0; row < 10; row++)
            {
                uint16_t idx = col + row * minibatch_size;
                if (pred[idx] > max)
                {
                    max = pred[idx];
                    idx_max = row;
                }
            }
            if (idx_max == test_label[idxTrainingData])
            {
                good++;
            }
        }
    }
    cudaFree(d_y);

    uint16_t ntests = (datasize / minibatch_size) * minibatch_size;
    return (100.0 * (double)(good) / ntests);
}

__global__ void populateX(__half *x, image *img, u_int16_t *minibatch_idx, u_int16_t minibatch_size, u_int16_t img_size)
{
    u_int16_t col = blockIdx.x * blockDim.x + threadIdx.x;
    u_int16_t row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < minibatch_size && row < img_size)
    {
        x[row * minibatch_size + col] = __int2half_rn(img[minibatch_idx[col]][row]) / __float2half(255.0f);
    }
}

__global__ void populateY(__half *y, byte *label, u_int16_t *minibatch_idx, u_int16_t minibatch_size, u_int16_t label_size)
{
    u_int16_t col = blockIdx.x * blockDim.x + threadIdx.x;
    u_int16_t row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < minibatch_size)
    {
        if (row < label_size)
        {
            y[row * minibatch_size + col] = __float2half(0.0f);
        }
        if (row == 0)
        {
            y[label[minibatch_idx[col]] * minibatch_size + col] = __float2half(1.0f);
        }
    }
}

void gpuPopulateMinibatch(__half *d_x, __half *d_y, u_int16_t *minibatch_idx, u_int16_t minibatch_size, image *d_img, u_int16_t img_size, byte *d_label, u_int16_t datasize)
{
    u_int16_t *d_minibatch_idx;

    cudaMalloc((void **)&d_minibatch_idx, minibatch_size * sizeof(u_int16_t));
    kernelRetchk;

    cudaMemcpy(d_minibatch_idx, minibatch_idx, minibatch_size * sizeof(u_int16_t), cudaMemcpyHostToDevice);
    kernelRetchk;

    dim3 threadsPerBlockX(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
    dim3 blocksPerGridX((minibatch_size + threadsPerBlockX.x - 1) / threadsPerBlockX.x,
                        (img_size + threadsPerBlockX.y - 1) / threadsPerBlockX.y);

    populateX<<<blocksPerGridX, threadsPerBlockX>>>(d_x, d_img, d_minibatch_idx, minibatch_size, img_size);
    kernelRetchk;

    dim3 threadsPerBlockY(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
    dim3 blocksPerGridY((minibatch_size + threadsPerBlockY.x - 1) / threadsPerBlockY.x,
                        (10 + threadsPerBlockY.y - 1) / threadsPerBlockY.y);

    populateY<<<blocksPerGridY, threadsPerBlockY>>>(d_y, d_label, d_minibatch_idx, minibatch_size, 10);
    kernelRetchk;

    cudaFree(d_minibatch_idx);
}