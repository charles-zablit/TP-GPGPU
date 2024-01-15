#include "model_utils.h"

void zero_to_n(unsigned n, unsigned *t)
{
    for (unsigned i = 0; i < n; i++)
    {
        t[i] = i;
    }
}

void shuffle(unsigned *t, const unsigned size, const unsigned number_of_switch)
{
    zero_to_n(size, t);
    for (unsigned i = 0; i < number_of_switch; i++)
    {
        unsigned x = rand() % size;
        unsigned y = rand() % size;
        unsigned tmp = t[x];
        t[x] = t[y];
        t[y] = tmp;
    }
}

double accuracy(image *d_test_img, byte *d_test_label, byte *test_label, unsigned datasize, unsigned minibatch_size, ann_t *nn)
{
    unsigned good = 0;
    unsigned idx[datasize];
    double *d_y;
    double *pred = (double *)malloc(10 * minibatch_size * sizeof(double));

    cudaMalloc((void **)&d_y, 10 * minibatch_size * sizeof(double));

    zero_to_n(datasize, idx);

    for (int i = 0; i < datasize - minibatch_size; i += minibatch_size)
    {
        gpuPopulateMinibatch(nn->layers[0]->d_activations->m, d_y, &idx[i], minibatch_size, d_test_img, 28 * 28, d_test_label, datasize);
        kernelRetchk;
        forward(nn);
        cudaMemcpy(pred, nn->layers[nn->number_of_layers - 1]->d_activations->m, 10 * minibatch_size * sizeof(double), cudaMemcpyDeviceToHost);
        kernelRetchk;

        for (int col = 0; col < minibatch_size; col++)
        {
            int idxTrainingData = col + i;
            double max = 0;
            unsigned idx_max = 0;
            for (int row = 0; row < 10; row++)
            {
                int idx = col + row * minibatch_size;
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

    unsigned ntests = (datasize / minibatch_size) * minibatch_size;
    return (100.0 * (double)(good) / ntests);
}

__global__ void populateX(double *x, image *img, unsigned *minibatch_idx, unsigned minibatch_size, unsigned img_size)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < minibatch_size && row < img_size)
    {
        x[row * minibatch_size + col] = (double)img[minibatch_idx[col]][row] / 255.0;
    }
}

__global__ void populateY(double *y, byte *label, unsigned *minibatch_idx, unsigned minibatch_size, unsigned label_size)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < minibatch_size)
    {
        if (row < label_size)
        {
            y[row * minibatch_size + col] = 0.0;
        }
        if (row == 0)
        {
            y[label[minibatch_idx[col]] * minibatch_size + col] = 1.0;
        }
    }
}

void gpuPopulateMinibatch(double *d_x, double *d_y, unsigned *minibatch_idx, unsigned minibatch_size, image *d_img, unsigned img_size, byte *d_label, unsigned datasize)
{
    unsigned *d_minibatch_idx;

    cudaMalloc((void **)&d_minibatch_idx, minibatch_size * sizeof(unsigned));
    kernelRetchk;

    cudaMemcpy(d_minibatch_idx, minibatch_idx, minibatch_size * sizeof(unsigned), cudaMemcpyHostToDevice);
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