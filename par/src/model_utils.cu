#include "model_utils.h"

void populate_minibatch(double *x, double *y, unsigned *minibatch_idx, unsigned minibatch_size, image *img, unsigned img_size, byte *label, unsigned label_size)
{
    for (int col = 0; col < minibatch_size; col++)
    {
        for (int row = 0; row < img_size; row++)
        {
            x[row * minibatch_size + col] = (double)img[minibatch_idx[col]][row] / 255.;
        }

        for (int row = 0; row < 10; row++)
        {
            y[row * minibatch_size + col] = 0.0;
        }

        y[label[minibatch_idx[col]] * minibatch_size + col] = 1.0;
    }
}

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

double accuracy(image *test_img, byte *test_label, unsigned datasize, unsigned minibatch_size, ann_t *nn)
{
    unsigned good = 0;
    unsigned idx[datasize];
    double *x = (double *)malloc(28 * 28 * minibatch_size * sizeof(double));
    double *y = (double *)malloc(10 * minibatch_size * sizeof(double));
    double *pred = (double *)malloc(10 * minibatch_size * sizeof(double));

    zero_to_n(datasize, idx);

    for (int i = 0; i < datasize - minibatch_size; i += minibatch_size)
    {
        populate_minibatch(x, y, &idx[i], minibatch_size, test_img, 28 * 28, test_label, 10);
        cudaMemcpy(nn->layers[0]->d_activations->m, x, 28 * 28 * minibatch_size * sizeof(double), cudaMemcpyHostToDevice);
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
    free(x);
    free(y);

    unsigned ntests = (datasize / minibatch_size) * minibatch_size;
    return (100.0 * (double)(good) / ntests);
}