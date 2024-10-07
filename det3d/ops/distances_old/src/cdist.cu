#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "cuda_utils.h"

__global__ void distance_kernel(int n, int m,
                                int c,
                                const float *__restrict__ src,
                                const float *__restrict__ dst,
                                float *__restrict__ dist) {
    int feat_index = blockIdx.x;
    src += n * feat_index
    dst += m * feat_index

    dist += 0

    int index = threadIdx.x;
    int stride = blockDim.x;

    // parallel feature dim
    for (int k = index, k < c; k += stride) {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                float d = (src[i * c + k] - dst[j * c + k]);
                if (d > 0) {
                    dist[i * m + j] += d;
                } else {
                    dist[i * m + j] -= d;
                }
            }
        }
    }
}


void distance_kernel_wrapper(int n, int m,
                             int c,
                             const float *src,
                             const float *dst,
                             float *dist) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    distance_kernel<<<c, opt_n_threads(m), 0, stream>>>(
        n, m, c, src, dst, dist);

    CUDA_CHECK_ERRORS();
}

