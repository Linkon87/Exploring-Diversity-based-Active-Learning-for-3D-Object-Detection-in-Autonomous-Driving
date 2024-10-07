#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "cuda_utils.h"



__global__ void L1DistanceKernel(
    int n, int m,
    int c,
    const float *__restrict__ src,
    const float *__restrict__ dst,
    float *__restrict__ dist
) {
    // get the if of current thread
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * m) return;

    int srcIdx = idx / m;
    int dstIdx = idx % m;

    for (int k = 0; k < c; ++k) {
        float d = src[srcIdx * c + k] - dst[dstIdx * c + k];
        if (d < 0) { d = -d; }
        dist[srcIdx * m + dstIdx] += d;
    }
}


__global__ void L2DistanceKernel(
    int n, int m,
    int c,
    const float *__restrict__ src,
    const float *__restrict__ dst,
    float *__restrict__ dist
) {
    // get the if of current thread
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * m) return;

    int srcIdx = idx / m;
    int dstIdx = idx % m;

    for (int k = 0; k < c; ++k) {
        float d = src[srcIdx * c + k] - dst[dstIdx * c + k];
        d = d * d;
        dist[srcIdx * m + dstIdx] += d;
    }
}

void L1DistanceKernelWrapper(
    int n,
    int m,
    int c,
    const float *src,
    const float *dst,
    float *dist
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    int n_threads = opt_n_threads(n * m);
	int block_count = (n * m) / n_threads + 1;

    L1DistanceKernel<<<block_count, n_threads, 0, stream>>>(
        n, m, c, src, dst, dist);

    CUDA_CHECK_ERRORS();
}

void L2DistanceKernelWrapper(
    int n,
    int m,
    int c,
    const float *src,
    const float *dst,
    float *dist
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    int n_threads = opt_n_threads(n * m);
	int block_count = (n * m) / n_threads + 1;

    L2DistanceKernel<<<block_count, n_threads, 0, stream>>>(
        n, m, c, src, dst, dist);

    CUDA_CHECK_ERRORS();
}


__global__ void fastL1DistanceKernel(
    int n, int m,
    int c,
    const float *__restrict__ src,
    const float *__restrict__ dst,
    float *__restrict__ dist
) {
    // get the if of current thread
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * m * c) return;

    int srcIdx = idx / (c * m);
    int dstIdx = (idx / c) % m;
    int featIdx = idx % c;

    float d = src[srcIdx * c + featIdx] - dst[dstIdx * c + featIdx];
    if (d < 0) { d = -d; }
    atomicAdd(dist + (idx / c), d); // TODO: too slow
}


__global__ void fastL2DistanceKernel(
    int n, int m,
    int c,
    const float *__restrict__ src,
    const float *__restrict__ dst,
    float *__restrict__ dist
) {
    // get the if of current thread
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * m * c) return;

    int srcIdx = idx / (c * m);
    int dstIdx = (idx / c) % m;
    int featIdx = idx % c;

    float d = src[srcIdx * c + featIdx] - dst[dstIdx * c + featIdx];
    d = d * d;
    atomicAdd(dist + (idx / c), d); // TODO: too slow
}



void fastL1DistanceKernelWrapper(
    int n,
    int m,
    int c,
    const float *src,
    const float *dst,
    float *dist
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    int n_threads = opt_n_threads(n * m * c);
	int block_count = (n * m * c) / n_threads + 1;

    fastL1DistanceKernel<<<block_count, n_threads, 0, stream>>>(
        n, m, c, src, dst, dist);

    CUDA_CHECK_ERRORS();
}


void fastL2DistanceKernelWrapper(
    int n,
    int m,
    int c,
    const float *src,
    const float *dst,
    float *dist
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    int n_threads = opt_n_threads(n * m * c);
	int block_count = (n * m * c) / n_threads + 1;

    fastL2DistanceKernel<<<block_count, n_threads, 0, stream>>>(
        n, m, c, src, dst, dist);

    CUDA_CHECK_ERRORS();
}


