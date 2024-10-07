#include <torch/extension.h>
// #include "utils.h"

void distance_kernel_wrapper(int n, int m,
                             int c,
                             const float *src,
                             const float *dst,
                             float *dist);


at::Tensor distance_l1(at::Tensor src, at::Tensor dst) {
    TORCH_CHECK(src.is_contiguous());
    TORCH_CHECK(dst.is_contiguous());
    TORCH_CHECK(src.scalar_type() == at::ScalarType::Float);
    TORCH_CHECK(dst.scalar_type() == at::ScalarType::Float);

    if (dst.type().is_cuda()) {
        TORCH_CHECK(src.device().is_cuda());
    }

    at::Tensor dist = torch::zeros(
        {src.size(0), dst.size(0)},
        at::device(src.device()).dtype(at::ScalarType::Float)
    );

    if (dst.type().is_cuda()) {
        distance_kernel_wrapper(src.size(0), dst.size(0),
                                src.size(1),
                                src.data<float>(),
                                dst.data<float>(),
                                dist.data<float>());
    } else {
        TORCH_CHECK(false, "CPU not supported");
    }

    return dist;
}

