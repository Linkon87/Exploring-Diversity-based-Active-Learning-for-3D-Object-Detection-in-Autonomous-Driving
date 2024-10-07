
#include "cdist.h"
#include "utils.h"

void L1DistanceKernelWrapper(
    int n,
    int m,
    int c,
    const float *src,
    const float *dst,
    float *dist);

void L2DistanceKernelWrapper(
    int n,
    int m,
    int c,
    const float *src,
    const float *dst,
    float *dist);

void fastL1DistanceKernelWrapper(
    int n,
    int m,
    int c,
    const float *src,
    const float *dst,
    float *dist);

void fastL2DistanceKernelWrapper(
    int n,
    int m,
    int c,
    const float *src,
    const float *dst,
    float *dist);


at::Tensor distance(at::Tensor src, at::Tensor dst, distanceType type) {
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
        switch (type){
            case L1:
                L1DistanceKernelWrapper(
                    src.size(0),
                    dst.size(0),
                    src.size(1),
                    src.data<float>(),
                    dst.data<float>(),
                    dist.data<float>());
                break;
            case L2:
                L2DistanceKernelWrapper(
                    src.size(0),
                    dst.size(0),
                    src.size(1),
                    src.data<float>(),
                    dst.data<float>(),
                    dist.data<float>());
                break;
            default:
                throw "Not Implementation";
        }
    } else {
        // TORCH_CHECK(false, "CPU not supported");
        int n = src.size(0);
        int m = dst.size(0);
        int c = src.size(1);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                for (int k = 0; k < c; ++k) {
                    float d = (src.data<float>()[i * c + k] - dst.data<float>()[j * c + k]);
                    switch (type){
                        case L1:
                            if (d < 0) { d = -d; };
                            break;
                        case L2:
                            d = d * d;
                            break;
                        default:
                            throw "Not Implementation";
                    }
                    dist.data<float>()[i * m + j] += d;
                }
            }
        }
    }
    return dist;
}


at::Tensor fastDistance(at::Tensor src, at::Tensor dst, distanceType type) {
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

        switch (type){
            case L1:
                fastL1DistanceKernelWrapper(
                    src.size(0),
                    dst.size(0),
                    src.size(1),
                    src.data<float>(),
                    dst.data<float>(),
                    dist.data<float>());
                break;
            case L2:
                fastL2DistanceKernelWrapper(
                    src.size(0),
                    dst.size(0),
                    src.size(1),
                    src.data<float>(),
                    dst.data<float>(),
                    dist.data<float>());
                break;
            default:
                throw "Not Implementation";
        }
    } else {
        // TORCH_CHECK(false, "CPU not supported");
        int n = src.size(0);
        int m = dst.size(0);
        int c = src.size(1);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                for (int k = 0; k < c; ++k) {
                    float d = (src.data<float>()[i * c + k] - dst.data<float>()[j * c + k]);
                    switch (type){
                        case L1:
                            if (d < 0) { d = -d; };
                            break;
                        case L2:
                            d = d * d;
                            break;
                        default:
                            throw "Not Implementation";
                    }
                    dist.data<float>()[i * m + j] += d;
                }
            }
        }
    }
    return dist;
}

