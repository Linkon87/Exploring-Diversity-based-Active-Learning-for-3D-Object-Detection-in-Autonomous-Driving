#pragma once
#include <stdio.h>
#include <vector>
#include <string>
#include <torch/extension.h>

using TensorVec = std::vector<at::Tensor>;
using IntVec = std::vector<int>;
using IntVecVec = std::vector<IntVec>;

enum distanceType {L1, L2};

at::Tensor distance(at::Tensor src, at::Tensor dst, distanceType type = L1);
at::Tensor fastDistance(at::Tensor src, at::Tensor dst, distanceType type = L1);

