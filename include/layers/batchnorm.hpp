#ifndef __BN_HPP__
#define __BN_HPP__

#include "common/tensor.hpp"
#include <vector>

struct BNParam {
    std::vector<float> gamma;
    std::vector<float> beta;
    std::vector<float> running_mean;
    std::vector<float> running_var;
    float eps = 1e-5f;
};

Tensor<float> batchnorm2d(const Tensor<float> &input, const BNParam &param);

#endif