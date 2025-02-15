#ifndef __LINEAR_HPP__
#define __LINEAR_HPP__

#include "common/tensor.hpp"
#include <vector>

struct LinearParam
{
    Tensor<float> weight;
    std::vector<float> bias;
};

Tensor<float> linear(const Tensor<float> &input,
                     const LinearParam &param);

#endif