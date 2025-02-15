#ifndef MY_LAYERS_FEEDFORWARD_HPP_
#define MY_LAYERS_FEEDFORWARD_HPP_

#include "common/tensor.hpp"
#include <vector>

struct FFParam {
    Tensor<float> W1;
    std::vector<float> b1;
    Tensor<float> W2;
    std::vector<float> b2;
};

Tensor<float> feed_forward(const Tensor<float> &x, const FFParam &param);

#endif // MY_LAYERS_FEEDFORWARD_HPP_